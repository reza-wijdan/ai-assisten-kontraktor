from fastapi import APIRouter, Depends
from ..schemas import QueryRequest, QueryResponse
from ..intent_recognizer import recognize_intent, plot_rf_boundary
from ..database import SessionLocal
from ..utils import find_equipment_by_name, aggregate_stock, LIST_ALL_KEYWORDS, preprocess, fuzzy_find_equipment
from sqlalchemy.orm import Session
from ..services.conversation import save_message, get_recent_history
from ..models import SenderEnum
from rapidfuzz import fuzz

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Fungsi untuk cek kemiripan fuzzy dengan threshold
def contains_fuzzy_keyword(text, keywords, threshold=80):
    for kw in keywords:
        score = fuzz.partial_ratio(kw, text)
        if score >= threshold:
            return True
    return False

@router.post("/query", response_model=QueryResponse)
def chat_endpoint(req: QueryRequest, db: Session = Depends(get_db)):
    user_id = req.user_id if hasattr(req, "user_id") else "anonymous"
    user_text = req.message.lower().strip()
    
    # Simpan pertanyaan user
    save_message(db, user_id, user_text, SenderEnum.user)

    # Ambil percakapan terakhir
    history = get_recent_history(db, user_id, limit=5)

    # Intent detection
    intent_info = recognize_intent(user_text)
    intent = intent_info.get("intent", "unknown")
    meta = {"source": intent_info.get("source"), "score": intent_info.get("score")}

    # Cek pesan AI terakhir apakah berupa closing_keyword
    last_ai_message = None
    last_ai_intent = None
    for h in reversed(history):
        if h.sender == SenderEnum.ai:
            last_ai_message = h.message.lower()
            if "ada lagi yang bisa saya bantu" in last_ai_message:
                last_ai_intent = "closing_keyword"
            break

    # Tangani closing confirmation
    if last_ai_intent == "closing_keyword" and intent == "closing_confirmation":
        answer = "Terima kasih sudah menggunakan layanan kami. Semoga harimu menyenangkan!"
        save_message(db, user_id, answer, SenderEnum.ai)
        return {
            "intent": "final_closing",
            "answer": answer,
            "meta": {"source": "closing_confirmation"},
            "show_order_form": False
        }

    if contains_fuzzy_keyword(user_text, LIST_ALL_KEYWORDS, threshold=80):
        all_equipments = find_equipment_by_name(db, "", limit=50)  # Ambil semua data
        if all_equipments:
            lines = [
                f"{e.name} — stok: {e.available_stock or e.stock} unit"
                for e in all_equipments
            ]
            answer = "Berikut semua alat yang tersedia:\n" + "\n".join(lines)
        else:
            answer = "Saat ini belum ada data alat yang tersedia."
        
        save_message(db, user_id, answer, SenderEnum.ai)
        return {
            "intent": "list_all_equipment",
            "answer": answer,
            "meta": meta,
            "show_order_form": False
        }    

    # Cari produk yang dimaksud
    equipments = find_equipment_by_name(db, user_text, limit=10)

    # Kalau tidak ketemu, coba fuzzy
    if not equipments:
        equipments = fuzzy_find_equipment(db, user_text, limit=5)

    if not equipments:
        for h in history:
            if h.sender == SenderEnum.user:
                prev_equipments = find_equipment_by_name(db, h.message, limit=10)
                if prev_equipments:
                    equipments = prev_equipments
                    break

    # Logika tambahan untuk konten di luar konteks
    VALID_INTENTS = {"booking", "check_stock", "ask_price", "closing_keyword", "closing_confirmation", "complaint_keyword", "greeting", "price_sewa"}
    if intent == "unknown" and not equipments:
        answer = "Maaf, saya tidak mengerti maksud Anda."
        save_message(db, user_id, answer, SenderEnum.ai)
        return {
            "intent": "unknown_out_of_context",
            "answer": answer,
            "meta": meta,
            "show_order_form": False
        }

    
    if intent in ["check_stock", "ask_price", "price_sewa"] and not equipments:
        answer = "Mohon maaf, alat tersebut belum tersedia."
        save_message(db, user_id, answer, SenderEnum.ai)
        return {
            "intent": intent,
            "answer": answer,
            "meta": meta,
            "show_order_form": False
        }

    ORDER_KEYWORDS = ["pesan", "beli", "order", "booking", "mau ambil", "mau pesan"]
    show_order_form = False

    if intent == "booking":
        product_name = equipments[0].name if equipments else "alat yang dimaksud"
        answer = (f"Baik, saya akan bantu menyiapkan form pemesanan untuk {product_name}.\n"
                  "Tolong sebutkan: jumlah unit, tanggal mulai sewa, dan durasi.")
        show_order_form = True

    elif intent == "check_stock":
        if not equipments:
            answer = "Maaf, saya tidak menemukan alat kontraktor yang sesuai. Bisa sebutkan nama atau model alatnya?"
        elif len(equipments) == 1:
            e = equipments[0]
            avail = e.available_stock if getattr(e, "available_stock", None) is not None else e.stock
            answer = f"{e.name} — stok saat ini: {avail} unit."
        else:
            lines = [f"{e.name} — tersedia: {e.available_stock or e.stock} unit" for e in equipments[:6]]
            total = aggregate_stock(equipments)
            answer = "Berikut stok yang saya temukan:\n" + "\n".join(lines) + f"\nTotal (gabungan): {total} unit."

    elif intent in ["ask_price", "price_sewa"]:
        if not equipments:
            answer = "Sebutkan nama atau model alat kontraktor yang ingin dicek harganya, ya."
        else:
            lines = [
                f"{e.name} — harga: Rp {int(e.price):,} / bulan — stok: {e.available_stock or e.stock}"
                for e in equipments
            ]
            answer = "Harga yang saya temukan:\n" + "\n".join(lines)

    elif intent == "complaint_keyword":
        lateness_keywords = ["belum sampai", "lama", "ditunda", "kapan datang", "kapan sampai"]
        if contains_fuzzy_keyword(user_text, lateness_keywords):
            answer = (
                "Mohon maaf atas keterlambatan pengiriman alat kontraktor Anda. "
                "Kami akan segera cek status pengiriman dan mengabari Anda."
            )
        else:
            answer = (
                "Terima kasih telah melaporkan masalah pada alat kami. "
                "Tim teknis kami akan segera menindaklanjuti dan menghubungi Anda."
            )

    elif intent == "greeting":
        answer = "Selamat datang! Ada yang bisa saya bantu?"

    elif intent == "closing_keyword":
        answer = "Ada lagi yang bisa saya bantu?"

    elif intent == "closing_confirmation":
        answer = "Terima kasih sudah menggunakan layanan kami. Semoga harimu menyenangkan!"    

    elif intent == "unknown" and not equipments:
        answer = "Maaf, saya tidak mengerti maksud Anda."    

    else:
        answer = "Maaf, saya belum mengerti. Bisa jelaskan lebih detail?"

    save_message(db, user_id, answer, SenderEnum.ai)
    plot_rf_boundary()
    return {
        "intent": intent,
        "answer": answer,
        "meta": meta,
        "show_order_form": show_order_form
    }


