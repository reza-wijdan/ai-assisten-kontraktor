from .embeddings import MODEL, INDEX, KB_INTENTS, KB_TEXTS
import faiss
import numpy as np
import re
from rapidfuzz import fuzz

STOCK_KEYWORDS = [
    # Bahasa Indonesia formal
    "stok", "tersedia", "ketersediaan", "jumlah", "sisa", "ready", "ada", "masih ada",
    # Bahasa santai / gaul
    "stoknya", "readykah", "masih ready", "ready ga", "ready nggak", "ada nggak",
    # Bahasa Inggris
    "stock", "available", "availability", "in stock", "have", "left", "remain"
]

PRICE_KEYWORDS = [
    # Bahasa Indonesia formal
    "harga", "biaya", "tarif", "ongkos", "bayaran", "total harga", "total biaya", "biayanya",
    # Bahasa santai / gaul
    "harga berapa", "berapa harganya", "harganya", "harga nya", "harga brp", "harga dong", "harga min",
    # Bahasa Inggris
    "price", "cost", "fee", "charge", "how much", "rate", "pricing", "worth"
]

BOOKING_KEYWORDS = [
    # Bahasa Indonesia formal
    "booking", "pesan", "pemesanan", "rental", "reservasi", "ambil",
    # Bahasa santai / gaul
    "sewain", "nyewa", "nyewa dong", "pesan dong", "pesen", "pengen sewa", "pengen booking", "sewain ga", "bisa booking", "pesanan", "memesan", "pemesan"
    # Bahasa Inggris
    "rent", "book", "reserve", "order", "take", "get", "renting", "hire", "lease"
]

CLOSING_KEYWORDS = [
    # Formal & umum
    "baiklah", "okey", "oke", "ok", "ya sudah", "baik", "sip", "mantap", "setuju", "makasih", "terimakasih min", "terima kasih",
    # Bahasa gaul
    "okedeh", "okelah", "yoi", "cus", "gas", "gaskan", "lanjutkan", "siap", "siapp", "sip lah",
    # Campur Inggris
    "alright", "okay", "okey", "okayy", "fine", "deal", "sounds good", "go ahead"
]

CLOSING_CONFIRMATION_KEYWORDS = [
    "sudah", "enggak", "tidak", "nggak", "ga", "gak", "cukup", "oke", "oke deh", "tidak jadi",
    "udah", "stop", "berhenti", "kelar"
]

COMPLAINT_KEYWORDS = [
    "rusak", "bermasalah", "error", "tidak berfungsi", "gagal", "kerusakan", "komplain",
    "problem", "issue", "gangguan", "tidak bisa", "keluhan", "laporan",
    "belum sampai", "lama", "ditunda", "kapan datang", "kapan sampai",
    "mogok", "macet", "ngadat"
]

GREETING_KEYWORDS = [
    # Salam pembuka formal dan santai
    "halo", "hai", "hallo", "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "hey", "hi", "hello", "apa kabar", "apa kabarnya"
]

PRICE_SEWA_PATTERNS = [
    "berapa sewa", "sewa berapa", "biaya sewa", "tarif sewa", "harga sewa"
]

def normalize_repeated_chars(text: str) -> str:
    # Ganti huruf berulang 2x+ menjadi satu huruf
    return re.sub(r'(.)\1{2,}', r'\1', text)

def preprocess(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^0-9a-zA-Z\u00C0-\u017F\s]", " ", t)
    t = normalize_repeated_chars(t)
    t = re.sub(r"\s+", " ", t)
    return t

def semantic_match(text: str, top_k: int = 3):
    v = MODEL.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(v)
    D, I = INDEX.search(v, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({"score": float(score), "intent": KB_INTENTS[idx], "example": KB_TEXTS[idx]})
    return results

def recognize_intent(text: str, threshold: float = 0.55):
    # 1) keyword (fast)
    kw = keyword_intent(text)
    if kw:
        return {"intent": kw, "source": "keyword", "score": None}

    # 2) semantic
    sem = semantic_match(text, top_k=3)
    if sem:
        best = sem[0]
        if best["score"] >= threshold:
            return {"intent": best["intent"], "source": "semantic", "score": best["score"], "example": best.get("example")}
        # low confidence semantic -> still return best with low flag
        return {"intent": best["intent"], "source": "semantic_low", "score": best["score"], "example": best.get("example")}

    return {"intent": "unknown", "source": "none", "score": None}


def keyword_intent(text: str):
    t = preprocess(text)
    words = t.split()

    # 1️⃣ Cek price_sewa dulu (supaya "berapa sewa" tidak nyangkut di booking)
    for w in words:
        for kw in PRICE_SEWA_PATTERNS:
            if fuzz.ratio(w, kw) >= 80:
                return "price_sewa"

    for kw in PRICE_SEWA_PATTERNS:
        if kw in t:
            return "price_sewa"

    # 2️⃣ Baru cek ask_price biasa
    for w in words:
        for kw in PRICE_KEYWORDS:
            if fuzz.ratio(w, kw) >= 80:
                return "ask_price"
    for kw in PRICE_KEYWORDS:
        if kw in t:
            return "ask_price"

    # 3️⃣ Lalu cek booking
    for w in words:
        for kw in BOOKING_KEYWORDS:
            if fuzz.ratio(w, kw) >= 80:
                return "booking"
    for kw in BOOKING_KEYWORDS:
        if kw in t:
            return "booking"

    # 4️⃣ Sisanya cek intent lain seperti stok, closing, dsb
    for w in words:
        for kw in STOCK_KEYWORDS:
            if fuzz.ratio(w, kw) >= 80:
                return "check_stock"
    for kw in STOCK_KEYWORDS:
        if kw in t:
            return "check_stock"

    for w in words:
        for kw in CLOSING_KEYWORDS:
            if fuzz.ratio(w, kw) >= 80:
                return "closing_keyword"
    for kw in CLOSING_KEYWORDS:
        if kw in t:
            return "closing_keyword"

    for w in words:
        for kw in CLOSING_CONFIRMATION_KEYWORDS:
            if fuzz.ratio(w, kw) >= 80:
                return "closing_confirmation"
    for kw in CLOSING_CONFIRMATION_KEYWORDS:
        if kw in t:
            return "closing_confirmation"

    for w in words:
        for kw in COMPLAINT_KEYWORDS:
            if fuzz.ratio(w, kw) >= 80:
                return "complaint_keyword"
    for kw in COMPLAINT_KEYWORDS:
        if kw in t:
            return "complaint_keyword"

    for w in words:
        for kw in GREETING_KEYWORDS:
            if fuzz.ratio(w, kw) >= 80:
                return "greeting"
    for kw in GREETING_KEYWORDS:
        if kw in t:
            return "greeting"

    return None
