from sqlalchemy.orm import Session
from .models import Equipment
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

def preprocess(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^0-9a-zA-Z\u00C0-\u017F\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

TYPE_KEYWORDS = [
    "truk", "dump truck", "dumptruck", "excavator", "eksavator",
    "bulldozer", "buldoser", "crane", "crawler crane", "road roller", "roller",
    "forklift", "grader", "loader"
]

LIST_ALL_KEYWORDS = [
    # Bentuk umum & formal
    "apa saja", "apa aja", "list", "daftar", "semua", "semuanya",
    "semua alat", "alat apa saja", "alat apa aja", "seluruh alat",
    "seluruhnya", "keseluruhan alat", "semua jenis alat", "jenis alat",
    "daftar alat", "list alat", "macam alat", "macam-macam alat",

    # Bentuk tambahan / informal
    "ada apa aja", "ada apa saja", "apa-apa aja", "apa-apa saja",
    "alatnya apa aja", "alatnya apa saja", "unit apa aja", "unit apa saja",
    "list semua", "list lengkap", "daftar lengkap", "list full", "full list",

    # Variasi bahasa campuran / singkatan
    "list tools", "tools apa aja", "tools apa saja", "equipment list",
    "equipment apa aja", "equipment apa saja", "alat berat apa aja",
    "alat berat apa saja", "list equipment", "unit tersedia", "semua unit",
    "unit yang ada", "stok alat", "stok semua"
]

def detect_type_from_text(text: str):
    t = preprocess(text)
    for tp in TYPE_KEYWORDS:
        if tp in t:
            return tp
    return None

def is_list_all_request(text: str):
    """Cek apakah user meminta daftar semua alat berat."""
    t = preprocess(text)
    for kw in LIST_ALL_KEYWORDS:
        if kw in t:
            return True
    return False

def get_all_equipment(db: Session):
    """Ambil semua alat berat dari database."""
    return db.query(Equipment).all()

def find_equipment_by_name(db: Session, query: str, limit: int = 10):
    q = preprocess(query)
    # coba ilike dulu
    rows = db.query(Equipment).filter(Equipment.name.ilike(f"%{q}%")).limit(limit).all()
    if rows:
        return rows
    # fallback fuzzy matching dengan process.extract
    all_rows = db.query(Equipment).all()
    choices = {r.name: r for r in all_rows}
    results = process.extract(q, choices.keys(), limit=limit, scorer=fuzz.token_set_ratio)
    matched = []
    for name, score in results:
        if score >= 60:
            matched.append(choices[name])
    return matched

def aggregate_stock(equipments):
    total = 0
    for e in equipments:
        if getattr(e, "available_stock", None) is not None:
            total += int(e.available_stock or 0)
        else:
            total += int(e.stock or 0)
    return total

def detect_type_from_text(value: str):
    """
    Mendeteksi tipe data dari string input.
    Urutan deteksi:
    1. Boolean (true/false)
    2. Integer
    3. Float
    4. String (default)
    """

    if not isinstance(value, str):
        return type(value).__name__

    text = value.strip()

    # 1. Boolean check
    if text.lower() in ["true", "false"]:
        return "bool"

    # 2. Integer check
    if re.fullmatch(r"[+-]?\d+", text):
        return "int"

    # 3. Float check (angka desimal, termasuk format .5 atau 5.)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)", text):
        return "float"

    # 4. Default: String
    return "str"