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
    "truk", "dump truck", "dumptruck", "excavator", "excavator", "eksavator",
    "bulldozer", "buldoser", "crane", "crawler crane", "road roller", "roller",
    "forklift", "grader", "loader"
]

def detect_type_from_text(text: str):
    t = preprocess(text)
    for tp in TYPE_KEYWORDS:
        if tp in t:
            return tp
    return None

def find_equipment_by_name(db: Session, query: str, limit: int = 10):
    q = preprocess(query)
    # coba ilike dulu
    rows = db.query(Equipment).filter(Equipment.name.ilike(f"%{q}%")).limit(limit).all()
    if rows:
        return rows
    # fallback fuzzy matching dengan process.extract
    all_rows = db.query(Equipment).all()
    choices = {r.name: r for r in all_rows}
    # dapatkan top limit hasil yang mirip dengan threshold 60 (bisa disesuaikan)
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
