from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from .kb_loader import load_kb
from .config import EMBEDDING_MODEL

KB_ENTRIES = load_kb()

# fallback minimal if CSV kosong
if not KB_ENTRIES:
    KB_ENTRIES = [
        {"text": "Berapa stok truk yang tersedia?", "intent": "check_stock"},
        {"text": "stok truk masih ada ga", "intent": "check_stock"},
        {"text": "Berapa harga excavator?", "intent": "ask_price"},
        {"text": "Saya mau booking buldoser", "intent": "booking"},
    ]

MODEL = SentenceTransformer(EMBEDDING_MODEL)
KB_TEXTS = [e["text"] for e in KB_ENTRIES]
KB_INTENTS = [e["intent"] for e in KB_ENTRIES]

EMBS = MODEL.encode(KB_TEXTS, convert_to_numpy=True)
# normalize for cosine (use inner product on normalized vectors)
faiss.normalize_L2(EMBS)
DIM = EMBS.shape[1]
INDEX = faiss.IndexFlatIP(DIM)
INDEX.add(EMBS)
