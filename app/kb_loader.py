import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "training_kb.csv")

def load_kb(path: str = CSV_PATH):
    items = []
    if not os.path.exists(path):
        return items
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        intent = str(row.get("intent", "")).strip()
        if q and intent:
            items.append({"text": q, "intent": intent})
    return items
