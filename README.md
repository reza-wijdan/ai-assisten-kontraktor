Cara deploy & jalankan (singkat)

Copy proyek ke server / mesin development.
Buat virtualenv (opsional) dan install deps:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Copy .env.example → .env dan isi DATABASE_URL

Jalankan:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


POST http://127.0.0.1:8000/assistant/query
Body: {"user_id":1, "message":"min, stok truk ada berapa?"}