from fastapi import FastAPI
from .routers import assistant
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="AI Assistant Konstruksi (read-only)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # ganti dengan origin frontend kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# DO NOT create tables here; service is read-only in production
app.include_router(assistant.router, prefix="/assistant", tags=["assistant"])

@app.get("/health")
def health():
    return {"status": "ok"}
