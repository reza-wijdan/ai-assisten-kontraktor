from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class QueryRequest(BaseModel):
    user_id: Optional[int]
    message: str

class QueryResponse(BaseModel):
    intent: str
    answer: str
    meta: Optional[dict] = None

class EquipmentOut(BaseModel):
    id: int
    name: str
    price: float
    image_url: Optional[str]
    description: Optional[str]
    category: Optional[str]
    stock: int
    available_stock: int
    manufacturer: Optional[str]
    model_number: Optional[str]
    warranty_months: Optional[int]
    weight: Optional[float]
    dimensions: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True
