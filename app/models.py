from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Enum, func
from .database import Base
import datetime
import enum

class SenderEnum(str, enum.Enum):
    user = "user"
    ai = "ai"

class Equipment(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    price = Column(Float, default=0.0)
    image_url = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String(128), nullable=True)
    stock = Column(Integer, default=0)
    available_stock = Column(Integer, default=0)
    manufacturer = Column(String(128), nullable=True)
    model_number = Column(String(128), nullable=True)
    warranty_months = Column(Integer, default=0)
    weight = Column(Float, default=0.0)
    dimensions = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True)  # Bisa pakai session ID atau login user ID
    message = Column(Text, nullable=False)
    sender = Column(Enum(SenderEnum), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
