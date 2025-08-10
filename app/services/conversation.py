from sqlalchemy.orm import Session
from ..models import ConversationHistory, SenderEnum

def save_message(db: Session, user_id: str, message: str, sender: SenderEnum):
    history = ConversationHistory(user_id=user_id, message=message, sender=sender)
    db.add(history)
    db.commit()
    db.refresh(history)

def get_recent_history(db: Session, user_id: str, limit: int = 5):
    return (
        db.query(ConversationHistory)
        .filter(ConversationHistory.user_id == user_id)
        .order_by(ConversationHistory.created_at.desc())
        .limit(limit)
        .all()
    )
