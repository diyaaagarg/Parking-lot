from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db.models import User, ChatLog
from src.api.auth import get_current_user
from src.rag.grounded_advisory import answer_grounded_advisory_query
from src.rag.agentic_booking import execute_agentic_booking_intent
from src.rag.report_generator import generate_operator_summary_report

router = APIRouter(prefix="/rag", tags=["GenAI & RAG Advisory"])

class AskQueryRequest(BaseModel):
    query: str

class AgenticBookRequest(BaseModel):
    query: str

@router.post("/ask")
def ask_rag_advisory(req: AskQueryRequest, db: Session = Depends(get_db)):
    result = answer_grounded_advisory_query(db, req.query)
    
    # Save chat log
    chat = ChatLog(user_query=req.query, ai_response=result["answer"])
    db.add(chat)
    db.commit()
    
    return result

@router.post("/agentic-book")
def agentic_book(
    req: AgenticBookRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    result = execute_agentic_booking_intent(db, current_user.id, req.query)
    
    # Save chat log
    chat = ChatLog(user_id=current_user.id, user_query=req.query, ai_response=result["message"])
    db.add(chat)
    db.commit()

    return result

@router.get("/operator-report")
def get_operator_report(db: Session = Depends(get_db)):
    report = generate_operator_summary_report(db)
    return report
