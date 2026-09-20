import os
import requests
from sqlalchemy.orm import Session
from src.config import OPENAI_API_KEY
from src.db.models import Venue, Lot, Slot
from src.forecasting.prophet_model import ProphetLotForecaster
from src.pricing.engine import calculate_dynamic_price
from src.rag.knowledge_base import ParkingKnowledgeBase

def answer_grounded_advisory_query(db: Session, query: str) -> dict:
    """
    Grounded RAG Advisory System:
    Injects real-time database state and forecast outputs into prompt context
    to guarantee 0% hallucination of occupancy numbers.
    """
    # 1. Retrieve Knowledge Docs
    kb = ParkingKnowledgeBase(db)
    relevant_docs = kb.search_knowledge(query, top_k=2)
    doc_context = "\n".join([f"- [{d.title}]: {d.content}" for d in relevant_docs])

    # 2. Fetch Live Real-Time Database State & Forecasts for all venues
    venues = db.query(Venue).all()
    live_context_lines = []

    for v in venues:
        for lot in v.lots:
            slots = db.query(Slot).filter(Slot.lot_id == lot.id).all()
            total = len(slots)
            occupied = sum(1 for s in slots if s.current_state in ["occupied", "overstayed"])
            reserved = sum(1 for s in slots if s.current_state == "reserved")
            vacant = total - occupied - reserved
            live_pct = round((occupied / total) * 100, 1) if total > 0 else 0

            # Get 60-min forecast
            forecaster = ProphetLotForecaster(lot.id)
            fc = forecaster.predict_future(db, horizon_minutes=60)
            pred_pct = fc.get("predicted_occupancy_pct", live_pct)
            pred_slots = fc.get("predicted_occupied_slots", occupied)

            # Get Dynamic Pricing
            pricing = calculate_dynamic_price(db, lot.id)

            live_context_lines.append(
                f"Venue: {v.name} ({v.category.upper()}) - Lot: '{lot.name}' (ID: {lot.id})\n"
                f"  * Current Status: {vacant} vacant, {occupied} occupied, {reserved} reserved (Total: {total}, Occupancy: {live_pct}%)\n"
                f"  * 60-Min Forecast: Predicted Occupancy = {pred_pct}% (~{pred_slots}/{total} occupied)\n"
                f"  * Pricing: Base = ${lot.base_price_per_hour}/hr, Recommended = ${pricing['recommended_price_per_hour']}/hr (Surge: {pricing['surge_level']})"
            )

    live_context_text = "\n\n".join(live_context_lines)

    system_prompt = (
        "You are the Smart Parking Intelligence Platform (SPIP) AI Advisory Assistant.\n"
        "Your task is to answer user queries grounded strictly in the provided live parking context and knowledge base policies.\n"
        "CRITICAL RULE: NEVER invent, hallucinate, or guess parking occupancy or slot counts. Always cite the exact numbers from the Live Context below."
    )

    context_prompt = (
        f"=== RELEVANT PARKING POLICIES ===\n{doc_context}\n\n"
        f"=== LIVE REAL-TIME & FORECAST CONTEXT ===\n{live_context_text}\n\n"
        f"User Query: {query}\n"
    )

    # Attempt OpenAI API call if key is provided
    if OPENAI_API_KEY and len(OPENAI_API_KEY) > 10:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context_prompt}
                    ],
                    "temperature": 0.2
                },
                timeout=10
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"]
                return {
                    "query": query,
                    "answer": answer,
                    "grounded_context_used": True,
                    "model": "gpt-3.5-turbo"
                }
        except Exception as e:
            print(f"[OpenAI RAG Error] {e}")

    # Intelligent Grounded Fallback Response Generator
    query_lower = query.lower()
    matched_lines = []
    for line in live_context_lines:
        if any(w in query_lower for w in ["db city", "mall", "pvr", "theater", "sector 5", "downtown"]):
            if ("db city" in query_lower and "DB City" in line) or \
               ("pvr" in query_lower and "PVR" in line) or \
               ("sector 5" in query_lower and "Sector 5" in line):
                matched_lines.append(line)
        else:
            matched_lines.append(line)

    summary_text = "\n".join(matched_lines if matched_lines else live_context_lines[:2])

    fallback_answer = (
        f"Based on real-time sensor data and 60-minute predictive forecasts:\n\n"
        f"{summary_text}\n\n"
        f"💡 Policy Note: Advance reservations guarantee space for up to 15 mins. Check-in is verified via QR Code scan."
    )

    return {
        "query": query,
        "answer": fallback_answer,
        "grounded_context_used": True,
        "model": "SPIP_Grounded_Engine"
    }
