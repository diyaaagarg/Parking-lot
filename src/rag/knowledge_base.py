import numpy as np
from sqlalchemy.orm import Session
from src.db.models import KnowledgeDoc, Venue, Lot

class ParkingKnowledgeBase:
    def __init__(self, db: Session):
        self.db = db

    def search_knowledge(self, query: str, top_k: int = 3) -> list:
        """
        Retrieves top-k relevant knowledge documents based on keyword similarity.
        """
        docs = self.db.query(KnowledgeDoc).all()
        if not docs:
            return []

        query_terms = set(query.lower().split())
        scored_docs = []

        for d in docs:
            text = f"{d.title} {d.category} {d.content}".lower()
            score = sum(1 for term in query_terms if term in text)
            scored_docs.append((score, d))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
