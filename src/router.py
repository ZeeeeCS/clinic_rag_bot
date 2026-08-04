"""router.py — QueryRouter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RoutingDecision:
    """Represents a simple routing decision for the RAG pipeline."""

    intent: str
    confidence: float
    reason: str


class QueryRouter:
    """Route a user query to the appropriate retrieval path."""

    def __init__(self) -> None:
        self.default_intent = "general_medical"

    def route(self, user_query: str) -> RoutingDecision:
        if not user_query or not isinstance(user_query, str):
            return RoutingDecision(
                intent=self.default_intent,
                confidence=0.0,
                reason="empty_query",
            )

        query = user_query.strip().lower()
        if any(keyword in query for keyword in ["pain", "symptom", "headache", "fever", "cold", "diabetes", "blood pressure"]):
            return RoutingDecision(
                intent="general_medical",
                confidence=0.8,
                reason="medical_keywords_detected",
            )

        return RoutingDecision(
            intent=self.default_intent,
            confidence=0.6,
            reason="default_route",
        )

    def classify(self, user_query: str) -> Optional[str]:
        return self.route(user_query).intent
