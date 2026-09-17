"""pipeline.py — ClinicRAGPipeline (orchestrates Router → Retriever → Generator → SafetyAgent)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from generator import AnswerGenerator
from router import QueryRouter
from retrievers import LocalRAGRetriever
from safety_agent import SafetyAgent


class ClinicRAGPipeline:
    """Simple orchestrator for routing, retrieving, and generating responses."""

    def __init__(
        self,
        router: Optional[QueryRouter] = None,
        retriever: Optional[LocalRAGRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        safety_agent: Optional[SafetyAgent] = None,
    ) -> None:
        self.router = router or QueryRouter()
        self.retriever = retriever or LocalRAGRetriever()
        self.generator = generator or AnswerGenerator()
        self.safety_agent = safety_agent or SafetyAgent()

    def run(self, user_query: str) -> str:
        if not user_query or not isinstance(user_query, str):
            return "Please provide a health-related question."

        safety_result = self.safety_agent.run(user_query)
        if "Immediate attention required" in safety_result:
            return safety_result

        decision = self.router.route(user_query)
        evidence = self.retriever.retrieve(user_query, top_k=3)
        return self.generator.generate(user_query, evidence)

    def process(self, user_query: str) -> str:
        return self.run(user_query)
