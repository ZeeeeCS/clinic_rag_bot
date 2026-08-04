"""generator.py — AnswerGenerator (LLM call wrapper)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class AnswerGenerator:
    """Generate a final answer from routed context and retrieved evidence."""

    def __init__(self) -> None:
        self.system_prompt = (
            "You are a helpful medical information assistant. "
            "Provide concise, non-diagnostic guidance and encourage urgent care for emergencies."
        )

    def generate(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> str:
        if not query or not isinstance(query, str):
            return "Please provide a health-related question."

        if not evidence:
            return (
                "I can help with general medical information. "
                "Please share more detail about your symptoms or question."
            )

        context = "\n".join(
            f"- {item.get('source') or 'source'}: {str(item.get('text', ''))[:220]}"
            for item in evidence[:3]
        )
        return (
            f"Based on the available evidence, here is a concise response to your question: \n{context}"
        )

    def __call__(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> str:
        return self.generate(query, evidence)
