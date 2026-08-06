"""retrievers.py — LocalRAGRetriever, PubMedRetriever."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


from config_loader import ConfigLoader
from logger import Logger
from pubmed import GeneralKnowledgeRetriever


class LocalRAGRetriever:
    """Wrapper around the general knowledge retriever."""

    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        self.config = config or ConfigLoader()
        self.logger = Logger()
        self._impl = GeneralKnowledgeRetriever(config=self.config)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        try:
            return self._impl.retrieve(query=query, top_k=top_k, confidence_threshold=0.3)
        except Exception as exc:
            self.logger.logger.warning("Local retriever failed: %s", exc)
            return []


class PubMedRetriever:
    """Placeholder for future PubMed integration."""

    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        self.config = config or ConfigLoader()
        self.logger = Logger()

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        
        self.logger.logger.info("PubMed retrieval is not enabled yet for query: %s", query)
        return []

