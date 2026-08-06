"""General knowledge retrieval for the medical assistant.

This module is intentionally limited to general medical knowledge stored in a
separate vector collection named "general_knowledge". It never queries clinic
patient data, appointments, medicine inventory, or other operational data.
"""

from typing import Any, Dict, List, Optional

try:
    from .config_loader import ConfigLoader
    from .embeddings import EmbeddingModelLocal
    from .logger import Logger
    from .vector_store import VectorStore
except ImportError:  # pragma: no cover - fallback for direct execution
    from config_loader import ConfigLoader
    from embeddings import EmbeddingModelLocal
    from logger import Logger
    from vector_store import VectorStore


class GeneralKnowledgeRetriever:
    """Retrieve relevant general medical evidence from the local vector store.

    This module does not generate answers or perform intent detection. It only
    retrieves supporting evidence chunks from the pre-ingested knowledge base.
    """

    def __init__(
        self,
        config: Optional[ConfigLoader] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[Any] = None,
        collection_name: str = "general_knowledge",
    ) -> None:
        self.config = config or ConfigLoader()
        self.logger = Logger()
        self.vector_store = vector_store or VectorStore()
        self.embedding_model = embedding_model or EmbeddingModelLocal()
        self.collection_name = collection_name

        retrieval_config = self.config.get("retrieval", {})
        self.default_top_k = retrieval_config.get("top_k", 3)
        self.default_confidence_threshold = retrieval_config.get(
            "similarity_threshold", 0.55
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top relevant knowledge chunks for the given query.

        Args:
            query: Raw user question text.
            top_k: Maximum number of results to return. Defaults to config value.
            confidence_threshold: Minimum similarity score required. Defaults to
                config retrieval.similarity_threshold.

        Returns:
            A list of result dictionaries containing text, title, source, url,
            and score. Returns an empty list when no chunk meets the confidence
            bar.
        """
        if not query or not isinstance(query, str):
            return []

        top_k = top_k if top_k is not None else self.default_top_k
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.default_confidence_threshold
        )

        try:
            query_vector = self.embedding_model.embed_text(query)
        except Exception as exc:
            self.logger.logger.error(f"Failed to embed query '{query}': {exc}")
            return []

        try:
            results = self.vector_store.query(
                query_vector=query_vector,
                top_k=top_k,
                collection_name=self.collection_name,
            )
        except Exception as exc:
            self.logger.logger.error(
                f"Failed to query collection '{self.collection_name}': {exc}"
            )
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: List[Dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if not document:
                continue

            # Convert Chroma distance to a normalized similarity score in the
            # range [0, 1]. This gives a simple confidence value for filtering.
            similarity_score = self._distance_to_similarity(distance)
            if similarity_score < threshold:
                continue

            retrieved.append(
                {
                    "text": document,
                    "title": self._extract_metadata_value(metadata, "title"),
                    "source": self._extract_metadata_value(metadata, "source")
                    or self._extract_metadata_value(metadata, "source_name"),
                    "url": self._extract_metadata_value(metadata, "url"),
                    "score": round(similarity_score, 4),
                }
            )

        return retrieved

    def _distance_to_similarity(self, distance: Any) -> float:
        """Convert a vector distance into a simple similarity score."""
        if distance is None:
            return 0.0
        try:
            distance_value = float(distance)
        except (TypeError, ValueError):
            return 0.0

        if distance_value < 0:
            return 0.0

        # A simple monotonic mapping: smaller distance => higher similarity.
        return max(0.0, 1.0 / (1.0 + distance_value))

    def _extract_metadata_value(self, metadata: Optional[Dict[str, Any]], key: str) -> str:
        """Safely read metadata values while supporting different shapes."""
        if not isinstance(metadata, dict):
            return ""

        value = metadata.get(key, "")
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)


def retrieve_general_knowledge(
    query: str,
    top_k: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper for retrieving general medical knowledge chunks."""
    retriever = GeneralKnowledgeRetriever()
    return retriever.retrieve(
        query=query,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )
