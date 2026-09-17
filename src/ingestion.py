"""ingestion.py — load the prepared local JSON data into the vector store."""

import json
from pathlib import Path
from typing import Any, Dict, List

from config_loader import ConfigLoader
from logger import Logger
from vector_store import VectorStore


class Ingestion:
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = Logger()
        self.vector_store = VectorStore()
        self.data_root = Path(__file__).resolve().parent.parent / "data" / "raw"

    def _load_json_documents(self, folder_name: str) -> List[Dict[str, Any]]:
        folder = self.data_root / folder_name
        if not folder.exists():
            return []

        documents: List[Dict[str, Any]] = []
        for path in sorted(folder.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception as exc:
                self.logger.logger.warning("Skipping %s: %s", path.name, exc)
                continue

            title = payload.get("title") or path.stem
            content = payload.get("content") or ""
            source = payload.get("source") or str(path)
            documents.append({
                "title": title,
                "content": content,
                "source": source,
            })

        return documents

    def ingest_nhs(self):
        for item in self._load_json_documents("nhs"):
            self.vector_store.add_document(
                title=item["title"],
                content=item["content"],
                source=item["source"],
            )

    def ingest_mayo(self):
        for item in self._load_json_documents("mayo"):
            self.vector_store.add_document(
                title=item["title"],
                content=item["content"],
                source=item["source"],
            )

    def ingest_medlineplus(self):
        for item in self._load_json_documents("medlineplus"):
            self.vector_store.add_document(
                title=item["title"],
                content=item["content"],
                source=item["source"],
            )

 
