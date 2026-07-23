"""ingestion.py — nhs.uk, mayoclinic.org, medlineplus.gov."""

import json
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Any
from embeddings import EmbeddingModelLocal
from vector_store import VectorStore
from logger import Logger
from config_loader import ConfigLoader


logger = Logger()
config = ConfigLoader()
embedding_model = EmbeddingModelLocal()
vector_store = VectorStore()
class Ingestion:
    
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = Logger()
        self.embedding_model = EmbeddingModelLocal()
        self.vector_store = VectorStore()
        self.nhs_urls = self.config.get("nhs_urls", [])
        self.mayo_urls = self.config.get("mayo_urls", [])
        self.medlineplus_urls = self.config.get("medlineplus_urls", [])
    def ingest_nhs(self):
        for url in self.nhs_urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text()
            self.vector_store.add_document(content, url)
    def ingest_mayo(self):
        for url in self.mayo_urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text()
            self.vector_store.add_document(content, url)
    def ingest_medlineplus(self):
        for url in self.medlineplus_urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text()
            self.vector_store.add_document(content, url)
#     def ingest_all(self):
#         self.ingest_nhs()
#         self.ingest_mayo()
#         self.ingest_medlineplus()
#         return self.vector_store.get_all_documents()
# if __name__ == "__main__":
#     ingestion = Ingestion()
#     ingestion.ingest_all()
#     print(ingestion.vector_store.get_all_documents())
