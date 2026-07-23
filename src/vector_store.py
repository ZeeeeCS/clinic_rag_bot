"""vector_store.py — VectorStore class (wraps Chroma)."""
import os
import chromadb
from config_loader import ConfigLoader
from logger import Logger

class VectorStore:
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = Logger()
        
        # Read vector store configuration from settings.yaml
        vs_config = self.config.get("vector_store", {})
        self.mode = vs_config.get("mode", "local")
        
        if self.mode == "cloud":
            self.logger.logger.info("Initializing Chroma CloudClient...")
            self.client = chromadb.CloudClient(
                tenant=os.environ.get("CHROMA_TENANT", ""),
                database=os.environ.get("CHROMA_DATABASE", ""),
                api_key=os.environ.get("CHROMA_API_KEY", "")
            )
        else:
            self.logger.logger.info("Initializing Chroma PersistentClient (local mode)...")
            persist_path = vs_config.get("persist_path", "./data/chroma_db")
            # Ensure the persist path folder exists
            os.makedirs(os.path.dirname(os.path.abspath(persist_path)), exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_path)

        # Get or create the standard collections
        self.clinic_collection = self.client.get_or_create_collection("clinic_data")
        self.pubmed_collection = self.client.get_or_create_collection("pubmed_cache")

    def _get_collection(self, collection_name):
        if collection_name == "clinic_data":
            return self.clinic_collection
        elif collection_name == "pubmed_cache":
            return self.pubmed_collection
        else:
            return self.client.get_or_create_collection(collection_name)

    def add_documents(self, documents, metadatas, ids, embeddings=None, collection_name="clinic_data"):
        collection = self._get_collection(collection_name)
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            self.logger.logger.info(f"Successfully added {len(ids)} documents to collection '{collection_name}'")
        except Exception as e:
            self.logger.logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            raise e

    def query(self, query_vector, top_k=4, collection_name="clinic_data"):
        collection = self._get_collection(collection_name)
        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            return results
        except Exception as e:
            self.logger.logger.error(f"Error querying collection '{collection_name}': {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    def add_images(self, ids, metadatas, embeddings, collection_name="clinic_data"):
        # For the multimodal stretch goal, we insert using precomputed embeddings
        collection = self._get_collection(collection_name)
        try:
            collection.add(
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings
            )
            self.logger.logger.info(f"Successfully added {len(ids)} images to collection '{collection_name}'")
        except Exception as e:
            self.logger.logger.error(f"Error adding images to collection '{collection_name}': {e}")
            raise e
    def get_all_document(self):
        return self
        