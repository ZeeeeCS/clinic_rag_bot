"""vector_store.py — VectorStore class (wraps Chroma)."""
import os
import chromadb

from embeddings import EmbeddingModelLocal
from config_loader import ConfigLoader
from logger import Logger

from dotenv import load_dotenv
load_dotenv()
class VectorStore:
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = Logger()

        vs_config = self.config.get("vector_store")
        self.mode = str(vs_config.get("mode", "local")).strip().lower()

        self.client = None
        self.clinic_collection = None
        self.pubmed_collection = None
        self.embedding_model = EmbeddingModelLocal()

        if chromadb is None:
            self.logger.logger.warning("chromadb is not available; using no-op vector store")
            return

        try:

            if self.mode == "local":
                self.logger.logger.info("Initializing Chroma PersistentClient (local mode)...")
                persist_path = vs_config.get("persist_path", "./data/chroma_db")
                os.makedirs(os.path.dirname(os.path.abspath(persist_path)), exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_path)

                self.clinic_collection = self.client.get_or_create_collection("clinic_data")
                self.pubmed_collection = self.client.get_or_create_collection("pubmed_cache")
        except Exception as exc:
            self.logger.logger.warning("Falling back to no-op vector store: %s", exc)

    def _get_collection(self, collection_name):
        if collection_name == "clinic_data":
            return self.clinic_collection
        elif collection_name == "pubmed_cache":
            return self.pubmed_collection
        else:
            return None

    def add_documents(self, documents, metadatas, ids, embeddings=None, collection_name="clinic_data"):
        collection = self._get_collection(collection_name)
        if collection is None:
            return
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings,
            )
            self.logger.logger.info(f"Successfully added {len(ids)} documents to collection '{collection_name}'")
        except Exception as e:
            self.logger.logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            raise e

    def add_document(self, title, content, source=None, metadata=None, embedding=None, collection_name="clinic_data"):
        if not title and not content:
            return None

        document_text = f"Title: {title}\n\n{content}".strip()
        metadata = metadata or {}
        metadata.update({
            "title": title or "",
            "source": source or metadata.get("source") or "",
            "content_preview": (content or "")[:300],
        })

        embedding_vector = embedding
        if embedding_vector is None:
            embedding_vector = self.embedding_model.embed_text(document_text) if hasattr(self, "embedding_model") and self.embedding_model is not None else None
        
        self.add_documents(
            documents=[document_text],
            metadatas=[metadata],
            ids=[f"doc-{abs(hash(document_text))}"],
            embeddings=[embedding_vector] if embedding_vector is not None else None,
            collection_name=collection_name,
        )
        return {"text": document_text, "metadata": metadata}

    def query(self, query_vector, top_k=4, collection_name="clinic_data"):
        collection = self._get_collection(collection_name)
        if collection is None:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
            )
            return results
        except Exception as e:
            self.logger.logger.error(f"Error querying collection '{collection_name}': {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    # def add_images(self, ids, metadatas, embeddings, collection_name="clinic_data"):
    #     # For the multimodal stretch goal, we insert using precomputed embeddings
    #     collection = self._get_collection(collection_name)
    #     try:
    #         collection.add(
    #             ids=ids,
    #             metadatas=metadatas,
    #             embeddings=embeddings
    #         )
    #         self.logger.logger.info(f"Successfully added {len(ids)} images to collection '{collection_name}'")
    #     except Exception as e:
    #         self.logger.logger.error(f"Error adding images to collection '{collection_name}': {e}")
    #         raise e
    def get_all_document(self):
        return self
if __name__ == "__main__":
    vs = VectorStore()
    
    