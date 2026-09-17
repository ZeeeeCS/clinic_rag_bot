import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModelLocal:
    def __init__(self):
        self.model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    
    def embed_text(self, text):
        # We normalize here so that Chroma's L2 distance becomes Cosine Similarity
        vec = self.model.encode(text)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    # def embed_batch(self, texts: list[str]) -> list[list[float]]:
    #     if self.model is None:
    #         return [self._fallback_vector(text) for text in texts]
    #     return self.model.encode(texts).tolist()

    # def embed_document(self, document: str) -> list[float]:
    #     return self.embed_text(document)

    # def embed_batch_documents(self, documents: list[str]) -> list[list[float]]:
    #     return self.embed_batch(documents)

    # def embed_patient(self, patient: str) -> list[float]:
    #     return self.embed_text(patient)

    # def embed_batch_patients(self, patients: list[str]) -> list[list[float]]:
    #     return self.embed_batch(patients)

    # def embed_medication(self, medication: str) -> list[float]:
    #     return self.embed_text(medication)

    # def embed_batch_medications(self, medications: list[str]) -> list[list[float]]:
    #     return self.embed_batch(medications)

    # def embed_all(self):
    #     return self.embed_document("")

    # def embed_batch_all(self, documents: list[str], patients: list[str], medications: list[str]) -> list[list[float]]:
    #     return self.embed_batch(documents + patients + medications)

    # def embed_all_documents(self, documents: list[str]) -> list[list[float]]:
    #     return self.embed_batch_all(documents, [], [])

    # def embed_all_patients(self, patients: list[str]) -> list[list[float]]:
    #     return self.embed_batch_all([], patients, [])

    # def embed_all_medications(self, medications: list[str]) -> list[list[float]]:
    #     return self.embed_batch_all([], [], medications)

    # def embed_all_documents_patients_medications(self, documents: list[str], patients: list[str], medications: list[str]) -> list[list[float]]:
    #     return self.embed_batch_all(documents, patients, medications)

