try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency fallback
    SentenceTransformer = None

try:
    from google.generativeai import embed_content
except Exception:  # pragma: no cover - optional dependency fallback
    embed_content = None


class EmbeddingModelCloud:
    def __init__(self):
        if embed_content is None:
            raise RuntimeError("google.generativeai is not available")
        self.model = embed_content.models.TextEmbeddingModel("models/text-embedding-004")
    def embed_text(self, text: str) -> list[float]:
        return self.model.embed(text)
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed(texts)
    def embed_document(self, document: str) -> list[float]:
        return self.model.embed(document)
    def embed_batch_documents(self, documents: list[str]) -> list[list[float]]:
        return self.model.embed(documents)
    def embed_patient(self, patient: str) -> list[float]:
        return self.model.embed(patient)
    def embed_batch_patients(self, patients: list[str]) -> list[list[float]]:
        return self.model.embed(patients)
    def embed_medication(self, medication: str) -> list[float]:
        return self.model.embed(medication)
    def embed_batch_medications(self, medications: list[str]) -> list[list[float]]:
        return self.model.embed(medications)
    def embed_all(self):
        return self.embed_document(self.embed_patient(self.embed_medication))
    def embed_batch_all(self, documents: list[str], patients: list[str], medications: list[str]) -> list[list[float]]:
        return self.embed_batch(documents + patients + medications)
    def embed_all_documents(self, documents: list[str]) -> list[list[float]]:
        return self.embed_batch_all(documents)
    def embed_all_patients(self, patients: list[str]) -> list[list[float]]:
        return self.embed_batch_all(patients)
    def embed_all_medications(self, medications: list[str]) -> list[list[float]]:
        return self.embed_batch_all(medications)
    def embed_all_documents_patients_medications(self, documents: list[str], patients: list[str], medications: list[str]) -> list[list[float]]:
        return self.embed_batch_all(documents + patients + medications)
class EmbeddingModelLocal:
    def __init__(self):
        if SentenceTransformer is None:
            self.model = None
            return
        # Loads the sentence-transformers model locally
        # Since our data is highly medical (Mayo Clinic, Medications, Clinical Notes),
        # a general model like MiniLM is poor. We use a medical-specific BERT model:
        self.model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    
    def embed_text(self, text: str) -> list[float]:
        if self.model is None:
            return [0.0] * 4
        # Returns a list of 768 numbers representing the text meaning
        return self.model.encode(text).tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            return [[0.0] * 4 for _ in texts]
        # Embed many texts at once (faster)
        return self.model.encode(texts).tolist()
    def embed_document(self, document: str) -> list[float]:
        return self.embed_text(document)
    def embed_batch_documents(self, documents: list[str]) -> list[list[float]]:
        return self.embed_batch(documents)
    def embed_patient(self, patient: str) -> list[float]:
        return self.embed_text(patient)
    def embed_batch_patients(self, patients: list[str]) -> list[list[float]]:
        return self.embed_batch(patients)
    def embed_medication(self, medication: str) -> list[float]:
        return self.embed_text(medication)
    def embed_batch_medications(self, medications: list[str]) -> list[list[float]]:
        return self.embed_batch(medications)
    def embed_all(self):
        return self.embed_document(self.embed_patient(self.embed_medication))
    def embed_batch_all(self, documents: list[str], patients: list[str], medications: list[str]) -> list[list[float]]:
        return self.embed_batch(documents + patients + medications)
    def embed_all_documents(self, documents: list[str]) -> list[list[float]]:
        return self.embed_batch_all(documents)
    def embed_all_patients(self, patients: list[str]) -> list[list[float]]:
        return self.embed_batch_all(patients)
    def embed_all_medications(self, medications: list[str]) -> list[list[float]]:
        return self.embed_batch_all(medications)
    def embed_all_documents_patients_medications(self, documents: list[str], patients: list[str], medications: list[str]) -> list[list[float]]:
        return self.embed_batch_all(documents + patients + medications)
    