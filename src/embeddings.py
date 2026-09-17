import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModelLocal:
    def __init__(self):
        self.model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    # tokens into vectors
    def embed_text(self, text):
        # We normalize here so that Chroma's L2 distance becomes Cosine Similarity
        vec = self.model.encode(text)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()


