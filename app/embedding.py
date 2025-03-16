from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self):
        #self.model = SentenceTransformer("BAAI/bge-small-en")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = self.model.get_sentence_embedding_dimension()  # ✅ Store the dimension

    def get_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def get_embedding_dimension(self):  # ✅ Fix: Add this method
        """Return the dimension of embeddings."""
        return self.dimension  # ✅ Return stored dimension

# Initialize embedding model
embedding_model = EmbeddingModel()
