# vector_store.py
import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer embedding model.
        all-MiniLM-L6-v2 outputs 384-dimensional dense vectors.
        """
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None  # Shape will be (N, 384)

    def add_texts(self, texts: list[str]) -> None:
        """
        Generates dense vector embeddings for a list of text chunks and stores them.
        
        Args:
            texts: List of document text chunks.
        """
        self.chunks = texts
        print(f"Encoding {len(texts)} chunks into 384-dimensional vectors...")
        # self.model.encode returns a NumPy array of shape (len(texts), 384)
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        print(f"Vector Index shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Searches for the top_k most semantically similar chunks for a given query.
        
        Args:
            query: Natural language query string.
            top_k: Number of top results to return.
            
        Returns:
            List of tuples: [(chunk_text, cosine_similarity_score), ...]
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("Vector store is empty! Call add_texts() first.")

        # Step 1: Embed the user query string -> shape: (384,)
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # TODO: Implement vector similarity search using pure NumPy!
        #
        # Formula for Cosine Similarity between vector A and vector B:
        #   cosine_sim = dot(A, B) / (norm(A) * norm(B))
        #
        # For matrix operations across all N chunks:
        #   1. Calculate dot products between self.embeddings matrix (N, 384) and query_embedding (384,)
        #      Hint: np.dot(self.embeddings, query_embedding) -> shape: (N,)
        #   2. Calculate norms for self.embeddings rows: np.linalg.norm(self.embeddings, axis=1) -> shape: (N,)
        #   3. Calculate norm for query_embedding: np.linalg.norm(query_embedding)
        #   4. Compute similarity array (N,)
        #   5. Find indices of top_k highest similarity scores (Hint: np.argsort)
        #   6. Return list of (self.chunks[idx], float(score)) tuples
        dot_p=np.dot(self.embeddings,query_embedding)
        #norms
        matrix_norm=np.linalg.norm(self.embeddings,axis=1)     
        query_norms=np.linalg.norm(query_embedding)
        #cosine Similarities
        scores=dot_p/(matrix_norm*query_norms)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(self.chunks[idx], float(scores[idx])) for idx in top_indices]
        return results
