# backend/app/vector_store.py
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict
try:
    from .config import settings
except ImportError:
    from config import settings

class VectorStore:
    def __init__(self, collection_name: str = "indian_laws"):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.col_name = collection_name

        # Use Chroma's fast built-in embedding (same model as all-MiniLM-L6-v2)
        embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,  # ← Auto-embeddings!
                metadata={"source": "legal_docs"}
            )

    def add(self, ids, documents, metadatas=None):
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
            # No embeddings needed — Chroma does it!
        )

    def query(self, query_embedding: List[float], n_results: int = 4) -> Dict:
        """Query the vector store"""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]  # ← Removed "ids"
        )

    def count(self):
        return self.collection.count()