# test_retrieval.py
from backend.app.vector_store import VectorStore
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 1. Connect to your actual vector store
vstore = VectorStore()
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 2. Your query
query = "tell me about Bharatiya nyaya sanhita?"
query_embedding = embedding_function([query])[0]

# 3. Retrieve from vector store
results = vstore.query(query_embedding, n_results=3)

# 4. Show results
print(f"Query: '{query}'")
print(f"\nTop {len(results['documents'][0])} results from vector store:")

for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
    similarity = 1 - dist  # Convert distance to similarity
    print(f"\n{i+1}. Similarity: {similarity:.4f}")
    print(f"   Distance: {dist:.4f}")
    print(f"   Metadata: {meta}")
    print(f"   Content: {doc[:150]}...")