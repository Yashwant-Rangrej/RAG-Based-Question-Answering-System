import numpy as np
from app.services.embedder import embedder
from app.services.vector_store import vector_store
from app.config import settings

def test_search(query: str):
    print(f"Testing query: '{query}'")
    embedding = embedder.embed_single(query)
    results = vector_store.search(embedding, k=5, threshold=0.0) # threshold 0 to see all matches
    
    if not results:
        print("No results found.")
    for res in results:
        print(f"Score: {res.similarity_score:.4f} | Text: {res.text[:100]}...")

if __name__ == "__main__":
    queries = [
        "answeer those quetions",
        "What is Supervised Learning?",
        "Difference between supervised and unsupervised learning"
    ]
    for q in queries:
        test_search(q)
        print("-" * 50)
