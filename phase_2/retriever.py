"""
retriever.py - Phase 2
Handles semantic retrieval from ChromaDB.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from phase_1.vector_store import get_collection, similarity_search
from phase_1.embedder import embed_texts

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))

def retrieve_context(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embeds the query and retrieves the most relevant chunks from ChromaDB.
    Returns a list of chunk dictionaries (text + metadata).
    """
    print(f"[Retriever] Processing query: '{query}'")
    
    # 1. Generate embedding for the query
    # embed_texts expects a list, returns a list of vectors
    print(f"[Retriever] Generating query embedding...")
    query_embeddings = embed_texts([query], task_type="RETRIEVAL_QUERY")
    
    if not query_embeddings or len(query_embeddings) == 0:
        print(f"[Retriever] ERROR: No embedding generated for query")
        return []
    
    query_vector = query_embeddings[0]
    print(f"[Retriever] Query embedding generated, vector length: {len(query_vector)}")

    # 2. Search ChromaDB
    # similarity_search already returns a list[dict] with 'text', 'metadata', 'distance'
    print(f"[Retriever] Searching ChromaDB with top_k={top_k}...")
    retrieved_chunks = similarity_search(query_vector, top_k=top_k)
    
    print(f"[Retriever] Retrieved {len(retrieved_chunks)} chunks")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  [{i+1}] {chunk['metadata'].get('scheme_name', 'N/A')} - distance: {chunk.get('distance', 'N/A')}")
    
    return retrieved_chunks

if __name__ == "__main__":
    # Quick test
    test_query = "What is the expense ratio of Axis Liquid Fund?"
    chunks = retrieve_context(test_query, top_k=2)
    print(f"Retrieved {len(chunks)} chunks for: '{test_query}'")
    for c in chunks:
        print(f"\n--- {c['metadata']['scheme_name']} ({c['metadata']['field_category']}) ---")
        print(c['text'])
