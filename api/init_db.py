"""
api/init_db.py - Initialize ChromaDB on Vercel startup
Loads pre-built chunks and creates embeddings on the fly.
"""

import json
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from phase_1.vector_store import reset_collection, upsert_chunks
from phase_1.embedder import embed_texts

CHUNKS_FILE = project_root / "data" / "chunks" / "all_chunks.json"

def init_database():
    """Initialize ChromaDB with pre-built chunks."""
    if not CHUNKS_FILE.exists():
        print(f"[InitDB] Chunks file not found: {CHUNKS_FILE}")
        return False
    
    try:
        print("[InitDB] Loading chunks from JSON...")
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"[InitDB] Loaded {len(chunks)} chunks")
        
        # Generate embeddings
        print("[InitDB] Generating embeddings...")
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        
        # Reset and populate collection
        print("[InitDB] Storing in ChromaDB...")
        reset_collection()
        count = upsert_chunks(chunks, embeddings)
        
        print(f"[InitDB] Successfully initialized with {count} documents")
        return True
        
    except Exception as e:
        print(f"[InitDB] Error initializing database: {e}")
        return False

if __name__ == "__main__":
    init_database()
