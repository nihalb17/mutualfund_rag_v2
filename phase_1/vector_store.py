"""
vector_store.py - Phase 1
Manages the ChromaDB persistent collection for the mutual fund knowledge base.
Provides functions to initialise the DB, upsert chunks with embeddings,
and run similarity / filtered searches.
"""

import os
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# -- Config --------------------------------------------------------------------
CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./vector_db")
COLLECTION_NAME = "mutualfund_kb"

# Check if running on Vercel (serverless environment)
IS_VERCEL = os.environ.get("VERCEL", "0") == "1"

if IS_VERCEL:
    # On Vercel, copy bundled vector_db to /tmp for read-write access
    BUNDLED_DB_PATH = Path(__file__).parent.parent / "vector_db"
    TMP_DB_PATH = Path("/tmp/vector_db")
    if BUNDLED_DB_PATH.exists() and not TMP_DB_PATH.exists():
        shutil.copytree(BUNDLED_DB_PATH, TMP_DB_PATH)
    _db_path = TMP_DB_PATH
else:
    # Resolve to absolute path relative to project root
    _db_path = (Path(__file__).parent.parent / CHROMA_DB_PATH).resolve()


# -- Global Instances (Singleton Pattern) ---------------------------------------
_client_instance = None
_collection_instance = None

def get_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client (cached)."""
    global _client_instance
    if _client_instance is None:
        _db_path.mkdir(parents=True, exist_ok=True)
        _client_instance = chromadb.PersistentClient(path=str(_db_path))
    return _client_instance


def get_collection(client: chromadb.PersistentClient = None):
    """Returns (or creates) the mutual-fund collection (cached)."""
    global _collection_instance
    if _collection_instance is not None:
        return _collection_instance
        
    if client is None:
        client = get_client()
    
    _collection_instance = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection_instance


# -- Ingestion -----------------------------------------------------------------
def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Inserts or updates chunks + their embeddings into ChromaDB.

    Args:
        chunks:     list of dicts with keys 'text' and 'metadata'
        embeddings: parallel list of float vectors

    Returns:
        Number of documents in the collection after upsert.
    """
    client = get_client()
    collection = get_collection(client)

    ids       = [f"chunk_{i:05d}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    count = collection.count()
    print(f"[VectorStore] Upserted {len(chunks)} chunks. "
          f"Collection '{COLLECTION_NAME}' now has {count} documents.")
    return count


# -- Retrieval ------------------------------------------------------------------
def similarity_search(
    query_embedding: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Runs a cosine-similarity search against the collection.

    Args:
        query_embedding: Embedded query vector.
        top_k:           Number of results to return.
        where:           Optional ChromaDB metadata filter, e.g.
                         {"scheme_name": "Axis Liquid Direct Fund Growth"}

    Returns:
        List of dicts with keys: text, metadata, distance.
    """
    collection = get_collection()
    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    output: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": doc, "metadata": meta, "distance": dist})
    return output


def get_document_count() -> int:
    """Returns the total number of documents in the collection."""
    return get_collection().count()


def reset_collection():
    """Deletes and recreates the collection (useful for re-ingestion)."""
    global _collection_instance
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[VectorStore] Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass
    # Reset the cached instance
    _collection_instance = None
    # Create new collection via get_collection to cache it properly
    collection = get_collection(client)
    print(f"[VectorStore] Created fresh collection '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    count = get_document_count()
    print(f"Collection '{COLLECTION_NAME}' has {count} documents.")
    print(f"DB path: {_db_path}")
