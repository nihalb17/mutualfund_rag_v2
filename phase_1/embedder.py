"""
embedder.py - Phase 1
Wraps the Gemini text-embedding-004 model for generating embeddings.
"""

import os
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Initialise Gemini
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError("GEMINI_API_KEY not set. Check your .env file.")

client = genai.Client(api_key=_api_key)


def embed_text(text: str) -> list[float]:
    """
    Embeds a single text string using Gemini text-embedding-004.
    Returns a list of floats (vector).
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Embeds a list of strings using batch embedding.
    """
    print(f"  [Embedder] Embedding {len(texts)} chunks (task={task_type})...")
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    embeddings = [embedding.values for embedding in result.embeddings]
    print(f"  [Embedder] Done. {len(embeddings)} embeddings generated.")
    return embeddings


def embed_query(query: str) -> list[float]:
    """
    Embeds a user query string (task_type='RETRIEVAL_QUERY').
    Used at retrieval time, not ingestion time.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


if __name__ == "__main__":
    test_text = "What is the expense ratio of Axis Liquid Direct Fund?"
    emb = embed_query(test_text)
    print(f"Query: '{test_text}'")
    print(f"Embedding dim: {len(emb)}")
    print(f"First 5 values: {emb[:5]}")
