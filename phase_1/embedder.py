"""
embedder.py - Phase 1
Wraps the Gemini text-embedding-004 model for generating embeddings.
"""

import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Rate limiting: 100 requests per minute max on free tier
# Using 0.7s delay = ~85 requests per minute (safe margin)
EMBEDDING_DELAY_SECONDS = 0.7

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
    Embeds a list of strings using batch embedding with rate limiting
    to stay under Gemini free tier limits (100 requests/min).
    Processes texts one at a time with delays to avoid rate limits.
    """
    print(f"  [Embedder] Embedding {len(texts)} chunks with rate limiting (task={task_type})...")
    embeddings = []

    for i, text in enumerate(texts):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings.append(result.embeddings[0].values)

            # Rate limiting: delay between requests to stay under 100/min
            if i < len(texts) - 1:  # Don't delay after the last request
                time.sleep(EMBEDDING_DELAY_SECONDS)

        except Exception as e:
            print(f"  [Embedder] Error embedding chunk {i}: {e}")
            raise

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
