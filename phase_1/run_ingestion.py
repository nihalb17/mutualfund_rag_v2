"""
run_ingestion.py — Phase 1  (Orchestrator)
Runs the full Phase 1 pipeline:
  1. Scrape Groww pages  (scraper.py)
  2. Parse raw text      (parser.py)
  3. Build chunks        (chunker.py)
  4. Embed chunks        (embedder.py)
  5. Store in ChromaDB   (vector_store.py)
"""

import sys
import time
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from scraper      import run_scraper
from parser       import parse_scheme, SCHEME_METADATA
from chunker      import chunk_all
from embedder     import embed_texts
from vector_store import upsert_chunks, reset_collection, get_document_count

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_raw_texts() -> dict[str, str]:
    """Loads already-scraped raw text files if they exist."""
    raw_texts = {}
    for scheme_key in SCHEME_METADATA:
        raw_path = RAW_DATA_DIR / f"{scheme_key}.txt"
        if raw_path.exists():
            raw_texts[scheme_key] = raw_path.read_text(encoding="utf-8")
    return raw_texts


def run_ingestion(skip_scrape: bool = False, reset_db: bool = True):
    """
    Full pipeline runner.

    Args:
        skip_scrape: If True and raw files already exist, skips scraping step.
        reset_db:    If True, drops and recreates the ChromaDB collection
                     before inserting (ensures clean state on re-runs).
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("  Mutual Fund RAG — Phase 1 Ingestion Pipeline")
    print("=" * 60)

    # ── Step 1: Scrape ───────────────────────────────────────────────────────
    if skip_scrape:
        raw_texts = load_raw_texts()
        if len(raw_texts) == len(SCHEME_METADATA):
            print("\n[Step 1] [SKIP] Skipping scrape - raw files already exist.")
        else:
            print("\n[Step 1] Raw files missing; running scraper...")
            raw_texts = run_scraper()
    else:
        print("\n[Step 1] Scraping Groww pages...")
        raw_texts = run_scraper()

    if not raw_texts:
        print("ERROR: No raw text scraped. Aborting.")
        sys.exit(1)

    # ── Step 2: Parse ────────────────────────────────────────────────────────
    print("\n[Step 2] Parsing raw text into structured fields...")
    parsed_data: dict[str, dict] = {}
    for scheme_key, raw_text in raw_texts.items():
        parsed_data[scheme_key] = parse_scheme(scheme_key, raw_text)
        print(f"  [OK] Parsed: {scheme_key}")

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    print("\n[Step 3] Building semantic chunks...")
    all_chunks = chunk_all(parsed_data)
    print(f"  [OK] Total chunks: {len(all_chunks)}")

    # ── Step 4: Embed ────────────────────────────────────────────────────────
    print(f"\n[Step 4] Embedding {len(all_chunks)} chunks with Gemini...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # ── Step 5: Store ────────────────────────────────────────────────────────
    print("\n[Step 5] Storing in ChromaDB...")
    if reset_db:
        reset_collection()
    count = upsert_chunks(all_chunks, embeddings)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  [DONE] Ingestion complete in {elapsed:.1f}s")
    print(f"  Collection 'mutualfund_kb' contains {count} documents.")
    print("=" * 60 + "\n")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 Ingestion Pipeline")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping if raw files already exist",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do NOT reset ChromaDB before inserting",
    )
    args = parser.parse_args()

    run_ingestion(skip_scrape=args.skip_scrape, reset_db=not args.no_reset)
