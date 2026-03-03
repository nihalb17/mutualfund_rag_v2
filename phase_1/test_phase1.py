"""
test_phase1.py - Phase 1 Test Suite
Tests T1.1 through T1.7 as defined in ARCHITECTURE.md.

Run with:
    cd mutualfund_rag_v2
    pytest phase_1/test_phase1.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Allow imports from phase_1
sys.path.insert(0, str(Path(__file__).parent))

RAW_DATA_DIR    = Path(__file__).parent.parent / "data" / "raw"
CHUNKS_DIR      = Path(__file__).parent.parent / "data" / "chunks"
SCHEME_KEYS     = [
    "axis_liquid_direct_fund",
    "axis_elss_tax_saver",
    "axis_flexi_cap_fund",
]
REQUIRED_METADATA_KEYS = {"scheme_name", "source_url", "field_category"}


# -----------------------------------------------------------------------------
# T1.1 - Scraper fetched all 3 raw files
# -----------------------------------------------------------------------------
class TestT1_1_Scraper:
    """T1.1: Verify raw text files exist and are non-empty for all 3 schemes."""

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_raw_file_exists(self, scheme_key):
        raw_path = RAW_DATA_DIR / f"{scheme_key}.txt"
        assert raw_path.exists(), (
            f"Raw file missing: {raw_path}. "
            "Please run: python phase_1/run_ingestion.py"
        )

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_raw_file_not_empty(self, scheme_key):
        raw_path = RAW_DATA_DIR / f"{scheme_key}.txt"
        if raw_path.exists():
            size = raw_path.stat().st_size
            assert size > 1000, (
                f"Raw file for {scheme_key} seems too small ({size} bytes). "
                "Scraping may have failed or been blocked."
            )

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_raw_file_contains_scheme_name(self, scheme_key):
        raw_path = RAW_DATA_DIR / f"{scheme_key}.txt"
        if raw_path.exists():
            text = raw_path.read_text(encoding="utf-8").lower()
            assert "axis" in text, (
                f"Groww page content for {scheme_key} does not mention 'axis'. "
                "Possibly wrong page or blocked."
            )


# -----------------------------------------------------------------------------
# T1.2 - Parser extracts key fields for all 3 schemes
# -----------------------------------------------------------------------------
class TestT1_2_Parser:
    """T1.2: Verify the parser extracts required fields from raw text."""

    @pytest.fixture(autouse=True)
    def parsed_data(self):
        from parser import parse_all_from_disk
        self._parsed = parse_all_from_disk()

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_scheme_parsed(self, scheme_key):
        assert scheme_key in self._parsed, f"{scheme_key} not parsed."

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_scheme_name_present(self, scheme_key):
        parsed = self._parsed[scheme_key]
        assert parsed.get("scheme_name"), f"scheme_name missing for {scheme_key}"

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_source_url_present(self, scheme_key):
        parsed = self._parsed[scheme_key]
        url = parsed.get("source_url", "")
        assert url.startswith("https://groww.in"), (
            f"source_url invalid for {scheme_key}: {url}"
        )

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_fund_house_present(self, scheme_key):
        parsed = self._parsed[scheme_key]
        assert parsed.get("fund_house"), f"fund_house missing for {scheme_key}"

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_category_present(self, scheme_key):
        parsed = self._parsed[scheme_key]
        assert parsed.get("category"), f"category missing for {scheme_key}"

    def test_elss_has_lock_in(self):
        """ELSS scheme must declare its 3-year lock-in period."""
        parsed = self._parsed["axis_elss_tax_saver"]
        lock_in = parsed.get("lock_in", "")
        assert "3" in lock_in or "year" in lock_in.lower(), (
            f"ELSS lock-in not correctly captured: {lock_in}"
        )

    @pytest.mark.parametrize("scheme_key", SCHEME_KEYS)
    def test_stamp_duty_present(self, scheme_key):
        parsed = self._parsed[scheme_key]
        stamp = parsed.get("stamp_duty", "")
        assert stamp and stamp != "Not available", (
            f"stamp_duty missing for {scheme_key}"
        )


# -----------------------------------------------------------------------------
# T1.3 - Each chunk has required metadata keys
# -----------------------------------------------------------------------------
class TestT1_3_ChunkMetadata:
    """T1.3: All chunks must have scheme_name, source_url, field_category."""

    @pytest.fixture(autouse=True)
    def load_chunks(self):
        combined_path = CHUNKS_DIR / "all_chunks.json"
        if not combined_path.exists():
            pytest.skip("Chunks not generated yet. Run run_ingestion.py first.")
        with open(combined_path, encoding="utf-8") as f:
            self._chunks = json.load(f)

    def test_chunks_not_empty(self):
        assert len(self._chunks) > 0, "No chunks found."

    def test_all_chunks_have_required_metadata(self):
        for i, chunk in enumerate(self._chunks):
            meta = chunk.get("metadata", {})
            missing = REQUIRED_METADATA_KEYS - set(meta.keys())
            assert not missing, (
                f"Chunk {i} missing metadata keys: {missing}\n"
                f"Chunk text: {chunk.get('text', '')[:80]}"
            )

    def test_all_chunks_have_text(self):
        for i, chunk in enumerate(self._chunks):
            assert chunk.get("text", "").strip(), (
                f"Chunk {i} has empty text."
            )

    def test_source_urls_are_groww(self):
        for chunk in self._chunks:
            url = chunk["metadata"].get("source_url", "")
            assert "groww.in" in url, f"Non-Groww URL in chunk: {url}"

    def test_all_three_schemes_have_chunks(self):
        scheme_names = {c["metadata"]["scheme_name"] for c in self._chunks}
        assert len(scheme_names) == 3, (
            f"Expected chunks for 3 schemes, found: {scheme_names}"
        )


# -----------------------------------------------------------------------------
# T1.4 - Embedder generates real vectors
# -----------------------------------------------------------------------------
class TestT1_4_Embedder:
    """T1.4: Embeddings are non-zero float vectors of expected dimension."""

    def test_embed_text_returns_vector(self):
        from embedder import embed_text
        emb = embed_text("expense ratio of Axis Liquid Fund")
        assert isinstance(emb, list), "Embedding should be a list."
        assert len(emb) > 0, "Embedding vector is empty."
        assert all(isinstance(v, float) for v in emb), (
            "All values in embedding must be floats."
        )

    def test_embedding_is_nonzero(self):
        from embedder import embed_text
        emb = embed_text("What is the NAV of Axis ELSS?")
        assert any(v != 0.0 for v in emb), "Embedding vector is all zeros."

    def test_query_embedding_returns_vector(self):
        from embedder import embed_query
        emb = embed_query("minimum SIP for Axis Flexi Cap Fund")
        assert len(emb) > 100, (
            f"Query embedding too short: {len(emb)} dims. "
            "Expected ~768 for text-embedding-004."
        )


# -----------------------------------------------------------------------------
# T1.5 - ChromaDB collection has correct document count
# -----------------------------------------------------------------------------
class TestT1_5_VectorStoreCount:
    """T1.5: ChromaDB collection document count matches chunk count."""

    def test_collection_is_populated(self):
        from vector_store import get_document_count
        count = get_document_count()
        assert count > 0, (
            "ChromaDB collection is empty. Run run_ingestion.py first."
        )

    def test_collection_count_matches_chunks(self):
        from vector_store import get_document_count
        combined_path = CHUNKS_DIR / "all_chunks.json"
        if not combined_path.exists():
            pytest.skip("all_chunks.json not found.")

        with open(combined_path, encoding="utf-8") as f:
            chunks = json.load(f)

        db_count    = get_document_count()
        chunk_count = len(chunks)
        assert db_count == chunk_count, (
            f"DB has {db_count} docs but {chunk_count} chunks were generated. "
            "Possible partial ingestion."
        )


# -----------------------------------------------------------------------------
# T1.6 - Similarity search returns relevant result
# -----------------------------------------------------------------------------
class TestT1_6_SimilaritySearch:
    """T1.6: Queries return semantically relevant chunks from the correct scheme."""

    def test_expense_ratio_query_returns_relevant_chunk(self):
        from embedder import embed_query
        from vector_store import similarity_search

        emb     = embed_query("expense ratio of Axis Liquid Fund")
        results = similarity_search(emb, top_k=3)

        assert len(results) > 0, "Similarity search returned no results."
        top_text = results[0]["text"].lower()
        assert any(kw in top_text for kw in ["expense", "ratio", "liquid"]), (
            f"Top result doesn't seem relevant:\n{results[0]['text'][:200]}"
        )

    def test_nav_query_returns_relevant_chunk(self):
        from embedder import embed_query
        from vector_store import similarity_search

        emb     = embed_query("What is the NAV of Axis ELSS Tax Saver?")
        results = similarity_search(emb, top_k=3)

        assert len(results) > 0, "Similarity search returned no results."
        texts = " ".join(r["text"].lower() for r in results)
        assert "elss" in texts or "nav" in texts or "net asset value" in texts, (
            f"NAV query results don't mention ELSS or NAV:\n{texts[:300]}"
        )

    def test_exit_load_query_returns_relevant_chunk(self):
        from embedder import embed_query
        from vector_store import similarity_search

        emb     = embed_query("exit load of Axis Flexi Cap Fund")
        results = similarity_search(emb, top_k=3)

        texts = " ".join(r["text"].lower() for r in results)
        assert "exit" in texts or "flexi" in texts, (
            f"Exit load query results seem off:\n{texts[:300]}"
        )


# -----------------------------------------------------------------------------
# T1.7 - Metadata filtering works
# -----------------------------------------------------------------------------
class TestT1_7_MetadataFilter:
    """T1.7: ChromaDB 'where' filter returns only chunks from the specified scheme."""

    @pytest.mark.parametrize("scheme_name,expected_url_fragment", [
        ("Axis Liquid Direct Fund Growth", "axis-liquid"),
        ("Axis ELSS Tax Saver Direct Plan Growth", "axis-elss"),
        ("Axis Flexi Cap Fund Direct Growth", "axis-flexi"),
    ])
    def test_filter_by_scheme_name(self, scheme_name, expected_url_fragment):
        from embedder import embed_query
        from vector_store import similarity_search

        emb     = embed_query("fund information")
        results = similarity_search(
            emb,
            top_k=5,
            where={"scheme_name": scheme_name},
        )

        assert len(results) > 0, (
            f"No results returned for scheme filter: {scheme_name}"
        )
        for r in results:
            actual_name = r["metadata"]["scheme_name"]
            assert actual_name == scheme_name, (
                f"Filter leak: expected '{scheme_name}', got '{actual_name}'"
            )
            assert expected_url_fragment in r["metadata"]["source_url"], (
                f"Source URL doesn't match scheme: {r['metadata']['source_url']}"
            )
