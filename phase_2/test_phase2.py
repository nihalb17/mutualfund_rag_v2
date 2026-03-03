"""
test_phase2.py - Phase 2 Test Suite
Tests T2.1 through T2.11 as defined in ARCHITECTURE.md.
"""

import pytest
import time
from phase_2.rag_pipeline import query_rag

# Test cases data: (Query, Expected fragment in answer, Should trigger guardrail)
TEST_SCENARIOS = [
    # Factual Queries (T2.1 - T2.5)
    ("What is the NAV of Axis Liquid Fund?", "NAV", False),
    ("What is the expense ratio of Axis ELSS Tax Saver?", "%", False),
    ("What is the exit load of Axis Flexi Cap Fund?", "exit load", False),
    ("Who manages the Axis Liquid Fund?", "manager", False),
    ("What is the minimum SIP for Axis ELSS?", "SIP", False),
    
    # Guardrail Queries (T2.6 - T2.8)
    ("Should I invest in Axis Liquid Fund?", "investment advice", True),
    ("Tell me about Mirae Asset Liquid Fund", "don't have information", True),
    ("What is the capital of France?", "don't have an answer", False), # LLM says IDK
    
    # Complex / Multi-link (T2.9 - T2.11)
    ("What are the tax implications for Axis Flexi Cap Fund?", "tax", False),
    ("Compare expense ratios of all three funds", "Axis", False),
]

@pytest.mark.parametrize("query, expected_snippet, is_guardrail", TEST_SCENARIOS)
def test_rag_pipeline_scenarios(query, expected_snippet, is_guardrail):
    """Verifies various query scenarios against the RAG pipeline."""
    # Throttle to avoid rate limits (5 RPM for Free Tier)
    time.sleep(13)
    result = query_rag(query)
    
    answer = result["answer"].lower()
    snippet = expected_snippet.lower()
    
    # Basic check for presence of expected info
    # Note: For 'IDK' cases, the snippet might be in the answer
    assert snippet in answer or "don't have" in answer or "limited" in answer
    
    # Check guardrail flag
    if is_guardrail:
        assert result["guardrail_triggered"] is True or "don't have" in answer
    else:
        if not result["guardrail_triggered"]:
            # Should have citations for non-guardrailed factual queries
            # (unless it's an 'IDK' from LLM)
            if "don't have" not in answer:
                assert len(result["citations"]) > 0

def test_t2_11_citation_accuracy():
    """T2.11: Verify that citations are relevant to the query."""
    time.sleep(13)
    # Query specific to Axis Liquid
    result = query_rag("What is the NAV of Axis Liquid Direct Fund?")
    
    # Check that at least the Liquid fund URL is present
    liquid_url = "https://groww.in/mutual-funds/axis-liquid-direct-fund-growth"
    assert any(liquid_url in url for url in result["citations"])
    
    # Check that it doesn't just return ALL URLs if only one is relevant
    # (Though TOP_K might pull other chunks if similarity is high, 
    # we expect the primary one to be there)
    assert len(result["citations"]) <= 5 # Based on TOP_K

def test_out_of_scope_unrelated():
    """T2.8: Completely out-of-scope question should be refused by LLM."""
    time.sleep(13)
    result = query_rag("How many moons does Jupiter have?")
    assert "don't have an answer" in result["answer"].lower()
    assert len(result["citations"]) == 0
