"""
test_api.py - Phase 4
Tests for the FastAPI backend using TestClient.
"""

import pytest
import time
from fastapi.testclient import TestClient
from phase_4.main import app
import json

client = TestClient(app)

def parse_stream(text):
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def test_t4_1_health_check():
    """T4.1: Health check endpoint returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_t4_2_new_session():
    """T4.2: New session creation returns a dummy ID."""
    response = client.post("/session/new")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] == "stateless_session"

def test_t4_4_guardrail_api():
    """T4.4: Investment advice guardrail via API."""
    session_id = "api_test_guardrail"
    
    response = client.post("/chat", json={
        "session_id": session_id,
        "message": "Should I invest in Axis Liquid Fund?"
    })
    
    assert response.status_code == 200
    data_list = parse_stream(response.text)
    data = data_list[-1]
    assert "cannot provide investment advice" in data["answer"].lower()
    assert data["guardrail_triggered"] is True

def test_t4_5_unknown_scheme_api():
    """T4.5: Unknown scheme via API."""
    session_id = "api_test_unknown"
    response = client.post("/chat", json={
        "session_id": session_id,
        "message": "Tell me about SBI Liquid Fund"
    })
    
    assert response.status_code == 200
    data_list = parse_stream(response.text)
    data = data_list[-1]
    assert "don't have information" in data["answer"].lower()
    assert data["guardrail_triggered"] is True

def test_t4_6_out_of_scope_api():
    """T4.6: Out-of-scope question via API."""
    time.sleep(13)
    session_id = "api_test_outofscope"
    response = client.post("/chat", json={
        "session_id": session_id,
        "message": "What is the capital of Japan?"
    })
    
    assert response.status_code == 200
    data_list = parse_stream(response.text)
    data = data_list[-1]
    
    # Assembly answers
    answer = "".join([d.get("chunk", "") for d in data_list])
    if "answer" in data:
        answer = data["answer"]
        
    assert "don't have an answer" in answer.lower()
    assert not data["guardrail_triggered"]

def test_t4_10_cors_headers():
    """T4.10: Check for CORS headers in preflight."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    # We use simple GET/POST and check headers on regular response
    # or OPTIONS if middleware is configured for it.
    response = client.post("/chat", json={"session_id": "test", "message": "hi"}, headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000" or response.headers.get("access-control-allow-origin") == "*"

def test_t4_3_chat_factual():
    """T4.3: Valid factual chat request."""
    # Note: Throttle as per Phase 2 learnings
    time.sleep(10)
    session_id = "api_test_factual"
    
    response = client.post("/chat", json={
        "session_id": session_id,
        "message": "What is the NAV of Axis Liquid Direct Fund?"
    })
    
    if response.status_code == 500 and "RESOURCE_EXHAUSTED" in response.text:
         pytest.skip("Gemini API quota exceeded")
    
    assert response.status_code == 200
    data_list = parse_stream(response.text)
    data = data_list[-1]
    assert "done" in data
    assert not data["guardrail_triggered"]
    
    answer = "".join([d.get("chunk", "") for d in data_list])
    
    # Should have at least one citation
    if "don't have an answer" not in answer.lower():
        assert len(data["citations"]) > 0
