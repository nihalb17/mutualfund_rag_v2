"""
api/index.py - Vercel Serverless Entry Point
FastAPI application for the Mutual Fund RAG Chatbot.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from phase_2.rag_pipeline import query_rag, query_rag_stream
from phase_4.schemas import ChatRequest, ChatResponse, SessionResponse, HealthResponse
from fastapi.responses import FileResponse, StreamingResponse

app = FastAPI(
    title="Mutual Fund RAG Chatbot API (Stateless)",
    description="Backend for querying Axis Mutual Fund schemes via RAG (No history support)",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns the health status of the API."""
    return {"status": "ok"}

@app.post("/session/new", response_model=SessionResponse)
async def create_session():
    """Returns a dummy session ID (stateless)."""
    return {"session_id": "stateless_session"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main RAG endpoint (Stateless).
    """
    try:
        return StreamingResponse(
            query_rag_stream(request.message),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """No-op (stateless)."""
    return {"message": "Stateless session - nothing to clear."}

# Static Files - Serve from public folder for Vercel
PUBLIC_PATH = project_root / "public"
if PUBLIC_PATH.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_PATH), html=True), name="frontend")
