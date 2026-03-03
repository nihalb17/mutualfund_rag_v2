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

# Initialize database on startup (for Vercel)
IS_VERCEL = os.environ.get("VERCEL", "0") == "1"
if IS_VERCEL:
    from api.init_db import init_database
    init_database()

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

# Root endpoint - serve index.html
@app.get("/")
async def root():
    """Serve the frontend HTML."""
    public_path = project_root / "public" / "index.html"
    if public_path.exists():
        return FileResponse(str(public_path))
    return {"message": "Mutual Fund RAG Chatbot API"}

# Serve static files
@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files from public folder."""
    file_full_path = project_root / "public" / file_path
    if file_full_path.exists() and file_full_path.is_file():
        return FileResponse(str(file_full_path))
    # If file not found, return index.html for SPA routing
    index_path = project_root / "public" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="Not Found")
