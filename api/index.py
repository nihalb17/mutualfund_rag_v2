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
startup_error = None
if IS_VERCEL:
    try:
        from api.init_db import init_database
        success, msg = init_database()
        print(f"[Startup] Database initialization: {'SUCCESS' if success else 'FAILED'} - {msg}")
        if not success:
            startup_error = msg
    except Exception as e:
        import traceback
        startup_error = f"{e}\n{traceback.format_exc()}"
        print(f"[Startup] Database initialization error: {startup_error}")

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

@app.get("/health")
async def health_check():
    """Returns the health status of the API."""
    try:
        from phase_1.vector_store import get_document_count
        count = get_document_count()
        return {
            "status": "ok",
            "documents": count,
            "vercel": IS_VERCEL
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "vercel": IS_VERCEL
        }

@app.get("/init-db")
async def init_db_endpoint():
    """Manually trigger database initialization."""
    global startup_error
    try:
        from api.init_db import init_database, CHUNKS_FILE
        import os
        
        result = {
            "chunks_file_exists": CHUNKS_FILE.exists(),
            "chunks_file_path": str(CHUNKS_FILE),
            "cwd": os.getcwd(),
            "ls_data": os.listdir("data") if os.path.exists("data") else "N/A",
            "startup_error": startup_error,
        }
        
        if CHUNKS_FILE.exists():
            success, msg = init_database()
            result["init_success"] = success
            result["init_message"] = msg
            
            # Try to get document count, but handle if collection doesn't exist
            try:
                from phase_1.vector_store import get_document_count
                result["documents_after"] = get_document_count()
            except Exception as count_error:
                result["documents_after"] = 0
                result["count_error"] = str(count_error)
        
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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
    """Serve the frontend HTML from phase_5 folder."""
    index_path = project_root / "phase_5" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Mutual Fund RAG Chatbot API - UI not found"}

# Serve static files from phase_5
@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files from phase_5 folder."""
    file_full_path = project_root / "phase_5" / file_path
    if file_full_path.exists() and file_full_path.is_file():
        return FileResponse(str(file_full_path))
    
    # Try to serve index.html for SPA routing
    index_path = project_root / "phase_5" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    
    raise HTTPException(status_code=404, detail=f"Not Found: {file_path}")
