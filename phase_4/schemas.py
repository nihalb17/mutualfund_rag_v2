"""
schemas.py - Phase 4
Pydantic models for the FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, example="session_123")
    message: str = Field(..., example="What is the NAV of Axis Liquid Fund?")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[str]
    guardrail_triggered: bool
    contextualized_query: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str

class HealthResponse(BaseModel):
    status: str
