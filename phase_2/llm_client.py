"""
llm_client.py - Phase 2
Wrapper for Gemini generation using the google-genai SDK.
"""

import os
from pathlib import Path
from google import genai
from google.genai.errors import ClientError
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

MODEL_ID = os.getenv("LLM_MODEL", "models/gemini-flash-latest")
FALLBACK_MODEL_ID = "models/gemini-2.5-flash"
_api_key = os.getenv("GEMINI_API_KEY")

# -- Global Instance (Singleton Pattern) ---------------------------------------
_client_instance = None

def get_gemini_client():
    global _client_instance
    if _client_instance is None:
        _client_instance = genai.Client(api_key=_api_key)
    return _client_instance

def generate_answer(prompt: str) -> str:
    """
    Calls Gemini model to generate a response based on the provided prompt.
    Falls back to Gemini 2.5 Pro if rate limit is exceeded.
    """
    client = get_gemini_client()
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
    except ClientError as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"[LLM Client] Rate limit exceeded for {MODEL_ID}. Falling back to {FALLBACK_MODEL_ID}...")
            response = client.models.generate_content(
                model=FALLBACK_MODEL_ID,
                contents=prompt
            )
        else:
            raise
    
    if not response or not response.text:
        return "I'm sorry, I couldn't generate a response."
        
    return response.text.strip()

def generate_answer_stream(prompt: str):
    """
    Streams a response from the Gemini model.
    Yields chunks of text.
    Falls back to Gemini 2.5 Pro if rate limit is exceeded.
    """
    client = get_gemini_client()
    
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL_ID,
            contents=prompt
        ):
            if chunk.text:
                yield chunk.text
    except ClientError as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"[LLM Client] Rate limit exceeded for {MODEL_ID}. Falling back to {FALLBACK_MODEL_ID}...")
            for chunk in client.models.generate_content_stream(
                model=FALLBACK_MODEL_ID,
                contents=prompt
            ):
                if chunk.text:
                    yield chunk.text
        else:
            raise

if __name__ == "__main__":
    # Quick test
    test_prompt = "Tell me a short fact about mutual funds."
    print("Prompt:", test_prompt)
    print("Response:", generate_answer(test_prompt))
