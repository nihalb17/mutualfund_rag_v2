import os
from google import genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")
_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=_api_key)

print("Listing embedding models...")
for m in client.models.list():
    if 'embedContent' in m.supported_actions:
        print(f"Name: {m.name}, Display: {m.display_name}")
