"""
Centralized configuration. Loads secrets from a local .env file so API
keys never get hardcoded into source files (or accidentally committed).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM_PROVIDER can be "openai" or "groq". Groq is free (no billing required)
# and offers an OpenAI-compatible API, so switching providers only changes
# the base_url and which key/model we read -- src/generator.py stays the same.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

if LLM_PROVIDER == "groq":
    LLM_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    LLM_BASE_URL = "https://api.groq.com/openai/v1"
else:
    LLM_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_BASE_URL = None  # use OpenAI's default

CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "sports_history"
SPORTS_FACTS_PATH = "./data/sports_facts.json"

SUPPORTED_SPORTS = ["Cricket", "Football", "Badminton", "Tennis", "Basketball"]
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]


def validate_config():
    """Returns a list of human-readable problems with the current config (empty = OK)."""
    problems = []
    if not LLM_API_KEY:
        key_name = "GROQ_API_KEY" if LLM_PROVIDER == "groq" else "OPENAI_API_KEY"
        problems.append(
            f"{key_name} is missing. Copy .env.example to .env and add your key."
        )
    return problems

