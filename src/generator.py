"""
The RAG "brain". Pulls context from ChromaDB (offline historic facts) and
DuckDuckGo (live web news), merges them, and asks the LLM to generate a
quiz. Uses OpenAI's JSON mode so output is always machine-parseable --
no fragile regex/string parsing of free-form text.
"""

import json
from openai import OpenAI

from src.config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
from src.database import query_historic_facts
from src.search import get_live_news_context

SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert sports quiz writer for a social media content team.
Write multiple-choice quiz questions relying strictly on the CONTEXT provided below.
Do not invent facts, statistics, dates, or names that are not supported by the context.
If the context is sparse for a topic, write questions only about what is actually covered
rather than guessing at unsupported details.

CONTEXT:
{context}
"""

USER_PROMPT_TEMPLATE = """Generate exactly {num_questions} unique multiple-choice quiz questions
about {sport} at {difficulty} difficulty.

Respond ONLY with valid JSON (no markdown fences, no commentary) matching this exact schema:
{{
  "sport": "{sport}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "question": "string",
      "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
      "correct_answer": "A" | "B" | "C" | "D",
      "explanation": "string, 1-2 sentences grounded in the provided context"
    }}
  ]
}}
"""


def gather_context(sport: str, difficulty: str) -> str:
    """Collects and merges offline (ChromaDB) and live (web search) context."""
    db_query = f"{sport} history championships records rules"
    db_matches = query_historic_facts(sport=sport, query_text=db_query, n_results=3)
    db_context = "\n".join(f"- {fact}" for fact in db_matches) if db_matches else "No offline historic data available for this sport."

    web_context = get_live_news_context(sport)

    return (
        f"=== HISTORICAL FACTS (offline knowledge base) ===\n{db_context}\n\n"
        f"=== LIVE WEB NEWS ===\n{web_context}"
    )


def compile_quiz_data(sport: str, difficulty: str, num_questions: int = 5):
    """
    Runs the full RAG pipeline:
      1. Retrieve offline facts from ChromaDB.
      2. Retrieve live news from web search.
      3. Merge into a grounded context block.
      4. Ask the LLM (JSON mode) to generate a structured quiz.

    Returns a tuple: (quiz_dict, context_used_string)
    Raises RuntimeError with a friendly message on failure.
    """
    if not LLM_API_KEY:
        raise RuntimeError(
            "No API key configured. Copy .env.example to .env and add your key."
        )

    context = gather_context(sport, difficulty)

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    system_msg = SYSTEM_INSTRUCTION_TEMPLATE.format(context=context)
    user_msg = USER_PROMPT_TEMPLATE.format(
        num_questions=num_questions, sport=sport, difficulty=difficulty
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        error_text = str(e)
        if "insufficient_quota" in error_text or "429" in error_text:
            raise RuntimeError(
                "Your OpenAI account has no available quota. The API is billed "
                "separately from ChatGPT Plus -- add a payment method and check your "
                "usage limits at platform.openai.com/settings/organization/billing/overview."
            ) from e
        raise RuntimeError(f"LLM request failed: {e}") from e

    raw_text = response.choices[0].message.content

    try:
        quiz = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Model did not return valid JSON, cannot render quiz: {e}"
        ) from e

    _validate_quiz_shape(quiz)
    return quiz, context


def _validate_quiz_shape(quiz: dict):
    """Defensive check so a malformed LLM response fails loudly and early."""
    if "questions" not in quiz or not isinstance(quiz["questions"], list) or not quiz["questions"]:
        raise RuntimeError("Model response did not include any questions.")

    for q in quiz["questions"]:
        required_keys = {"question", "options", "correct_answer", "explanation"}
        if not required_keys.issubset(q.keys()):
            raise RuntimeError(f"Malformed question object, missing keys: {q}")
        if q["correct_answer"] not in q["options"]:
            raise RuntimeError(f"correct_answer '{q['correct_answer']}' not among options: {q}")

