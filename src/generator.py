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
Your quiz questions MUST be strictly grounded in the CONTEXT provided below.

CRITICAL REQUIREMENTS:
1. Do NOT invent facts, statistics, dates, or names not in the context
2. The correct answer MUST be verifiable directly from the context
3. All four options must be plausible but only ONE is correct
4. Each distractor (wrong answer) should be subtly different, not obviously wrong
5. Questions must be clear, unambiguous, and factually accurate
6. Explanations must cite specific facts from the context

CONTEXT:
{context}

DIFFICULTY GUIDELINES:
- Easy: Direct factual recall from the context (who, what, when, where)
- Medium: Requires connecting multiple facts or understanding context
- Hard: Requires inference, comparison, or analysis of provided facts
"""

USER_PROMPT_TEMPLATE = """Generate exactly {num_questions} UNIQUE multiple-choice quiz questions
about {sport} at {difficulty} difficulty level.

STRICT RULES FOR THIS REQUEST:
1. Each question must have exactly ONE correct answer based on the provided context
2. All distractors must be plausible but clearly false when compared to context
3. Never ask questions that could have multiple valid answers
4. Do not create ambiguous or trick questions
5. Ensure proper difficulty progression (no easy questions when difficulty is Hard)
6. Each question must be independently verifiable from the context provided

Respond ONLY with valid JSON (no markdown fences, no commentary) matching this exact schema:
{{
  "sport": "{sport}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "question": "string - clear, unambiguous question grounded in context",
      "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
      "correct_answer": "A" | "B" | "C" | "D",
      "explanation": "string - 1-2 sentences explaining why this is correct, with facts from context",
      "difficulty_justification": "string - brief explanation of why this question matches the {difficulty} difficulty level"
    }}
  ]
}}
"""


def gather_context(sport: str, difficulty: str) -> str:
    """
    Collects and merges offline (ChromaDB) and live (web search) context.
    Retrieves more context for harder difficulties to enable better questions.
    """
    # Adjust retrieval based on difficulty
    n_results = {"Easy": 3, "Medium": 5, "Hard": 7}.get(difficulty, 5)
    
    # Use difficulty-aware queries
    difficulty_keywords = {
        "Easy": f"{sport} basic rules history facts winners champions",
        "Medium": f"{sport} records championships statistics tournament details strategies",
        "Hard": f"{sport} advanced tactics rivalries historical comparisons statistical analysis"
    }
    db_query = difficulty_keywords.get(difficulty, f"{sport} history championships records rules")
    
    db_matches = query_historic_facts(sport=sport, query_text=db_query, n_results=n_results)
    db_context = "\n".join(f"- {fact}" for fact in db_matches) if db_matches else "No offline historic data available for this sport."

    web_context = get_live_news_context(sport)

    return (
        f"=== HISTORICAL FACTS (offline knowledge base) ===\n{db_context}\n\n"
        f"=== LIVE WEB NEWS ===\n{web_context}"
    )


def compile_quiz_data(sport: str, difficulty: str, num_questions: int = 5):
    """
    Runs the full RAG pipeline:
      1. Retrieve offline facts from ChromaDB (difficulty-aware).
      2. Retrieve live news from web search.
      3. Merge into a grounded context block.
      4. Ask the LLM (JSON mode + lower temp) to generate a structured quiz.
      5. Validate questions for accuracy and difficulty.

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
        # Use lower temperature for more consistent, factually accurate output
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,  # Reduced from 0.8 for accuracy over creativity
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
    _validate_quiz_quality(quiz, difficulty)
    return quiz, context


def _validate_quiz_shape(quiz: dict):
    """Defensive check so a malformed LLM response fails loudly and early."""
    if "questions" not in quiz or not isinstance(quiz["questions"], list) or not quiz["questions"]:
        raise RuntimeError("Model response did not include any questions.")

    for i, q in enumerate(quiz["questions"]):
        required_keys = {"question", "options", "correct_answer", "explanation"}
        if not required_keys.issubset(q.keys()):
            raise RuntimeError(f"Question {i+1}: Malformed question object, missing keys: {q}")
        if q["correct_answer"] not in q["options"]:
            raise RuntimeError(f"Question {i+1}: correct_answer '{q['correct_answer']}' not among options: {q}")
        if len(q["options"]) != 4:
            raise RuntimeError(f"Question {i+1}: Must have exactly 4 options, found {len(q['options'])}")
        if not all(isinstance(opt, str) and len(opt.strip()) > 0 for opt in q["options"].values()):
            raise RuntimeError(f"Question {i+1}: All options must be non-empty strings")


def _validate_quiz_quality(quiz: dict, difficulty: str):
    """
    Additional validation to ensure questions meet quality standards:
    - No duplicate questions
    - Options are sufficiently different
    - Difficulty matches requested level
    """
    questions = quiz["questions"]
    
    # Check for duplicate questions
    question_texts = [q["question"].lower().strip() for q in questions]
    if len(question_texts) != len(set(question_texts)):
        raise RuntimeError("Quiz contains duplicate questions. Regenerating...")
    
    # Check that options are sufficiently different
    for i, q in enumerate(questions):
        options = list(q["options"].values())
        
        # Check for duplicate options
        options_lower = [opt.lower().strip() for opt in options]
        if len(options_lower) != len(set(options_lower)):
            raise RuntimeError(f"Question {i+1}: Contains duplicate options")
        
        # Basic check: options shouldn't be too similar (same first 10 chars)
        first_10_chars = [opt[:10].lower() for opt in options]
        if len(first_10_chars) != len(set(first_10_chars)):
            raise RuntimeError(f"Question {i+1}: Options are too similar (may be ambiguous)")
        
        # Warn if correct answer seems too obvious (very short while others are long, etc)
        correct_answer = q["options"][q["correct_answer"]]
        correct_len = len(correct_answer)
        other_lens = [len(q["options"][k]) for k in q["options"] if k != q["correct_answer"]]
        
        # If all others are much shorter or much longer, it might be suspicious
        if all(ln < correct_len * 0.6 for ln in other_lens) or all(ln > correct_len * 1.5 for ln in other_lens):
            raise RuntimeError(f"Question {i+1}: Correct answer may be too obvious (length mismatch)")

