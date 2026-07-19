# 🏆 AI-Powered Sports Quiz Generation Agent

An interactive Streamlit web app that generates factually grounded, multiple-choice
sports quizzes using **Retrieval-Augmented Generation (RAG)**: a local **ChromaDB**
vector database of historic sports facts, combined with **live web search**
(DuckDuckGo), feeding an LLM (OpenAI) that is instructed to answer only from
the retrieved context.

Built for content creators who want to publish interactive, participation-driving
sports content instead of static news posts.

## How it works

```
User picks Sport + Difficulty
        │
        ▼
 ┌────────────────────┐   1. Query offline facts  ──▶ ChromaDB (vector search)
 │   RAG Orchestrator   │
 │  (src/generator.py)  │   2. Query live news      ──▶ DuckDuckGo Search
 └────────────────────┘
        │  merges both into one grounded context block
        ▼
 ┌────────────────────┐
 │   OpenAI LLM (JSON   │   Generates 3-5 MCQ questions, strictly grounded
 │   mode) — no free-   │   in the retrieved context. JSON mode guarantees
 │   text parsing        │   the output is always machine-parseable.
 └────────────────────┘
        │
        ▼
 Streamlit renders an interactive quiz with click-to-reveal answers,
 explanations, a running score, and a copy-paste block for social media.
```

**Why JSON mode instead of text parsing?** Free-text LLM output ("A) Option" vs
"A. Option") is fragile to parse. This project asks the model to return a strict
JSON schema (`response_format={"type": "json_object"}`), so quiz rendering never
breaks on formatting quirks — a small but meaningful upgrade for reliability.

## Project structure

```
sports-quiz-agent/
├── .env.example           # Template for your API key — copy to .env
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── sports_facts.json  # Offline knowledge base (5 sports, 22 facts)
├── chroma_db/              # Auto-created on first run — vector store lives here
├── src/
│   ├── __init__.py
│   ├── config.py           # Env loading, sport/difficulty constants
│   ├── database.py         # ChromaDB setup, population, metadata-filtered queries
│   ├── search.py           # Live DuckDuckGo web search with graceful fallback
│   └── generator.py        # RAG orchestration + OpenAI JSON-mode generation
└── app.py                  # Streamlit dashboard (entry point)
```

## Setup

### 1. Prerequisites
- Python 3.10 or 3.11 recommended (3.12 also works with this project's pinned
  dependency versions, but ChromaDB's compiled dependencies are most battle-tested
  on 3.10/3.11).
- An LLM API key -- pick one:
  - **[Groq](https://console.groq.com/keys) (free, no credit card required)** -- recommended default.
  - **[OpenAI](https://platform.openai.com/api-keys) (paid, requires billing set up)**.

### 2. Install

```bash
cd sports-quiz-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Open `.env`. By default `LLM_PROVIDER=groq` -- just paste your free key:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

To use OpenAI instead, switch the provider and fill in that section instead:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
```

`.env` is already listed in `.gitignore` — never commit it.

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit will open the dashboard in your browser (usually `http://localhost:8501`).
On first launch it automatically populates ChromaDB from `data/sports_facts.json`
(downloading a small embedding model the first time — this needs internet access).

## Using the app

1. Pick a **sport** and **difficulty** in the sidebar.
2. Choose how many questions (3–5).
3. Click **Generate Fresh Quiz**.
4. Answer each question, click **Check Answer** to reveal correctness + explanation.
5. Your score updates live. Regenerate any time for a fresh set of questions.
6. Expand **"Inspect Ground Truth"** to audit exactly what context grounded the quiz.
7. Copy the plain-text block at the bottom straight into a social post.

## Extending the knowledge baser

Add more facts to `data/sports_facts.json` (same `{"sport": ..., "fact": ...}` shape),
then force a rebuild:

```python
from src.database import setup_and_populate_db
setup_and_populate_db(force=True)
```

Add new sports by including them in `SUPPORTED_SPORTS` in `src/config.py` as well
as facts for them in the JSON file — otherwise ChromaDB retrieval for that sport
will simply return no offline facts (the web search still runs).

## Troubleshooting

| Problem | Fix |
|---|---|
| `sqlite3` version error from ChromaDB on Linux | `pip install pysqlite3-binary`, then at the top of `src/database.py` add:<br>`__import__('pysqlite3'); import sys; sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')` |
| "No API key configured" | Make sure you copied `.env.example` to `.env` and it's in the project root (not inside `src/`), and that `LLM_PROVIDER` matches the key you filled in. |
| `insufficient_quota` / 429 error from OpenAI | Your OpenAI account has no billing set up (API access is billed separately from ChatGPT Plus) or hit its spend cap. Easiest fix: switch to the free Groq provider instead (`LLM_PROVIDER=groq` in `.env`). |
| Web search returns nothing / rate-limited | DuckDuckGo's free search occasionally rate-limits; the app falls back gracefully to offline-only facts and still generates a quiz. |
| LLM response fails validation | The app surfaces the exact error (missing keys, bad `correct_answer`) in the sidebar — just click **Generate Fresh Quiz** again; JSON mode makes this rare. |
| Corrupted/failed embedding model download | Check your network connection/firewall — ChromaDB's default embedder downloads a small ONNX model from Hugging Face on first run only. |

## Evaluation checklist (maps to assignment rubric)

- ✅ Sport + difficulty selection, regenerate button
- ✅ ChromaDB vector retrieval, filtered by sport via metadata
- ✅ Live DuckDuckGo web search integrated into the same context
- ✅ LLM instructed to ground answers strictly in retrieved context (anti-hallucination)
- ✅ Structured JSON output (5 questions, 4 options, correct answer, explanation)
- ✅ Interactive Streamlit dashboard with score tracking and answer reveal
- ✅ Modular code (`config` / `database` / `search` / `generator` / `app`)
- ✅ `.env`-based secret handling, `.gitignore`, README with setup steps
