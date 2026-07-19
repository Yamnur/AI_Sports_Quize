"""
Streamlit front-end for the AI-Powered Sports Quiz Generator.
Coordinates the RAG pipeline (ChromaDB + web search + LLM) and renders
an interactive quiz with click-to-reveal answers and a running score.
"""

import streamlit as st

from src.config import SUPPORTED_SPORTS, DIFFICULTY_LEVELS, validate_config
from src.database import setup_and_populate_db
from src.generator import compile_quiz_data


# --- One-time setup -----------------------------------------------------

@st.cache_resource
def prepare_knowledge_base():
    return setup_and_populate_db()


st.set_page_config(page_title="Sports Quiz Agent", page_icon="🏆", layout="centered")

config_problems = validate_config()

prepare_knowledge_base()

st.title("🏆 AI-Powered Sports Quiz Generator")
st.caption("Grounded in a local ChromaDB knowledge base + live web search (RAG). Built for social-media-ready interactive content.")

if config_problems:
    for p in config_problems:
        st.warning(p)

# --- Session state --------------------------------------------------------

if "quiz" not in st.session_state:
    st.session_state.quiz = None
    st.session_state.context_used = None
    st.session_state.answers_revealed = {}
    st.session_state.score = 0
    st.session_state.attempted = 0

# --- Sidebar controls -------------------------------------------------------

st.sidebar.header("Quiz Settings")
sport_choice = st.sidebar.selectbox("Sport", SUPPORTED_SPORTS)
difficulty = st.sidebar.select_slider("Difficulty", options=DIFFICULTY_LEVELS, value="Medium")
num_questions = st.sidebar.slider("Number of questions", min_value=3, max_value=5, value=5)

generate_clicked = st.sidebar.button("🎲 Generate Fresh Quiz", use_container_width=True, type="primary")

if generate_clicked:
    with st.spinner("Retrieving historical facts and scouring the live web..."):
        try:
            quiz, context_used = compile_quiz_data(sport_choice, difficulty, num_questions)
            st.session_state.quiz = quiz
            st.session_state.context_used = context_used
            st.session_state.answers_revealed = {}
            st.session_state.score = 0
            st.session_state.attempted = 0
            st.sidebar.success("Quiz generated!")
        except RuntimeError as e:
            st.sidebar.error(str(e))

st.sidebar.divider()
st.sidebar.caption("Regenerating replaces the current quiz and resets your score.")

# --- Main quiz display -------------------------------------------------------

if not st.session_state.quiz:
    st.info("👈 Pick a sport and difficulty, then click **Generate Fresh Quiz** to get started.")
else:
    quiz = st.session_state.quiz
    st.subheader(f"{quiz['sport']} Quiz — {quiz['difficulty']} difficulty")

    score_col, count_col = st.columns(2)
    score_col.metric("Score", f"{st.session_state.score} / {st.session_state.attempted}")
    count_col.metric("Questions", len(quiz["questions"]))

    st.divider()

    for i, q in enumerate(quiz["questions"]):
        st.markdown(f"**Q{i + 1}. {q['question']}**")

        option_labels = [f"{key}) {val}" for key, val in q["options"].items()]
        option_keys = list(q["options"].keys())

        choice = st.radio(
            "Choose an answer:",
            options=option_keys,
            format_func=lambda k, opts=q["options"]: f"{k}) {opts[k]}",
            key=f"radio_{i}",
            index=None,
            label_visibility="collapsed",
        )

        check_col, _ = st.columns([1, 4])
        if check_col.button("Check Answer", key=f"check_{i}"):
            if choice is None:
                st.warning("Pick an option first.")
            else:
                already_revealed = st.session_state.answers_revealed.get(i, False)
                st.session_state.answers_revealed[i] = True
                if not already_revealed:
                    st.session_state.attempted += 1
                    if choice == q["correct_answer"]:
                        st.session_state.score += 1

        if st.session_state.answers_revealed.get(i):
            correct = q["correct_answer"]
            if choice == correct:
                st.success(f"✅ Correct! Answer: {correct}) {q['options'][correct]}")
            else:
                st.error(f"❌ Not quite. Correct answer: {correct}) {q['options'][correct]}")
            st.markdown(f"*Explanation: {q['explanation']}*")

        st.divider()

    with st.expander("🔍 Inspect Ground Truth (RAG Context Used)"):
        st.code(st.session_state.context_used, language="markdown")

    st.text_area(
        "📋 Copy-paste-ready quiz text (for socials)",
        value="\n\n".join(
            f"Q{i+1}. {q['question']}\n"
            + "\n".join(f"{k}) {v}" for k, v in q["options"].items())
            + f"\nCorrect Answer: {q['correct_answer']}) {q['options'][q['correct_answer']]}\n"
            f"Explanation: {q['explanation']}"
            for i, q in enumerate(quiz["questions"])
        ),
        height=250,
    )
