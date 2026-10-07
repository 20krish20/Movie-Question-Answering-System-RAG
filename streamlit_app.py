"""
Streamlit web app for the Movie RAG QA system.

Run locally:  streamlit run streamlit_app.py
Deployed on:  Streamlit Community Cloud (LLM_API_KEY set under app Settings -> Secrets)
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from movie_rag import llm  # noqa: E402
from movie_rag.service import MAX_K, MAX_QUERY_LEN, MovieQA  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("movie_rag.app")

SEMANTIC_EXAMPLES = [
    "Movies about AI turning against humans",
    "A heist movie with a clever twist ending",
    "Feel-good animated films about friendship",
    "Dark psychological thrillers with an unreliable narrator",
    "Space exploration and survival on another planet",
    "Christopher Nolan mind-bending movies",
]
FACTUAL_EXAMPLES = [
    "What is the average rating of action movies after 2010?",
    "Top 5 highest grossing movies directed by Steven Spielberg",
    "How many movies have a rating greater than 8.5?",
    "Which director has the most movies with rating above 8?",
    "How many movies did Martin Scorsese direct?",
    "Average runtime of horror movies released before 2000",
]
MODES = {"Auto (router decides)": None, "Semantic search": "semantic", "Factual (LLM → pandas)": "factual"}
MAX_FACTUAL_PER_SESSION = 30  # protects the shared free LLM quota

st.set_page_config(page_title="Movie RAG QA", page_icon="🎬", layout="wide")

LLM_SETTINGS = ("LLM_BASE_URL", "LLM_MODEL", "LLM_REASONING_EFFORT")
KEY_NAMES = ("LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")


def load_secrets() -> str:
    """Copy Streamlit secrets into env vars so movie_rag.llm reads them the same way everywhere.

    Forgiving on purpose: accepts GROQ_API_KEY/OPENAI_API_KEY, any letter case, and keys
    nested under a [section]. Returns a short diagnostic (secret names only, never values).
    """
    try:
        flat = {}
        for name, value in st.secrets.to_dict().items():
            if isinstance(value, dict):  # e.g. [groq] api_key = "..."
                for sub, sub_value in value.items():
                    flat[sub.upper()] = sub_value
                    flat[f"{name}_{sub}".upper()] = sub_value
            else:
                flat[name.upper()] = value
    except FileNotFoundError:
        return "no secrets file"
    except Exception as e:  # invalid TOML, e.g. a bare key pasted without NAME = "..."
        return f"secrets could not be parsed ({type(e).__name__}); use the format LLM_API_KEY = \"gsk_...\""

    api_key = next((flat[k] for k in (*KEY_NAMES, "API_KEY") if flat.get(k)), None)
    if api_key:
        os.environ.setdefault("LLM_API_KEY", str(api_key).strip())
    for key in LLM_SETTINGS:
        if flat.get(key):
            os.environ.setdefault(key, str(flat[key]))
    return f"secret names found: {', '.join(sorted(flat)) or 'none'}"


SECRETS_STATUS = load_secrets()


@st.cache_resource(show_spinner="Loading movies, FAISS index and embedding model…")
def get_qa() -> MovieQA:
    return MovieQA()


@st.cache_data(max_entries=500, show_spinner=False)
def ask_cached(query: str, k: int, force_type):
    return get_qa().ask(query, k=k, force_type=force_type)


def run_query(query: str, k: int, mode: str):
    force_type = MODES[mode]
    try:
        out = ask_cached(query.strip(), k, force_type)
    except ValueError as e:
        st.warning(str(e))
        return
    except llm.LLMNotConfigured:
        st.warning("Factual queries need an LLM key, which isn't configured on this deployment. "
                   "Semantic search still works.")
        return
    except Exception as e:
        log.exception("query failed: %r", query)
        st.error(f"Sorry, I couldn't answer that: {e}")
        return

    log.info("query=%r type=%s", out["query"], out["query_type"])
    st.caption(f"Route: **{out['query_type']}** → `{out['pipeline_used']}`")

    if out["query_type"] == "factual":
        if out["result_kind"] == "table":
            st.dataframe(out["result"], hide_index=True, width="stretch")
        else:
            st.markdown(f"### {out['result']}")
        with st.expander("Generated pandas code", expanded=False):
            st.code(out["generated_code"], language="python")
        return

    rows = pd.DataFrame([
        {
            "Title": r["Title"],
            "Year": r["Year"],
            "Genres": r["Genres"],
            "Rating": r["Rating"],
            "Similarity": round(r["score"], 3),
            "Plot": r["Summary"],
        }
        for r in out["retrieved"]
    ])
    st.dataframe(rows, hide_index=True, width="stretch")


qa = get_qa()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", list(MODES), index=0)
    k = st.slider("Results (semantic)", 1, MAX_K, 5)
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "- *Descriptive* questions → mpnet embeddings + FAISS cosine search.\n"
        "- *Numeric* questions → an LLM writes one line of pandas, which is checked by an "
        "AST allowlist and executed in a restricted scope.\n\n"
        "[Source on GitHub](https://github.com/20krish20/Movie-Question-Answering-System-RAG)"
    )

st.title("🎬 Movie RAG QA")
st.write(f"Ask anything about the IMDB top {qa.index.ntotal:,} movies.")
if not llm.is_configured():
    st.info("No LLM key configured: factual queries are disabled, semantic search works.  \n"
            f"Diagnostics: {SECRETS_STATUS}. Expected a secret like `LLM_API_KEY = \"gsk_...\"`.")

st.session_state.setdefault("query", "")
st.session_state.setdefault("factual_count", 0)


def use_example(q: str):
    st.session_state.query = q


col1, col2 = st.columns(2)
with col1:
    st.markdown("**Try a semantic query**")
    for q in SEMANTIC_EXAMPLES:
        st.button(q, key=f"s_{q}", on_click=use_example, args=(q,), width="stretch")
with col2:
    st.markdown("**Try a factual query**")
    for q in FACTUAL_EXAMPLES:
        st.button(q, key=f"f_{q}", on_click=use_example, args=(q,), width="stretch")

query = st.text_input("Your question", key="query", max_chars=MAX_QUERY_LEN,
                      placeholder="e.g. movies about time travel paradoxes")

if query.strip():
    from movie_rag.pipelines.router import classify_query_type

    will_be_factual = (MODES[mode] or classify_query_type(query)) == "factual"
    if will_be_factual and st.session_state.factual_count >= MAX_FACTUAL_PER_SESSION:
        st.warning("You've reached the factual-query limit for this session. Semantic search still works.")
    else:
        if will_be_factual:
            st.session_state.factual_count += 1
        with st.spinner("Thinking…"):
            run_query(query, k, mode)
