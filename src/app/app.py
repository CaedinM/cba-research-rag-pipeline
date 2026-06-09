"""
app.py — minimal Streamlit frontend for the NBA CBA 2023 RAG system.

Run with:
    .venv/bin/streamlit run src/app/app.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent

load_dotenv(ROOT / ".env.local")

sys.path.insert(0, str(ROOT / "src" / "generation"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))

import anthropic
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from retrieve import (
    BM25_PARAMS_PATH,
    CHUNK_STORE_PATH,
    EMBED_MODEL,
    load_bm25,
    load_chunk_store,
)
from generate import DEFAULT_MODEL, rag, rag_agentic


@st.cache_resource(show_spinner="Loading models and index...")
def load_resources():
    """Load the embedding model, BM25 encoder, Pinecone index, chunk store, and client once."""
    embed_model = SentenceTransformer(EMBED_MODEL)
    bm25 = load_bm25(BM25_PARAMS_PATH)
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index("cba")
    chunk_store = load_chunk_store(CHUNK_STORE_PATH)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return embed_model, bm25, index, chunk_store, client


st.set_page_config(page_title="NBA CBA RAG", page_icon="🏀")
st.title("🏀 NBA CBA 2023 — Ask a Question")

with st.sidebar:
    top_k = st.slider("Top-K sections", min_value=3, max_value=15, value=10)
    use_agent = st.toggle("Agentic pipeline", value=True)

embed_model, bm25, index, chunk_store, client = load_resources()

question = st.text_input("Your question", placeholder="What is the rookie scale salary?")

if st.button("Ask", type="primary") and question.strip():
    shared = dict(
        top_k=top_k,
        model=DEFAULT_MODEL,
        embed_model=embed_model,
        bm25=bm25,
        index=index,
        chunk_store=chunk_store,
        client=client,
    )
    with st.spinner("Thinking..."):
        result = rag_agentic(question, **shared) if use_agent else rag(question, **shared)

    st.markdown("### Answer")
    st.markdown(result["answer"])

    if use_agent:
        st.caption(f"Tool calls made: {result.get('tool_calls_made', 0)}")

    with st.expander(f"Sources ({len(result['chunks'])})"):
        for c in result["chunks"]:
            st.markdown(f"**#{c.rank} {c.label}** — score {c.score:.4f}")
