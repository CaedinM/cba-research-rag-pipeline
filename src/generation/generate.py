"""
generate.py — NBA CBA 2023 generation layer

Two pipelines:
  rag_agentic() — default. Seeds context with an initial retrieve(), then lets
                  Claude call the retrieve tool up to MAX_TOOL_ROUNDS more times
                  to follow cross-references or fill gaps.
  rag()         — simple single-shot pipeline kept for eval scripts.

Usage:
    python generate.py "What is the rookie scale salary?"
    python generate.py "How do Bird Rights work?"
    python generate.py "What is the MLE?" --no-agent          # single-shot
    python generate.py "Bird Rights" --model claude-sonnet-4-6
"""

import argparse
import os
import sys
from pathlib import Path

import anthropic
import opik
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env.local")

os.environ.setdefault("USE_TF", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
from retrieve import (
    RankedChunk,
    expand_query,
    load_bm25,
    load_chunk_store,
    retrieve,
    CHUNK_STORE_PATH,
    BM25_PARAMS_PATH,
    EMBED_MODEL,
)

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL  = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 2048
MAX_TOOL_ROUNDS = 3   # max additional retrieve() round-trips the agent may make

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert on the NBA Collective Bargaining Agreement (2023).
Answer questions accurately using only the provided CBA excerpts.
Cite the article and section for every claim you make.
If the excerpts do not contain enough information to answer, say so.\
"""

AGENTIC_SYSTEM_PROMPT = """\
You are an expert on the NBA Collective Bargaining Agreement (2023).
You are given initial CBA excerpts relevant to the question.
If those excerpts reference other sections you need for a complete answer,
or if the initial context is insufficient, use the retrieve tool to look them up.
Answer accurately using only information from CBA excerpts (provided or retrieved).
Cite the article and section for every claim you make.
If you cannot find the information after searching, say so.\
"""

# ── Tool definition ────────────────────────────────────────────────────────────

RETRIEVE_TOOL: dict = {
    "name": "retrieve",
    "description": (
        "Search the NBA CBA 2023 for relevant sections. "
        "Use this when the provided excerpts reference another section you need, "
        "or when the initial context is insufficient to fully answer the question. "
        "Be specific: include article/section references if known, or describe the exact rule."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search the CBA.",
            }
        },
        "required": ["query"],
    },
}

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _format_context(chunks: list[RankedChunk]) -> str:
    return "\n\n---\n\n".join(f"[{c.label}]\n{c.text}" for c in chunks)


def _terminology_note(question: str, expanded: str) -> str:
    if expanded == question:
        return ""
    return (
        f"Note: \"{question}\" uses colloquial NBA terminology. "
        f"In CBA language this refers to: \"{expanded}\".\n\n"
    )


# ── Non-agentic pipeline (used by eval) ────────────────────────────────────────

@opik.track(name="generate", capture_input=True, capture_output=True)
def generate(
    question: str,
    chunks: list[RankedChunk],
    *,
    model: str = DEFAULT_MODEL,
    expanded_question: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> str:
    """Single-shot generation from pre-retrieved chunks."""
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    opik.update_current_span(metadata={"model": model, "num_chunks": len(chunks)})

    note = _terminology_note(question, expanded_question or question)
    user_message = f"CBA Excerpts:\n\n{_format_context(chunks)}\n\n{note}Question: {question}"

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


@opik.track(name="rag", capture_input=True, capture_output=True)
def rag(
    question: str,
    *,
    top_k: int = 7,
    model: str = DEFAULT_MODEL,
    embed_model=None,
    bm25=None,
    index=None,
    chunk_store: dict | None = None,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Non-agentic RAG: single retrieve → single generate. Used by eval scripts."""
    expanded = expand_query(question)
    chunks = retrieve(
        question,
        top_k=top_k,
        model=embed_model,
        bm25=bm25,
        index=index,
        chunk_store=chunk_store,
    )
    answer = generate(question, chunks, model=model, expanded_question=expanded, client=client)
    return {"answer": answer, "chunks": chunks}


# ── Agentic pipeline ───────────────────────────────────────────────────────────

@opik.track(name="rag_agentic", capture_input=True, capture_output=True)
def rag_agentic(
    question: str,
    *,
    top_k: int = 7,
    model: str = DEFAULT_MODEL,
    embed_model=None,
    bm25=None,
    index=None,
    chunk_store: dict | None = None,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Agentic RAG pipeline.

    Seeds context with an initial retrieve(), then runs a tool-use loop letting
    Claude call retrieve() up to MAX_TOOL_ROUNDS additional times when it needs
    to follow cross-references or fill gaps in the initial context.

    Returns {"answer": str, "chunks": list[RankedChunk], "tool_calls_made": int}.
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if chunk_store is None:
        chunk_store = load_chunk_store(CHUNK_STORE_PATH)

    retrieve_kwargs = dict(
        model=embed_model, bm25=bm25, index=index,
        chunk_store=chunk_store, top_k=top_k,
    )

    # ── Seed context ───────────────────────────────────────────────────────
    initial_chunks = retrieve(question, **retrieve_kwargs)
    all_chunks = list(initial_chunks)
    seen_ids: set[str] = {c.chunk_id for c in initial_chunks}

    expanded = expand_query(question)
    note = _terminology_note(question, expanded)
    user_content = (
        f"CBA Excerpts:\n\n{_format_context(initial_chunks)}\n\n"
        f"{note}Question: {question}"
    )
    messages: list[dict] = [{"role": "user", "content": user_content}]

    opik.update_current_span(metadata={
        "model": model, "top_k": top_k, "max_tool_rounds": MAX_TOOL_ROUNDS,
    })

    # ── Tool-use loop ──────────────────────────────────────────────────────
    tool_calls_made = 0
    answer = ""

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        # On the final round, withhold the tool so Claude must produce an answer.
        offer_tools = round_num < MAX_TOOL_ROUNDS

        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=AGENTIC_SYSTEM_PROMPT,
            tools=[RETRIEVE_TOOL] if offer_tools else [],
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            answer = next((b.text for b in response.content if hasattr(b, "text")), "")
            break

        if response.stop_reason == "tool_use":
            # Append assistant turn (may include both text and tool_use blocks).
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                new_chunks = retrieve(block.input.get("query", ""), **retrieve_kwargs)
                for c in new_chunks:
                    if c.chunk_id not in seen_ids:
                        all_chunks.append(c)
                        seen_ids.add(c.chunk_id)
                tool_calls_made += 1
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": (
                        _format_context(new_chunks) if new_chunks
                        else "No relevant sections found."
                    ),
                })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason (max_tokens, etc.) — surface whatever text exists.
        answer = next((b.text for b in response.content if hasattr(b, "text")), "[Response truncated]")
        break

    opik.update_current_span(metadata={"tool_calls_made": tool_calls_made})
    return {"answer": answer, "chunks": all_chunks, "tool_calls_made": tool_calls_made}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Query NBA CBA with RAG.")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("--top-k",   type=int, default=7, metavar="N")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--no-agent", action="store_true",
                        help="Use single-shot RAG instead of the agentic pipeline")
    args = parser.parse_args()

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Loading BM25 encoder...")
    bm25 = load_bm25(BM25_PARAMS_PATH)

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        sys.exit("Error: PINECONE_API_KEY not set.")

    index  = Pinecone(api_key=pinecone_key).Index("cba")
    store  = load_chunk_store(CHUNK_STORE_PATH)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    shared = dict(
        top_k=args.top_k, model=args.model,
        embed_model=embed_model, bm25=bm25,
        index=index, chunk_store=store, client=client,
    )

    print(f'\nQuestion: "{args.question}"\n')

    if args.no_agent:
        result = rag(args.question, **shared)
        tool_calls = None
    else:
        result = rag_agentic(args.question, **shared)
        tool_calls = result.get("tool_calls_made", 0)

    print("Sources:")
    for c in result["chunks"]:
        print(f"  #{c.rank} {c.label}  (score={c.score:.4f})")

    if tool_calls is not None:
        print(f"\nTool calls made: {tool_calls}")

    print(f"\nAnswer:\n{result['answer']}\n")


if __name__ == "__main__":
    main()
