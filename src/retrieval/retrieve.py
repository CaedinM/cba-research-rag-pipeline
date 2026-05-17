"""
retrieve.py — NBA CBA 2023 retrieval layer
Embeds a question, queries Pinecone, and returns ranked chunks from chunk_store.json.

Usage:
    python retrieve.py "What is the rookie scale salary?"
    python retrieve.py "Can a team waive a player after the trade deadline?" --top-k 8
    python retrieve.py "What are the rules for two-way contracts?" --top-k 5 --show-full
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import opik

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env.local")

os.environ.setdefault("USE_TF", "0")

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX_NAME        = "cba"
EMBED_MODEL       = "multi-qa-mpnet-base-dot-v1"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_STORE_PATH  = Path(__file__).parent.parent.parent / "data" / "processed" / "chunk_store.json"
BM25_PARAMS_PATH  = Path(__file__).parent.parent.parent / "data" / "processed" / "bm25_params.json"
HYBRID_ALPHA      = 0.68   # 1.0 = pure dense, 0.0 = pure BM25
PREVIEW_LEN       = 300

# Colloquial NBA terms → CBA document vocabulary.
# Applied before embedding so the vector matches text that's actually in the document.
JARGON: dict[str, str] = {
    "bird rights":                       "qualifying veteran free agent exception salary cap",
    "larry bird exception":              "qualifying veteran free agent exception salary cap",
    "early bird exception":              "early qualifying veteran free agent",
    "non-bird exception":                "non-qualifying veteran free agent",
    "luxury tax":                        "tax level",
    "luxury tax threshold":              "tax level",
    "hard cap":                          "hard cap taxpayer mid-level exception",
    "mid-level exception":               "non-taxpayer mid-level exception",
    "mle":                               "non-taxpayer mid-level exception",
    "max contract":                      "maximum player salary",
    "max salary":                        "maximum player salary",
    "supermax":                          "designated veteran player extension",
    "designated player extension":       "designated veteran player extension",
    "rfa":                               "restricted free agent",
    "ufa":                               "unrestricted free agent",
    "tpe":                               "traded player exception",
    "traded player exception":           "trade exception",
    "sign and trade":                    "sign-and-trade",
    "sign-and-trade":                    "sign-and-trade",
    "stretch provision":                 "stretch",
    "stretch player":                    "stretch",
    "rookie extension":                  "rookie scale extension",
    "rookie contract":                   "rookie scale contract",
    "two-way contract":                  "two-way contract",
    "minimum salary":                    "minimum player salary",
    "veteran minimum":                   "minimum player salary",
    "salary cap exception":              "salary cap exception",
    "marijuana":                         "marijuana and alcohol treatment program",
    "cannabis":                          "marijuana and alcohol treatment program",
    "weed":                              "marijuana and alcohol treatment program",
    "non-guaranteed":                    "compensation protection",
    "non-guaranteed contract":           "compensation protection",
    "unguaranteed":                      "compensation protection",
    "guarantee":                         "compensation protection",
}


def _normalize(text: str) -> str:
    """Lowercase and collapse punctuation variants to plain spaces."""
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[-–—]", " ", text)       # dashes → spaces
    text = re.sub(r"[''`']", "", text)  # strip apostrophes
    return re.sub(r"\s+", " ", text).strip()


# Pre-normalize jargon keys once at import time.
_JARGON_NORM: list[tuple[str, str]] = [
    (_normalize(k), v) for k, v in JARGON.items()
]

_FUZZY_THRESHOLD = 0.85


def expand_query(question: str) -> str:
    """Replace colloquial NBA jargon with CBA document terminology.

    Handles punctuation/spacing variants (normalization) and slight
    spelling mistakes (fuzzy word-window match on multi-word keys).
    Single-word keys use exact match only to avoid false positives.
    """
    working = _normalize(question)

    for norm_key, cba_term in _JARGON_NORM:
        if norm_key in working:
            working = working.replace(norm_key, cba_term)
            continue

        # Fuzzy path: slide a same-length word window over the query.
        key_words = norm_key.split()
        n = len(key_words)
        if n < 2:
            continue
        words = working.split()
        for i in range(len(words) - n + 1):
            window = " ".join(words[i : i + n])
            if SequenceMatcher(None, norm_key, window).ratio() >= _FUZZY_THRESHOLD:
                words[i : i + n] = [cba_term]
                working = " ".join(words)
                break

    return working


@dataclass
class RankedChunk:
    rank:          int
    score:         float
    chunk_id:      str
    label:         str
    article:       str
    article_title: str
    section:       str
    text:          str

    def preview(self, chars: int = PREVIEW_LEN) -> str:
        return self.text[:chars] + ("..." if len(self.text) > chars else "")


def load_chunk_store(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_bm25(path: Path = BM25_PARAMS_PATH) -> BM25Encoder:
    return BM25Encoder().load(str(path))


def load_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


def _hybrid_vectors(dense: list[float], sparse: dict, alpha: float) -> tuple[list[float], dict]:
    """Scale dense and sparse vectors by alpha for Pinecone hybrid query."""
    a = float(alpha)  # guard against numpy scalars (e.g. from np.arange)
    return (
        [v * a for v in dense],
        {"indices": sparse["indices"], "values": [v * (1 - a) for v in sparse["values"]]},
    )


def _keyword_overlap(text: str, query_tokens: set[str]) -> float:
    """Fraction of query tokens (≥4 chars) that appear in chunk text."""
    if not query_tokens:
        return 0.0
    text_lower = text.lower()
    return sum(1 for t in query_tokens if t in text_lower) / len(query_tokens)


@opik.track(name="retrieve", capture_input=True, capture_output=True)
def retrieve(
    question: str,
    *,
    top_k: int = 5,
    alpha: float = HYBRID_ALPHA,
    model: SentenceTransformer | None = None,
    bm25: BM25Encoder | None = None,
    reranker: CrossEncoder | None = None,
    index=None,
    chunk_store: dict | None = None,
) -> list[RankedChunk]:
    """
    Embed question → query Pinecone → (optionally) cross-encoder rerank → hydrate from chunk_store.
    Returns a ranked list of RankedChunk objects.
    Accepts pre-loaded model/index/store/reranker so callers can reuse them across queries.
    """
    opik.update_current_span(metadata={"alpha": alpha, "top_k": top_k, "rerank": reranker is not None})
    if chunk_store is None:
        chunk_store = load_chunk_store(CHUNK_STORE_PATH)

    if model is None:
        model = SentenceTransformer(EMBED_MODEL)

    if bm25 is None:
        bm25 = load_bm25()

    if index is None:
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_key:
            sys.exit("Error: PINECONE_API_KEY not set.")
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(INDEX_NAME)

    expanded = expand_query(question)
    dense    = model.encode([expanded])[0].tolist()
    sparse   = bm25.encode_queries(expanded)
    hdense, hsparse = _hybrid_vectors(dense, sparse, alpha)

    # Over-fetch: large sections (e.g. Article VII Section 6) produce many chunks;
    # we need enough candidates so the right one survives section-level dedup.
    response = index.query(
        vector           = hdense,
        sparse_vector    = hsparse,
        top_k            = top_k * 8,
        include_metadata = True,
    )

    matches = response["matches"]

    # Cross-encoder reranking: rescore every candidate on (query, text) before dedup.
    # Keep scores in a separate map to avoid mutating Pinecone's ScoredVector objects.
    score_map: dict[str, float] = {m["id"]: m["score"] for m in matches}
    if reranker is not None and matches:
        pairs = [(expanded, chunk_store.get(m["id"], {}).get("text", "")) for m in matches]
        ce_scores = reranker.predict(pairs)
        score_map = {m["id"]: float(s) for m, s in zip(matches, ce_scores)}

    # Two-pass dedup: group candidates by (article, section), then pick the chunk
    # with the highest keyword overlap with the expanded query from each group.
    # This prevents a high-scoring but off-topic sibling chunk from blocking the
    # most relevant chunk in a large section (e.g. Article VII Section 6).
    query_tokens = {t for t in _normalize(expanded).split() if len(t) >= 4}

    section_groups: dict[tuple[str, str], list[tuple]] = {}
    for match in matches:
        chunk_id = match["id"]
        record   = chunk_store.get(chunk_id, {})
        article  = record.get("article", "")
        section  = record.get("section", "")

        # Skip UPC exhibit boilerplate — substantive rules are restated in the articles
        if article == "ARTICLE XLII" and section.startswith("Section 3"):
            continue

        key = (article, section)
        section_groups.setdefault(key, []).append((match, record))

    # For each section, rank candidates by keyword overlap and keep the top 2.
    # Large sections (e.g. Article VII Section 6 covers every salary cap exception)
    # may need more than one chunk to represent all relevant sub-rules.
    MAX_PER_SECTION = 2
    flat: list[tuple[float, dict, dict]] = []
    for candidates in section_groups.values():
        ranked = sorted(
            candidates,
            key=lambda x: (
                _keyword_overlap(x[1].get("text", ""), query_tokens),
                score_map[x[0]["id"]],
            ),
            reverse=True,
        )
        for match, record in ranked[:MAX_PER_SECTION]:
            flat.append((score_map[match["id"]], match, record))

    flat.sort(key=lambda x: x[0], reverse=True)

    results: list[RankedChunk] = []
    for hybrid_score, match, record in flat[:top_k]:
        chunk_id = match["id"]
        results.append(RankedChunk(
            rank          = len(results) + 1,
            score         = hybrid_score,
            chunk_id      = chunk_id,
            label         = record.get("label", match["metadata"].get("label", "")),
            article       = record.get("article", ""),
            article_title = record.get("article_title", ""),
            section       = record.get("section", ""),
            text          = record.get("text", "[text not found in chunk store]"),
        ))

    return results


def print_results(results: list[RankedChunk], show_full: bool = False) -> None:
    for r in results:
        bar = "=" * 72
        print(f"\n{bar}")
        print(f"#{r.rank}  score={r.score:.4f}  [{r.chunk_id}]")
        print(f"    {r.label}")
        print(bar)
        print(r.text if show_full else r.preview())


def main() -> None:

    parser = argparse.ArgumentParser(description="Query NBA CBA vector index.")
    parser.add_argument("question",    help="Natural language question to look up")
    parser.add_argument("--top-k",     type=int, default=5, metavar="N",
                        help="Number of results to return (default: 5)")
    parser.add_argument("--show-full", action="store_true",
                        help="Print full chunk text instead of a preview")
    parser.add_argument("--rerank", action="store_true",
                        help="Enable cross-encoder reranking (experimental; requires domain-matched model)")
    args = parser.parse_args()

    print(f"Loading model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Loading BM25 encoder...")
    bm25 = load_bm25()

    reranker = None
    if args.rerank:
        print(f"Loading reranker '{RERANKER_MODEL}'...")
        reranker = load_reranker()

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        sys.exit("Error: PINECONE_API_KEY not set.")

    pc    = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)
    store = load_chunk_store(CHUNK_STORE_PATH)

    print(f'\nQuery: "{args.question}"')
    print(f"Top-{args.top_k} results from '{INDEX_NAME}':\n")

    results = retrieve(
        args.question,
        top_k=args.top_k,
        model=model,
        bm25=bm25,
        reranker=reranker,
        index=index,
        chunk_store=store,
    )

    print_results(results, show_full=args.show_full)
    print()


if __name__ == "__main__":
    main()
