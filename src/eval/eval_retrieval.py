"""
eval_retrieval.py — retrieval quality sweep with IR metrics

Runs NBA CBA questions and reports per-query:
  - dot-product scores for each result
  - Hit@1/3/5, MRR, Recall@5 when ground_truth.json is present

Run setup_ground_truth.py first to generate ground_truth.json, then
annotate it with the correct (article, section) pairs per query.
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env.local")

os.environ.setdefault("USE_TF", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
from retrieve import retrieve, load_chunk_store, load_bm25, load_reranker, EMBED_MODEL, RERANKER_MODEL

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

CHUNK_STORE_PATH   = Path(__file__).parent.parent.parent / "data" / "processed" / "chunk_store.json"
BM25_PARAMS_PATH   = Path(__file__).parent.parent.parent / "data" / "processed" / "bm25_params.json"
GROUND_TRUTH_PATH  = Path(__file__).parent / "ground_truth.json"

# ------------------------------------------------------------------
# Test queries: (question, expected_topic_hint)
# ------------------------------------------------------------------
QUERIES = [
    # Salary / contracts
    ("What is the rookie scale salary?",                       "rookie / Article VIII"),
    ("What is the maximum salary a veteran player can earn?",  "max salary / Article VII"),
    ("What are the rules for Bird Rights?",                    "qualifying offer / Article VII"),
    ("What are the first apron penalties?",                    "apron / Article VII Section 2"),
    ("How does the mid-level exception work?",                 "MLE / Article VII"),
    ("What is a two-way contract?",                            "two-way / Article XXIX or similar"),
    ("What is the minimum player salary?",                     "minimums / Article II or VII"),

    # Trades / transactions
    ("When is the trade deadline?",                            "trade deadline / Article VII"),
    ("What players are eligible to be traded?",                "trade eligibility"),
    ("Can a team waive a player after the trade deadline?",    "waivers"),

    # CBA structure / governance
    ("What is the luxury tax threshold?",                      "tax threshold / Article VII"),
    ("How are grievances handled under the CBA?",              "grievance / Article XXXI"),
    ("What are the rules for player agents?",                  "agent / Article XXXV or similar"),

    # Drug testing / conduct
    ("What is the drug testing policy?",                       "anti-drug / Article XXXIII"),
    ("What happens if a player tests positive for marijuana?", "marijuana / Article XXXIII"),

    # Playoffs / eligibility
    ("How many players must be on a roster for the playoffs?", "roster / Article IV or similar"),
    ("What are the rules for two-way players in the playoffs?","two-way playoffs"),

    # Extensions
    ("What is a rookie scale extension?",                      "rookie scale extension / Article VII"),
    ("What is a designated veteran player extension?",         "supermax / Article VII Section 7"),

    # Free agency
    ("What are the rules for restricted free agency?",         "RFA / Article XI"),
    ("How does a sign-and-trade work?",                        "sign-and-trade / Article VII Section 8"),

    # Trade mechanics
    ("What is the traded player exception?",                   "TPE / Article VII Section 6"),
    ("How does salary matching work in trades?",               "salary matching / Article VII"),

    # Cap / salary mechanics
    ("How is team salary calculated?",                         "team salary / Article VII Section 4"),
    ("What are the rules for incentive compensation?",         "LTBI/ETBI / Article II + VII"),
    ("What happens to a player's salary if they are waived?",  "waiver salary / Article II Section 4"),
    ("What is the minimum team salary?",                       "team floor / Article VII Section 2"),
    ("What are the rules for non-guaranteed contracts?",       "non-guaranteed / Article II Section 4"),
    ("What is basketball related income?",                     "BRI / Article VII Section 1"),

    # G League
    ("Can a player be assigned to the G League?",              "G League / Article XLI"),

    # Governance / labor
    ("What are the anti-collusion provisions?",                "collusion / Article XIV"),
    ("What are the provisions for force majeure or work stoppage?", "force majeure / Article XXXIX"),

    # Conduct / training
    ("What are the rules for player conduct and discipline?",  "conduct / Article VI"),
    ("What are training camp rules?",                          "training camp / Article XX"),

    # Health / publicity
    ("What are the rules for player health and medical examinations?", "health / Article XXII"),
    ("What are the rules for player appearances and promotional obligations?", "appearances / Article XXXVII"),
    ("What are the rules for international player payments?",  "international payments / Article VII Section 3"),
]

TOP_K = 5


_SECTION_NUM_RE = re.compile(r'^(Section\s+\d+[\w.]*)', re.IGNORECASE)


def _section_key(section: str) -> str:
    """Normalize a section string to just its number prefix.

    "Section 4. Determination of Team Salary." → "Section 4"
    "Section 11."                               → "Section 11"
    "Section 1"                                 → "Section 1"
    """
    m = _SECTION_NUM_RE.match(section.strip())
    if m:
        return m.group(1).rstrip(".")
    return section.strip().rstrip(".")


def _parse_section_entry(entry) -> tuple[str, str] | None:
    """Parse one element of relevant_sections into (article, section_key).

    Accepts any of:
      ["ARTICLE VIII", "Section 1."]   — intended two-element format
      ["ARTICLE VIII, Section 1"]      — single combined string in a list
      "ARTICLE VIII, Section 1"        — bare combined string
    """
    if isinstance(entry, list):
        if len(entry) == 2 and all(isinstance(x, str) for x in entry):
            return (entry[0].strip(), _section_key(entry[1]))
        if len(entry) == 1 and isinstance(entry[0], str):
            entry = entry[0]  # fall through to string parsing
        else:
            return None

    if isinstance(entry, str):
        # "ARTICLE VII, Section 4"  or  "ARTICLE VII Section 4"
        m = re.match(
            r'^(ARTICLE\s+[IVXLCDM]+(?:\b[^\s,]*)?)\s*,?\s*(Section\s+\d+[\w.]*)',
            entry.strip(), re.IGNORECASE
        )
        if m:
            return (m.group(1).strip().upper(), _section_key(m.group(2)))

    return None


def load_ground_truth(path: Path) -> dict[str, set[tuple[str, str]]]:
    """Load ground_truth.json → {question: {(article, section_key), ...}}.

    Tolerates mixed annotation formats and prints a warning for any entry
    it cannot parse.
    """
    if not path.exists():
        return {}
    entries = json.loads(path.read_text())
    gt: dict[str, set[tuple[str, str]]] = {}
    for entry in entries:
        q = entry["question"]
        raw = entry.get("relevant_sections", [])
        # Support bare string at top level: "ARTICLE VII, Section 2"
        if isinstance(raw, str):
            raw = [raw]
        pairs: set[tuple[str, str]] = set()
        for item in raw:
            parsed = _parse_section_entry(item)
            if parsed:
                pairs.add(parsed)
            else:
                print(f"  [warn] could not parse ground truth entry: {item!r}")
        if pairs:
            gt[q] = pairs
    return gt


def ir_metrics(
    results: list,
    relevant: set[tuple[str, str]],
) -> dict:
    """Compute Hit@K, MRR, Recall@5 for a single query.

    Matching is done on (article, section_key) where section_key is the
    normalized section number prefix, e.g. "Section 4".
    """
    hit1 = hit3 = hit5 = mrr = 0.0
    found: set[tuple[str, str]] = set()

    for r in results:
        key = (r.article, _section_key(r.section))
        if key in relevant:
            found.add(key)
            if r.rank == 1:
                hit1 = 1.0
            if r.rank <= 3:
                hit3 = 1.0
            if r.rank <= 5:
                hit5 = 1.0
            if mrr == 0.0:
                mrr = 1.0 / r.rank

    recall5 = len(found) / len(relevant) if relevant else 0.0
    return {"hit1": hit1, "hit3": hit3, "hit5": hit5, "mrr": mrr, "recall5": recall5}


def main():
    print(f"Loading chunk store from {CHUNK_STORE_PATH}...")
    store = load_chunk_store(CHUNK_STORE_PATH)
    print(f"  {len(store)} chunks loaded.\n")

    print(f"Loading model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading BM25 encoder...")
    bm25 = load_bm25(BM25_PARAMS_PATH)

    # Cross-encoder reranker is domain-mismatched for CBA legal text with ms-marco.
    # Enable by passing reranker=load_reranker() once a better model is identified.
    reranker = None

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        sys.exit("Error: PINECONE_API_KEY not set.")
    pc    = Pinecone(api_key=pinecone_key)
    index = pc.Index("cba")

    stats = index.describe_index_stats()
    print(f"Pinecone index 'cba': {stats['total_vector_count']} vectors\n")

    gt = load_ground_truth(GROUND_TRUTH_PATH)
    if gt:
        print(f"Ground truth loaded: {len(gt)} annotated queries.")
    else:
        print(f"No ground truth found at {GROUND_TRUTH_PATH}.")
        print("  Run setup_ground_truth.py and annotate ground_truth.json for IR metrics.\n")
    print("=" * 80)

    all_scores_top1: list[float] = []
    all_scores_avg5: list[float] = []
    ir_rows: list[dict]          = []

    for q, hint in QUERIES:
        results = retrieve(q, top_k=TOP_K, model=model, bm25=bm25, reranker=reranker, index=index, chunk_store=store)

        scores = [r.score for r in results]
        top1   = scores[0] if scores else 0.0
        avg5   = sum(scores) / len(scores) if scores else 0.0
        all_scores_top1.append(top1)
        all_scores_avg5.append(avg5)

        print(f"Q: {q}")
        print(f"   expected topic : {hint}")
        for r in results:
            bar = "##" if r.rank == 1 else "  "
            print(f"   {bar} #{r.rank} score={r.score:.4f}  {r.label}")
        print(f"   top-1={top1:.4f}  avg-top5={avg5:.4f}")

        relevant = gt.get(q)
        if relevant is not None:
            m = ir_metrics(results, relevant)
            ir_rows.append(m)
            print(f"   Hit@1={m['hit1']:.0f}  Hit@3={m['hit3']:.0f}  Hit@5={m['hit5']:.0f}  "
                  f"MRR={m['mrr']:.3f}  Recall@5={m['recall5']:.3f}")
        else:
            print("   [no ground truth — skipping IR metrics]")
        print()

    # Summary
    n = len(QUERIES)
    print("=" * 80)
    print("SUMMARY")
    print(f"  Queries         : {n}")
    print(f"  Avg top-1 score : {sum(all_scores_top1)/n:.4f}")
    print(f"  Avg avg-5 score : {sum(all_scores_avg5)/n:.4f}")

    if ir_rows:
        na = len(ir_rows)
        print(f"\n  IR metrics ({na}/{n} queries annotated):")
        print(f"    Mean Hit@1   : {sum(r['hit1']   for r in ir_rows)/na:.3f}")
        print(f"    Mean Hit@3   : {sum(r['hit3']   for r in ir_rows)/na:.3f}")
        print(f"    Mean Hit@5   : {sum(r['hit5']   for r in ir_rows)/na:.3f}")
        print(f"    Mean MRR     : {sum(r['mrr']    for r in ir_rows)/na:.3f}")
        print(f"    Mean Recall@5: {sum(r['recall5']for r in ir_rows)/na:.3f}")


if __name__ == "__main__":
    main()
