"""
opik_eval.py — LLM-as-a-judge evaluation via Opik

Runs a scored experiment against the RAG pipeline using three judges:
  - AnswerRelevance  : is the answer relevant to the question?
  - Hallucination    : does the answer stay grounded in the retrieved context?
  - KeyInfoPresent   : does the answer contain a required key fact (when specified)?

The dataset lives in Opik and can be managed from the UI or re-seeded with
--push-dataset. Experiments run against whatever items are currently in the dataset.

Usage:
    python opik_eval.py                                    # run eval (agentic pipeline)
    python opik_eval.py --no-agent                         # run eval (single-shot pipeline)
    python opik_eval.py --experiment "post-reingest"       # custom experiment name
    python opik_eval.py --push-dataset                     # (re-)seed dataset, skip eval
    python opik_eval.py --push-dataset --experiment "..."  # seed + eval in one go
"""

import argparse
import os
import sys
from pathlib import Path

import anthropic
import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import AnswerRelevance, Hallucination
from opik.evaluation.metrics.base_metric import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env.local")
os.environ.setdefault("USE_TF", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))

from retrieve import load_bm25, load_chunk_store, CHUNK_STORE_PATH, BM25_PARAMS_PATH, EMBED_MODEL
from generate import rag, rag_agentic, DEFAULT_MODEL

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

DATASET_NAME = "cba-rag-eval"

# Seed queries. must_include: a key fact the answer must state; None = skip that check.
# Add more queries here or directly in the Opik UI.
SEED_QUERIES = [
    {
        "question": "How do Bird Rights work?",
        "must_include": (
            "Teams can exceed the salary cap to re-sign their own player "
            "who has been with them for at least 3 seasons "
            "(Qualifying Veteran Free Agent Exception)."
        ),
    },
    {
        "question": "What is the mid-level exception?",
        "must_include": (
            "The non-taxpayer mid-level exception allows teams over the salary cap "
            "to sign free agents up to a specified dollar amount."
        ),
    },
    {
        "question": "What is the luxury tax threshold?",
        "must_include": None,
    },
    {
        "question": "How does a sign-and-trade work?",
        "must_include": None,
    },
    {
        "question": "What is a rookie scale contract?",
        "must_include": None,
    },
    {
        "question": "What are the rules for restricted free agency?",
        "must_include": (
            "The prior team has the right to match any offer sheet "
            "signed by a restricted free agent."
        ),
    },
    {
        "question": "What happens if a player tests positive for marijuana?",
        "must_include": None,
    },
    {
        "question": "What is the supermax extension?",
        "must_include": (
            "The designated veteran player extension allows a team to sign their own "
            "player to a contract starting at a higher percentage of the salary cap "
            "than a standard maximum contract."
        ),
    },
]


# ── Metrics ────────────────────────────────────────────────────────────────────

class KeyInfoPresent(BaseMetric):
    """LLM judge: checks whether a required key fact appears in the answer.

    Scores 1.0 if the answer conveys the required information (even in different
    words), 0.0 if absent or unclear. Skips (scores 1.0) when reference is None.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        super().__init__(name="key_info_present")
        self._model = model

    def score(self, output: str, reference: str | None = None, **kwargs) -> ScoreResult:
        if not reference:
            return ScoreResult(
                value=1.0,
                name=self.name,
                reason="No required information specified for this query.",
            )

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        prompt = (
            f"Does the following answer clearly convey this key piece of information?\n\n"
            f"Required information: {reference}\n\n"
            f"Answer:\n{output}\n\n"
            "Reply YES if the answer conveys it (even in different words), "
            "or NO if it is absent or unclear. Then give a one-sentence explanation.\n"
            "Format: YES/NO: <explanation>"
        )
        response = client.messages.create(
            model=self._model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        value = 1.0 if text.upper().startswith("YES") else 0.0
        return ScoreResult(value=value, name=self.name, reason=text)


# ── Dataset ────────────────────────────────────────────────────────────────────

def push_dataset(opik_client: opik.Opik) -> opik.Dataset:
    """Recreate the eval dataset from SEED_QUERIES.

    Deletes the existing dataset first so re-runs don't accumulate duplicates.
    Any queries added via the Opik UI will be lost — manage those in SEED_QUERIES
    if you want them to persist across seeds.
    """
    try:
        opik_client.delete_dataset(DATASET_NAME)
        print(f"Deleted existing dataset '{DATASET_NAME}'.")
    except Exception:
        pass  # dataset didn't exist yet

    dataset = opik_client.get_or_create_dataset(DATASET_NAME)
    dataset.insert([
        {
            "input":     {"question": q["question"]},
            "reference": q["must_include"],
        }
        for q in SEED_QUERIES
    ])
    print(f"Pushed {len(SEED_QUERIES)} items to dataset '{DATASET_NAME}'.")
    return dataset


# ── Task ───────────────────────────────────────────────────────────────────────

def make_task(embed_model, bm25, index, store, anthropic_client, *, use_agent: bool = True):
    """Return a task function that calls the RAG pipeline and returns scored fields.

    Field contract (keys Opik metrics read from the merged item + task output):
        input     → str  question (overrides the dataset item's dict so metrics get a string)
        output    → str  generated answer
        context   → list[str]  retrieved chunk texts (for AnswerRelevance / Hallucination)
        reference → str | None  required key fact (for KeyInfoPresent)
    """
    pipeline = rag_agentic if use_agent else rag

    def task(sample: dict) -> dict:
        question = sample["input"]["question"]
        result = pipeline(
            question,
            embed_model=embed_model,
            bm25=bm25,
            index=index,
            chunk_store=store,
            client=anthropic_client,
        )
        return {
            "input":     question,                          # string expected by built-in judges
            "output":    result["answer"],
            "context":   [c.text for c in result["chunks"]],
            "reference": sample.get("reference"),
        }

    return task


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge eval via Opik.")
    parser.add_argument("--experiment", default="cba-rag", help="Experiment name in Opik")
    parser.add_argument("--no-agent",   action="store_true", help="Use single-shot pipeline")
    parser.add_argument("--push-dataset", action="store_true",
                        help="Recreate the Opik dataset from SEED_QUERIES before evaluating")
    parser.add_argument("--push-dataset-only", action="store_true",
                        help="Recreate the dataset then exit without running eval")
    args = parser.parse_args()

    opik_client = opik.Opik()

    if args.push_dataset or args.push_dataset_only:
        push_dataset(opik_client)
        if args.push_dataset_only:
            return

    dataset = opik_client.get_dataset(DATASET_NAME)

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print("Loading BM25 encoder...")
    bm25 = load_bm25(BM25_PARAMS_PATH)

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        sys.exit("Error: PINECONE_API_KEY not set.")

    index            = Pinecone(api_key=pinecone_key).Index("cba")
    store            = load_chunk_store(CHUNK_STORE_PATH)
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    use_agent = not args.no_agent
    task = make_task(embed_model, bm25, index, store, anthropic_client, use_agent=use_agent)

    metrics = [
        AnswerRelevance(),   # input, output, context
        Hallucination(),     # input, output, context
        KeyInfoPresent(),    # output, reference
    ]

    evaluate(
        experiment_name=args.experiment,
        dataset=dataset,
        task=task,
        scoring_metrics=metrics,
        experiment_config={
            "pipeline": "agentic" if use_agent else "single-shot",
            "model":    DEFAULT_MODEL,
            "top_k":    7,
        },
    )
    print(f"\nExperiment '{args.experiment}' complete. View results in the Opik UI.")


if __name__ == "__main__":
    main()
