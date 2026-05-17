"""
opik_eval.py — LLM-as-a-judge evaluation via Opik

Runs a scored experiment against the RAG pipeline using three judges:
  - AnswerRelevance  : is the answer relevant to the question?
  - Hallucination    : does the answer stay grounded in the retrieved context?
  - KeyInfoPresent   : does the answer contain a required key fact (when specified)?

Dataset management is done manually in the Opik UI. Create a dataset named
"cba-rag-eval" with items in this format:
  input     → {"question": "..."}   — the question to ask the RAG pipeline
  reference → "key fact string"     — optional; omit or set null to skip KeyInfoPresent

Usage:
    python opik_eval.py                              # run eval (agentic pipeline)
    python opik_eval.py --no-agent                   # run eval (single-shot pipeline)
    python opik_eval.py --experiment "post-reingest" # custom experiment name
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
    args = parser.parse_args()

    opik_client = opik.Opik()
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
            "top_k":    10,
        },
    )
    print(f"\nExperiment '{args.experiment}' complete. View results in the Opik UI.")


if __name__ == "__main__":
    main()
