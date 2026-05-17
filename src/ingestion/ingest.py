"""
ingest.py — NBA CBA 2023 ingestion pipeline
Extracts text via PyMuPDF, chunks by Article/Section, embeds with a Q&A-optimized
model, generates BM25 sparse vectors, and upserts both to Pinecone for hybrid search.
Full chunk text is stored in a local chunk_store.json keyed by chunk_id.

Usage:
    python ingest.py --pdf ../../data/raw/nba_cba_2023.pdf
    python ingest.py --pdf ../../data/raw/nba_cba_2023.pdf --recreate-index   # delete + rebuild index
    python ingest.py --pdf ../../data/raw/nba_cba_2023.pdf --dry-run          # chunk only, no Pinecone
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env.local")

import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec

# Transformers pulls TensorFlow/Keras by default; this pipeline only needs PyTorch.
os.environ.setdefault("USE_TF", "0")
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_NAME       = "cba"
EMBED_MODEL      = "multi-qa-mpnet-base-dot-v1"   # 768-dim, Q&A-optimised, dot product
EMBED_DIM        = 768
INDEX_METRIC     = "dotproduct"
INDEX_CLOUD      = "aws"
INDEX_REGION     = "us-east-1"
BATCH_SIZE       = 100
CHUNK_STORE_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "chunk_store.json"
BM25_PARAMS_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "bm25_params.json"

# Chunking size limits
MAX_CHUNK_CHARS  = 1500   # hard ceiling; sections over this are split recursively
OVERLAP_CHARS    = 200    # tail of previous piece prepended to next for context
MIN_CHUNK_CHARS  = 150    # stubs below this are merged into the previous chunk or dropped

SPLIT_SEPARATORS = ["\n\n", "\n", ". "]   # tried in order during recursive split

SECTION_TITLE_RE   = re.compile(r'^Section\s+\d+[\w.]*\.\s+.{1,60}\.$')
SECTION_CONTENT_RE = re.compile(r'^Section\s+\d+[\w.]*\.')
ARTICLE_RE         = re.compile(r'^(ARTICLE\s+[IVXLCDM]+)\s*$', re.IGNORECASE)
ARTICLE_TITLE_RE   = re.compile(r"^[A-Z][A-Za-z'\-,&\s]{0,58}$")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_text_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    article: str
    article_title: str
    section: str
    text: str
    chunk_id: str
    section_intro: str = ""   # first ~300 chars of the section's opening chunk; empty for the intro chunk itself

    @property
    def label(self) -> str:
        parts = [p for p in [self.article, self.article_title, self.section] if p]
        return " › ".join(parts)

    @property
    def embed_text(self) -> str:
        """Text used for dense embedding.

        Every chunk is prefixed with its article/section header. For chunks that
        are not the first chunk of their section (i.e. the section was split due
        to size), the opening text of the section is also injected. This ensures
        sub-rule chunks in large sections carry the section's purpose in their
        vector — e.g. every Article VII Section 6 chunk captures "Exceptions to
        the Salary Cap / Veteran Free Agent Exception" even if those words only
        appear in the section's first chunk.
        """
        header_parts = []
        if self.article:
            line = self.article
            if self.article_title:
                line += f": {self.article_title}"
            header_parts.append(line)
        if self.section:
            header_parts.append(self.section)
        header = "\n".join(header_parts)

        if self.section_intro:
            return f"{header}\n\nSection opening: {self.section_intro}\n\n{self.text}" if header else f"Section opening: {self.section_intro}\n\n{self.text}"
        return f"{header}\n\n{self.text}" if header else self.text

    def to_store_record(self) -> dict:
        """Full record written to chunk_store.json.

        Non-intro chunks prepend a one-line section heading so Claude sees
        which section the sub-rule belongs to without reading a second chunk.
        """
        if self.section_intro:
            # Show enough of the section opening that Claude can identify the
            # section's purpose without reading a separate chunk.
            intro_display = self.section_intro[:200].rsplit(" ", 1)[0] + "…"
            display_text = f"[Section context: {intro_display}]\n\n{self.text}"
        else:
            display_text = self.text
        return {
            "chunk_id":      self.chunk_id,
            "article":       self.article,
            "article_title": self.article_title,
            "section":       self.section,
            "label":         self.label,
            "text":          display_text,
        }

    def to_pinecone_metadata(self) -> dict:
        """
        Lightweight metadata only — no full text.
        Pinecone's 40KB limit is easily hit by long legal clauses,
        so full text lives in chunk_store.json instead.
        """
        return {
            "chunk_id":      self.chunk_id,
            "article":       self.article,
            "article_title": self.article_title,
            "section":       self.section,
            "label":         self.label,
        }

def chunk_by_section(text: str) -> list[Chunk]:
    """
    First pass: split only at Article and Section boundaries.
    Subsections (a)(b)(i)(ii)... are kept with their parent section so the
    embedding captures the full context of the provision.
    Oversized sections are handled in post_process_chunks.
    """
    lines = [l.strip() for l in text.splitlines()]
    raw: list[Chunk] = []
    chunk_idx = 0

    current_article       = ""
    current_article_title = ""
    current_section       = ""
    body_lines: list[str] = []

    def flush():
        nonlocal chunk_idx
        body = re.sub(r'\s+', ' ', " ".join(body_lines).strip())
        if len(body) >= 50 and (current_article or current_section):
            raw.append(Chunk(
                article       = current_article,
                article_title = current_article_title,
                section       = current_section,
                text          = body,
                chunk_id      = f"chunk-{chunk_idx:05d}",
            ))
            chunk_idx += 1
        body_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        if not line:
            i += 1
            continue

        if ARTICLE_RE.match(line):
            flush()
            if line == line.upper():
                current_article       = line.upper()
                current_article_title = ""
                current_section       = ""
                j = i + 1
                while j < len(lines) and not lines[j]:
                    j += 1
                if j < len(lines):
                    peek = lines[j]
                    if (not ARTICLE_RE.match(peek)
                            and not SECTION_CONTENT_RE.match(peek)
                            and ARTICLE_TITLE_RE.match(peek.title())):
                        current_article_title = peek.title()
                        i = j
            i += 1
            continue

        if SECTION_CONTENT_RE.match(line):
            flush()
            current_section = line
            if not SECTION_TITLE_RE.match(line):
                body_lines.append(line)
            i += 1
            continue

        # Subsections (a)(b)(i)(ii) etc. are intentionally NOT flushed here.
        # They accumulate in body_lines and stay with their parent section.
        body_lines.append(line)
        i += 1

    flush()
    return inject_section_intros(post_process_chunks(raw))


def _split_recursive(text: str, separators: list[str], max_size: int) -> list[str]:
    """Split text on separators in order until every piece fits within max_size."""
    if len(text) <= max_size:
        return [text]

    for i, sep in enumerate(separators):
        if sep not in text:
            continue

        pieces, current = [], ""
        for part in text.split(sep):
            candidate = (current + sep + part) if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                next_seps = separators[i + 1:]
                if len(part) > max_size:
                    if next_seps:
                        pieces.extend(_split_recursive(part, next_seps, max_size))
                    else:
                        # No separators left — hard-split so no piece exceeds max_size
                        pieces.extend(part[j: j + max_size] for j in range(0, len(part), max_size))
                    current = ""
                else:
                    current = part
        if current:
            pieces.append(current)
        return pieces or [text]

    # Last resort: hard character split
    return [text[j: j + max_size] for j in range(0, len(text), max_size)]


def split_with_overlap(text: str) -> list[str]:
    """Split text into MAX_CHUNK_CHARS pieces, prepending OVERLAP_CHARS of the
    previous piece to each successor so context bleeds across split points."""
    pieces = _split_recursive(text, SPLIT_SEPARATORS, MAX_CHUNK_CHARS)
    if len(pieces) <= 1:
        return pieces
    overlapped = [pieces[0]]
    for i in range(1, len(pieces)):
        tail = pieces[i - 1][-OVERLAP_CHARS:]
        overlapped.append(tail + "\n" + pieces[i])
    return overlapped


def post_process_chunks(raw: list[Chunk]) -> list[Chunk]:
    """
    Second pass:
    - Split any chunk exceeding MAX_CHUNK_CHARS using split_with_overlap.
    - Absorb or drop chunks below MIN_CHUNK_CHARS.
    - Re-assign sequential chunk IDs.
    """
    result: list[Chunk] = []
    idx = 0

    for chunk in raw:
        pieces = (
            split_with_overlap(chunk.text)
            if len(chunk.text) > MAX_CHUNK_CHARS
            else [chunk.text]
        )

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            # Absorb tiny piece into the previous chunk if same section
            if len(piece) < MIN_CHUNK_CHARS and result:
                prev = result[-1]
                if prev.article == chunk.article and prev.section == chunk.section:
                    result[-1] = Chunk(
                        article       = prev.article,
                        article_title = prev.article_title,
                        section       = prev.section,
                        text          = prev.text + " " + piece,
                        chunk_id      = prev.chunk_id,
                    )
                    continue

            if len(piece) < MIN_CHUNK_CHARS:
                continue  # drop unreachable orphan stubs

            result.append(Chunk(
                article       = chunk.article,
                article_title = chunk.article_title,
                section       = chunk.section,
                text          = piece,
                chunk_id      = f"chunk-{idx:05d}",
            ))
            idx += 1

    return result


SECTION_INTRO_CHARS = 300   # how much of the first chunk to inject into later chunks

def inject_section_intros(chunks: list[Chunk]) -> list[Chunk]:
    """Third pass: for sections split across multiple chunks, bake the opening
    text of the first chunk into every subsequent chunk via section_intro.

    This ensures sub-rule chunks carry their section's purpose in both their
    dense embedding and their chunk_store display text — solving the problem
    where a generic sub-rule chunk (e.g. Article VII Section 6 subsection 3)
    doesn't score well for queries about the section's overall topic (e.g.
    'salary cap exceptions / Bird Rights') because that context only lives
    in the section's first chunk.
    """
    # Group chunk indices by (article, section), preserving order
    from collections import defaultdict
    section_order: dict[tuple, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        section_order[(c.article, c.section)].append(i)

    result = list(chunks)
    for indices in section_order.values():
        if len(indices) < 2:
            continue
        intro_text = chunks[indices[0]].text[:SECTION_INTRO_CHARS]
        for i in indices[1:]:
            c = result[i]
            result[i] = Chunk(
                article       = c.article,
                article_title = c.article_title,
                section       = c.section,
                text          = c.text,
                chunk_id      = c.chunk_id,
                section_intro = intro_text,
            )

    return result


# ---------------------------------------------------------------------------
# Chunk store
# ---------------------------------------------------------------------------

def save_chunk_store(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = {c.chunk_id: c.to_store_record() for c in chunks}
    path.write_text(json.dumps(store, indent=2))
    print(f"  Chunk store saved to '{path}' ({len(store)} records).")


def load_chunk_store(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def fit_and_save_bm25(chunks: list[Chunk], path: str) -> BM25Encoder:
    print(f"Fitting BM25 encoder on {len(chunks)} chunks...")
    bm25 = BM25Encoder()
    bm25.fit([c.text for c in chunks])
    bm25.dump(path)
    print(f"  BM25 params saved to '{path}'.")
    return bm25


# ---------------------------------------------------------------------------
# Pinecone index management
# ---------------------------------------------------------------------------

def ensure_index(pc: Pinecone, recreate: bool = False) -> None:
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME in existing:
        if recreate:
            print(f"--recreate-index: deleting '{INDEX_NAME}'...")
            pc.delete_index(INDEX_NAME)
            print(f"  Deleted.")
        else:
            return

    print(f"Creating index '{INDEX_NAME}' ({EMBED_DIM}d, {INDEX_METRIC})...")
    pc.create_index(
        name      = INDEX_NAME,
        dimension = EMBED_DIM,
        metric    = INDEX_METRIC,
        spec      = ServerlessSpec(cloud=INDEX_CLOUD, region=INDEX_REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
    print(f"  Index ready.")


# ---------------------------------------------------------------------------
# Upsert (dense + sparse)
# ---------------------------------------------------------------------------

def upsert(chunks: list[Chunk], pc: Pinecone, model: SentenceTransformer, bm25: BM25Encoder) -> None:
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    print(f"Connected to '{INDEX_NAME}'. Vectors before upsert: {stats['total_vector_count']}")

    print(f"Embedding and upserting {len(chunks)} chunks (batch={BATCH_SIZE})...")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch       = chunks[start : start + BATCH_SIZE]
        texts       = [c.text for c in batch]
        embed_texts = [c.embed_text for c in batch]
        dense_vecs  = model.encode(embed_texts, show_progress_bar=False).tolist()
        sparse_vecs = bm25.encode_documents(texts)

        vectors = [
            {
                "id":           c.chunk_id,
                "values":       dense,
                "sparse_values": sparse,
                "metadata":     c.to_pinecone_metadata(),
            }
            for c, dense, sparse in zip(batch, dense_vecs, sparse_vecs)
        ]
        index.upsert(vectors=vectors)

        uploaded = min(start + BATCH_SIZE, len(chunks))
        print(f"  {uploaded}/{len(chunks)}", end="\r")

    print()
    stats = index.describe_index_stats()
    print(f"Done. '{INDEX_NAME}' now contains {stats['total_vector_count']} vectors.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest NBA CBA PDF into Pinecone.")
    parser.add_argument("--pdf", required=True, help="Path to nba_cba_2023.pdf")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract and chunk only — skip Pinecone upsert")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all vectors before upserting (keeps index structure)")
    parser.add_argument("--recreate-index", action="store_true",
                        help="Delete and recreate the Pinecone index (required when changing model/dimensions)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        sys.exit(f"Error: PDF not found at {args.pdf!r}")

    # 1. Extract
    print(f"Extracting text from {args.pdf}...")
    raw_text = extract_text_pymupdf(args.pdf)
    print(f"  {len(raw_text):,} characters extracted.")

    # 2. Chunk
    print("Chunking by Article/Section...")
    chunks = chunk_by_section(raw_text)
    print(f"  {len(chunks)} chunks produced.")

    print("\nSample chunks:")
    for c in chunks[:3]:
        print(f"  [{c.label}]")
        print(f"  {c.text[:100]}...")
    print()

    # 3. Save chunk store
    print("Saving chunk store...")
    save_chunk_store(chunks, CHUNK_STORE_PATH)

    # 4. Fit + save BM25 (needed at query time regardless of dry-run)
    bm25 = fit_and_save_bm25(chunks, BM25_PARAMS_PATH)

    if args.dry_run:
        print("Dry-run — skipping Pinecone upsert.")
        return

    # 5. Embed + upsert
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        sys.exit("Error: PINECONE_API_KEY environment variable not set.")

    pc = Pinecone(api_key=pinecone_key)
    ensure_index(pc, recreate=args.recreate_index)

    if args.reset and not args.recreate_index:
        print(f"--reset: clearing all vectors in '{INDEX_NAME}'...")
        index = pc.Index(INDEX_NAME)
        index.delete(delete_all=True)
        stats = index.describe_index_stats()
        print(f"  Cleared. Vectors now: {stats['total_vector_count']}")

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    upsert(chunks, pc, model, bm25)


if __name__ == "__main__":
    main()
