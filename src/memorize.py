"""memorize.py — curate V8 commit analyses into a long-term vector memory.

Watches a repowatcher output directory for new .md analysis files, runs a
curation agent to decide what (if anything) is worth storing as a lasting
engineering insight, and writes distilled entries to a local Qdrant DB.

Usage:
    # Watch mode — continuous (Ctrl-C to stop)
    memorize --db PATH --output-dir PATH [--poll SECONDS]

    # One-shot — process specific files
    memorize --db PATH FILE [FILE ...]

    # Maintenance
    memorize --db PATH --inspect
    memorize --db PATH --search QUERY [--subsystem S] [--limit N]
    memorize --db PATH --list [--subsystem S] [--type T] [--limit N]
    memorize --db PATH --delete ID
    memorize --db PATH --delete-commit HASH
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastembed import TextEmbedding as _FE
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    VectorParams,
)

# ── constants ─────────────────────────────────────────────────────────────────

CURATION_MODEL = "google_genai:gemini-3-flash-preview"
DEDUPE_THRESHOLD = 0.82  # cosine similarity above which we check for overlap
DEDUPE_K = 3  # how many similar entries to surface for the dedup decision
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # local fastembed, 768 dims
COLLECTION = "v8_memory"
VECTOR_SIZE = 768

SUBSYSTEMS = [
    "gc",
    "scavenger",
    "minorms",
    "ignition",
    "maglev",
    "turbofan",
    "turboshaft",
    "liftoff",
    "wasm",
    "ic",
    "builtins",
    "csa",
    "torque",
    "parser",
    "api",
    "runtime",
    "sandbox",
    "profiler",
    "debug",
]
TYPES = ["performance", "correctness", "design", "gotcha", "api-change", "refactor"]

DEDUPE_SYSTEM = """\
You are maintaining a V8 engineering knowledge base. A new insight is ready to be stored,
but similar entries already exist. Decide how to proceed:

- add: the new entry covers genuinely new ground not captured by the existing entries
- skip: the existing entries already cover this insight well enough
- replace: the new entry supersedes one or more existing entries — provide their IDs

Prefer a compact, high-quality knowledge base over accumulating redundant facts.
When the existing entries are substantively equivalent, choose skip.
When the new entry is clearly more complete or more accurate, choose replace.
"""

CURATION_SYSTEM = f"""\
You are a V8 engine expert building a long-term engineering knowledge base.

You will be given an expert analysis of a V8 git commit. Your job is to decide
whether the analysis contains a lasting insight worth storing — something that will
be useful when a V8 engineer asks a question weeks or months from now, unaware of
this specific commit.

Store an entry if the analysis reveals:
- A non-obvious invariant, constraint, or gotcha in a V8 subsystem
- An architectural pattern, design decision, or tradeoff with lasting relevance
- A subtle correctness or performance consideration for a specific area
- Context that explains *why* something is the way it is

Do NOT store if:
- The analysis is shallow or mostly restates the commit message
- The change is purely mechanical (rename, format, version bump)
- The insight is too commit-specific to be useful without reading the diff

If storing: write the insight as a self-contained engineering note — as if writing
for a wiki. Do not reference "this commit" or "this change". Write what is true
about V8, informed by what you learned from the commit.

Subsystems: {", ".join(SUBSYSTEMS)}
Types: {", ".join(TYPES)}
"""


# ── pydantic schema for structured curation output ────────────────────────────


class _Curation(BaseModel):
    store: bool = Field(description="Whether this analysis contains a lasting insight")
    text: str | None = Field(
        None,
        description="Self-contained engineering insight (wiki-style, not commit-centric)",
    )
    subsystems: list[str] = Field(
        default_factory=list, description="V8 subsystems this insight applies to"
    )
    type: str | None = Field(None, description="One of: " + ", ".join(TYPES))


class _DedupeDecision(BaseModel):
    action: str = Field(description="One of: add, skip, replace")
    replace_ids: list[str] = Field(
        default_factory=list,
        description="Qdrant point IDs of existing entries to remove (replace action only)",
    )
    reason: str = Field(description="One sentence explanation")


# ── embeddings (local fastembed, no API key needed) ───────────────────────────


class _FastEmbeddings(Embeddings):
    """Thin LangChain-compatible wrapper around fastembed.TextEmbedding."""

    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self._model = _FE(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()


# ── qdrant setup ──────────────────────────────────────────────────────────────


def _make_client(db_path: Path) -> QdrantClient:
    db_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(db_path))


def _ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _make_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client, collection_name=COLLECTION, embedding=_FastEmbeddings()
    )


# ── frontmatter parsing ───────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a .md file. Returns (meta_dict, body)."""
    if not text.startswith("---\n"):
        return {}, text
    try:
        end = text.index("\n---\n", 4)
    except ValueError:
        return {}, text
    meta: dict = {}
    for line in text[4:end].splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            meta[k.strip()] = v.strip()
    return meta, text[end + 5 :]


# ── curation agent ────────────────────────────────────────────────────────────


def _curate(analysis_body: str) -> tuple[_Curation, dict]:
    model = init_chat_model(CURATION_MODEL).with_structured_output(
        _Curation, include_raw=True
    )
    result = model.invoke(
        [
            SystemMessage(content=CURATION_SYSTEM),
            HumanMessage(content=analysis_body.strip()),
        ]
    )
    return result["parsed"], result["raw"].usage_metadata or {}


def _dedupe(
    new_text: str, similar: list[tuple[str, str, float]]
) -> tuple[_DedupeDecision, dict]:
    model = init_chat_model(CURATION_MODEL).with_structured_output(
        _DedupeDecision, include_raw=True
    )
    existing = "\n\n---\n\n".join(
        f"[id={sid}  similarity={score:.2f}]\n{text}" for sid, text, score in similar
    )
    result = model.invoke(
        [
            SystemMessage(content=DEDUPE_SYSTEM),
            HumanMessage(
                content=f"New entry:\n{new_text}\n\nExisting similar entries:\n{existing}"
            ),
        ]
    )
    return result["parsed"], result["raw"].usage_metadata or {}


# ── ingestion ─────────────────────────────────────────────────────────────────


def _fmt_tokens(usage: dict) -> str:
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    return f"tokens in={inp} out={out}"


def ingest_file(
    path: Path, client: QdrantClient, store: QdrantVectorStore, verbose: bool = False
) -> bool:
    """Curate and store a single .md file. Returns True if an entry was stored."""
    text = path.read_text()
    meta, body = _parse_frontmatter(text)

    if not body.strip():
        logging.info("%s — empty body, skipping", path.name)
        return False

    logging.info("%s — curating …", path.name)
    try:
        result, usage = _curate(body)
    except Exception as e:
        logging.warning("%s — curation failed: %s", path.name, e)
        return False

    if verbose:
        print(f"{path.name}  {_fmt_tokens(usage)}")

    if not result.store or not result.text:
        logging.info("%s → SKIP", path.name)
        if verbose:
            print(f"  → skip")
        return False

    # Dedup: check for similar existing entries before storing
    if client.collection_exists(COLLECTION):
        emb = _FastEmbeddings().embed_query(result.text)
        hits = client.query_points(
            COLLECTION,
            query=emb,
            limit=DEDUPE_K,
            score_threshold=DEDUPE_THRESHOLD,
            with_payload=True,
        ).points
        if hits:
            similar = [
                (str(h.id), (h.payload or {}).get("page_content", ""), h.score)
                for h in hits
            ]
            try:
                decision, dedup_usage = _dedupe(result.text, similar)
            except Exception as e:
                logging.warning("%s — dedup failed: %s — storing anyway", path.name, e)
                decision = None

            if decision is not None:
                logging.info(
                    "%s → dedup: %s — %s", path.name, decision.action, decision.reason
                )
                if verbose:
                    print(
                        f"  dedup → {decision.action}: {decision.reason}  {_fmt_tokens(dedup_usage)}"
                    )
                if decision.action == "skip":
                    return False
                if decision.action == "replace" and decision.replace_ids:
                    client.delete(
                        COLLECTION,
                        points_selector=PointIdsList(points=decision.replace_ids),
                    )
                    logging.info(
                        "%s → removed %d superseded entries",
                        path.name,
                        len(decision.replace_ids),
                    )

    logging.info(
        "%s → STORE [%s / %s]", path.name, ", ".join(result.subsystems), result.type
    )
    if verbose:
        print(f"  → store [{', '.join(result.subsystems)} / {result.type}]")
        print(f"  {result.text}")

    from langchain_core.documents import Document

    doc = Document(
        page_content=result.text,
        metadata={
            "commit": meta.get("commit", ""),
            "date": meta.get("date", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "subsystems": result.subsystems,
            "type": result.type or "",
            "source_file": path.name,
            "stored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    store.add_documents([doc])
    return True


# ── state (tracks which .md files have been processed) ────────────────────────


class _State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict = {"processed": []}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception:
                pass

    @property
    def processed(self) -> set[str]:
        with self._lock:
            return set(self._data.get("processed", []))

    def mark(self, filename: str) -> None:
        with self._lock:
            lst = self._data.setdefault("processed", [])
            if filename not in lst:
                lst.append(filename)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2))
            tmp.replace(self.path)


# ── modes ─────────────────────────────────────────────────────────────────────


def run_files(
    paths: list[Path],
    db_path: Path,
    client: QdrantClient,
    store: QdrantVectorStore,
    verbose: bool = False,
) -> None:
    state = _State(db_path / "memorize_state.json")
    for path in paths:
        if path.name in state.processed:
            logging.info("%s — already processed, skipping", path.name)
            continue
        ingest_file(path, client, store, verbose=verbose)
        state.mark(path.name)


def run_watch(
    output_dir: Path,
    db_path: Path,
    client: QdrantClient,
    store: QdrantVectorStore,
    poll: int,
    verbose: bool = False,
) -> None:
    state = _State(db_path / "memorize_state.json")
    stop = threading.Event()

    def _sig(signum, frame):
        logging.info("Signal %s — stopping …", signum)
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    logging.info("Watching %s every %ds …", output_dir, poll)
    while not stop.is_set():
        processed = state.processed
        for md in sorted(output_dir.glob("*.md")):
            if stop.is_set():
                break
            if md.name in processed:
                continue
            try:
                ingest_file(md, client, store, verbose=verbose)
                state.mark(md.name)
            except Exception as e:
                logging.warning("%s — error: %s (will retry)", md.name, e)
        stop.wait(poll)

    logging.info("Exited.")


# ── maintenance ───────────────────────────────────────────────────────────────


def cmd_inspect(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        print("Collection does not exist yet.")
        return

    info = client.get_collection(COLLECTION)
    total = info.points_count
    print(f"Entries: {total}")
    if not total:
        return

    # Scroll all points to aggregate metadata
    subsystem_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    dates: list[str] = []
    offset = None

    while True:
        points, offset = client.scroll(
            COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=100,
            offset=offset,
        )
        for p in points:
            m = (p.payload or {}).get("metadata", {})
            for s in m.get("subsystems", []):
                subsystem_counts[s] = subsystem_counts.get(s, 0) + 1
            t = m.get("type", "")
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1
            if m.get("date"):
                dates.append(m["date"])
        if offset is None:
            break

    if dates:
        print(f"Date range: {min(dates)} → {max(dates)}")

    if subsystem_counts:
        print("\nBy subsystem:")
        for s, n in sorted(subsystem_counts.items(), key=lambda x: -x[1]):
            print(f"  {s:<16} {n}")

    if type_counts:
        print("\nBy type:")
        for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:<16} {n}")


def cmd_search(
    client: QdrantClient,
    store: QdrantVectorStore,
    query: str,
    subsystem: str | None,
    limit: int,
) -> None:
    qdrant_filter = None
    if subsystem:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.subsystems", match=MatchValue(value=subsystem)
                )
            ]
        )
    results = store.similarity_search_with_score(query, k=limit, filter=qdrant_filter)
    for doc, score in results:
        m = doc.metadata
        print(
            f"\n[{score:.3f}] {m.get('date', '')}  {', '.join(m.get('subsystems', []))}  {m.get('type', '')}"
        )
        print(f"  source: {m.get('source_file', '')}")
        print(f"  {doc.page_content[:300].replace(chr(10), ' ')}")


def cmd_list(
    client: QdrantClient, subsystem: str | None, type_: str | None, limit: int | None
) -> None:
    qdrant_filter = None
    must = []
    if subsystem:
        must.append(
            FieldCondition(key="metadata.subsystems", match=MatchValue(value=subsystem))
        )
    if type_:
        must.append(FieldCondition(key="metadata.type", match=MatchValue(value=type_)))
    if must:
        qdrant_filter = Filter(must=must)

    shown = 0
    offset = None
    while True:
        batch, offset = client.scroll(
            COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=100,
            offset=offset,
            scroll_filter=qdrant_filter,
        )
        for p in batch:
            m = (p.payload or {}).get("metadata", {})
            snippet = (p.payload or {}).get("page_content", "")[:80].replace("\n", " ")
            print(
                f"{p.id}  {m.get('date', '')}  [{', '.join(m.get('subsystems', []))}]  {m.get('type', '')}  {snippet}"
            )
            shown += 1
            if limit is not None and shown >= limit:
                return
        if offset is None:
            break


def cmd_delete(client: QdrantClient, point_id: str) -> None:
    client.delete(COLLECTION, points_selector=PointIdsList(points=[point_id]))
    print(f"Deleted {point_id}")


def cmd_delete_commit(client: QdrantClient, commit_hash: str) -> None:
    client.delete(
        COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.commit", match=MatchValue(value=commit_hash)
                    )
                ]
            )
        ),
    )
    print(f"Deleted all entries for commit {commit_hash}")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate V8 commit analyses into a vector memory DB."
    )
    parser.add_argument(
        "--db", required=True, type=Path, help="Path to Qdrant DB directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="repowatcher output dir to watch"
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Poll interval in watch mode (default: 30)",
    )
    parser.add_argument(
        "--model",
        default=CURATION_MODEL,
        help=f"Curation model (default: {CURATION_MODEL})",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print curator decisions and token counts to stdout",
    )

    # Maintenance flags (mutually exclusive with watch/process)
    parser.add_argument("--inspect", action="store_true", help="Show DB stats and exit")
    parser.add_argument("--search", metavar="QUERY", help="Semantic search and exit")
    parser.add_argument("--list", action="store_true", help="List entries and exit")
    parser.add_argument("--delete", metavar="ID", help="Delete entry by point ID")
    parser.add_argument(
        "--delete-commit", metavar="HASH", help="Delete all entries for a commit"
    )

    # Filters used by --search and --list
    parser.add_argument("--subsystem", help="Filter by subsystem")
    parser.add_argument("--type", dest="type_", help="Filter by type (for --list)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Result limit (default: 5 for --search, unlimited for --list)",
    )

    # Positional: specific files to process
    parser.add_argument("files", nargs="*", type=Path, help=".md files to process")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = args.db.resolve()
    client = _make_client(db_path)

    # Maintenance commands (no embedding needed for delete/inspect)
    if args.inspect:
        cmd_inspect(client)
        return

    if args.delete:
        cmd_delete(client, args.delete)
        return

    if args.delete_commit:
        cmd_delete_commit(client, args.delete_commit)
        return

    # Commands that need the vector store
    _ensure_collection(client)
    store = _make_store(client)

    if args.search:
        cmd_search(client, store, args.search, args.subsystem, args.limit or 5)
        return

    if args.list:
        cmd_list(client, args.subsystem, args.type_, args.limit)
        return

    # Process / watch modes
    if args.files:
        run_files(
            [p.resolve() for p in args.files],
            db_path,
            client,
            store,
            verbose=args.verbose,
        )
    elif args.output_dir:
        output_dir = args.output_dir.resolve()
        if not output_dir.is_dir():
            logging.error("%s is not a directory", output_dir)
            sys.exit(1)
        run_watch(output_dir, db_path, client, store, args.poll, verbose=args.verbose)
    else:
        parser.error("Provide --output-dir (watch mode) or file paths to process")


if __name__ == "__main__":
    main()
