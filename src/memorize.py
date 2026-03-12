"""memorize.py — curate V8 commit analyses into a long-term vector memory.

Watches a repowatcher output directory for new .md analysis files, runs a
curation agent to decide what (if anything) is worth storing as a lasting
engineering insight, and writes distilled entries to a local Chroma DB.

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

    # One-off migration from an old Qdrant DB
    memorize --db PATH --migrate /path/to/old/qdrant/db
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

import chromadb
from fastembed import TextEmbedding as _FE
from platformdirs import user_data_dir
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ── constants ─────────────────────────────────────────────────────────────────

CURATION_MODEL = "google_genai:gemini-3-flash-preview"
DEDUPE_THRESHOLD = 0.82  # cosine similarity above which we check for overlap
DEDUPE_K = 3  # how many similar entries to surface for the dedup decision
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # local fastembed, 768 dims
COLLECTION = "v8_memory"
DEFAULT_DB = Path(user_data_dir("ai-tools")) / "v8-memory"
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
        description="Chroma document IDs of existing entries to remove (replace action only)",
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


# ── chroma setup ──────────────────────────────────────────────────────────────


def _make_client(db_path: Path) -> chromadb.PersistentClient:
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def _ensure_collection(client: chromadb.ClientAPI) -> None:
    client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def _make_store(client: chromadb.ClientAPI) -> Chroma:
    return Chroma(
        client=client, collection_name=COLLECTION, embedding_function=_FastEmbeddings()
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
    path: Path, client: chromadb.ClientAPI, store: Chroma, verbose: bool = False
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
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    n = min(DEDUPE_K, collection.count())
    if n > 0:
        emb = _FastEmbeddings().embed_query(result.text)
        res = collection.query(
            query_embeddings=[emb],
            n_results=n,
            include=["documents", "distances"],
        )
        # Chroma cosine distance = 1 - similarity
        hits = [
            (id_, doc, 1.0 - dist)
            for id_, doc, dist in zip(
                res["ids"][0], res["documents"][0], res["distances"][0]
            )
            if 1.0 - dist >= DEDUPE_THRESHOLD
        ]
        if hits:
            try:
                decision, dedup_usage = _dedupe(result.text, hits)
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
                    collection.delete(ids=decision.replace_ids)
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

    doc = Document(
        page_content=result.text,
        metadata={
            "commit": meta.get("commit", ""),
            "date": meta.get("date", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "subsystems": ",".join(result.subsystems),
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
    client: chromadb.ClientAPI,
    store: Chroma,
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
    client: chromadb.ClientAPI,
    store: Chroma,
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


def cmd_inspect(client: chromadb.ClientAPI) -> None:
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        print("Collection does not exist yet.")
        return

    total = collection.count()
    print(f"Entries: {total}")
    if not total:
        return

    results = collection.get(include=["metadatas"])
    subsystem_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    dates: list[str] = []

    for m in results["metadatas"]:
        for s in (m.get("subsystems") or "").split(","):
            s = s.strip()
            if s:
                subsystem_counts[s] = subsystem_counts.get(s, 0) + 1
        t = m.get("type", "")
        if t:
            type_counts[t] = type_counts.get(t, 0) + 1
        if m.get("date"):
            dates.append(m["date"])

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
    store: Chroma,
    query: str,
    subsystem: str | None,
) -> None:
    chroma_filter = _build_filter(subsystem, None)
    results = store.similarity_search_with_score(query, k=1000, filter=chroma_filter)
    for doc, score in results:
        m = doc.metadata
        subs = (m.get("subsystems") or "").split(",")
        print(
            f"\n[{score:.3f}] {m.get('date', '')}  {', '.join(s for s in subs if s)}  {m.get('type', '')}"
        )
        print(f"  source: {m.get('source_file', '')}")
        print(f"  {doc.page_content[:300].replace(chr(10), ' ')}")


def cmd_list(
    client: chromadb.ClientAPI, subsystem: str | None, type_: str | None
) -> None:
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        print("Collection does not exist yet.")
        return

    kwargs: dict = {"include": ["documents", "metadatas"]}
    chroma_filter = _build_filter(subsystem, type_)
    if chroma_filter:
        kwargs["where"] = chroma_filter

    results = collection.get(**kwargs)
    for id_, doc, m in zip(results["ids"], results["documents"], results["metadatas"]):
        subs = (m.get("subsystems") or "").split(",")
        snippet = (doc or "")[:80].replace("\n", " ")
        print(
            f"{id_}  {m.get('date', '')}  [{', '.join(s for s in subs if s)}]  {m.get('type', '')}  {snippet}"
        )


def cmd_delete(client: chromadb.ClientAPI, point_id: str) -> None:
    collection = client.get_collection(COLLECTION)
    collection.delete(ids=[point_id])
    print(f"Deleted {point_id}")


def cmd_delete_commit(client: chromadb.ClientAPI, commit_hash: str) -> None:
    collection = client.get_collection(COLLECTION)
    collection.delete(where={"commit": {"$eq": commit_hash}})
    print(f"Deleted all entries for commit {commit_hash}")


def _build_filter(subsystem: str | None, type_: str | None) -> dict | None:
    parts = []
    if subsystem:
        parts.append({"subsystems": {"$contains": subsystem}})
    if type_:
        parts.append({"type": {"$eq": type_}})
    if not parts:
        return None
    return {"$and": parts} if len(parts) > 1 else parts[0]


# ── migration from Qdrant ─────────────────────────────────────────────────────


def cmd_migrate(client: chromadb.ClientAPI, store: Chroma, qdrant_path: Path) -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("qdrant-client is not installed — cannot migrate.", file=sys.stderr)
        sys.exit(1)

    src = QdrantClient(path=str(qdrant_path))
    if not src.collection_exists(COLLECTION):
        print(f"No '{COLLECTION}' collection found in {qdrant_path}")
        return

    total = src.get_collection(COLLECTION).points_count or 0
    print(f"Migrating {total} entries from {qdrant_path} …")

    migrated = 0
    offset = None
    while True:
        points, offset = src.scroll(
            COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=100,
            offset=offset,
        )
        docs = []
        for p in points:
            payload = p.payload or {}
            m = payload.get("metadata", {})
            # subsystems was stored as a list in Qdrant — join to string for Chroma
            subs = m.get("subsystems", [])
            m["subsystems"] = ",".join(subs) if isinstance(subs, list) else (subs or "")
            docs.append(
                Document(page_content=payload.get("page_content", ""), metadata=m)
            )
        if docs:
            store.add_documents(docs)
            migrated += len(docs)
            print(f"  {migrated}/{total}", end="\r", flush=True)
        if offset is None:
            break

    print(f"\nDone — migrated {migrated} entries.")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate V8 commit analyses into a vector memory DB."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to Chroma DB directory (default: {DEFAULT_DB})",
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

    # Maintenance flags
    parser.add_argument("--inspect", action="store_true", help="Show DB stats and exit")
    parser.add_argument("--search", metavar="QUERY", help="Semantic search and exit")
    parser.add_argument("--list", action="store_true", help="List entries and exit")
    parser.add_argument("--delete", metavar="ID", help="Delete entry by document ID")
    parser.add_argument(
        "--delete-commit", metavar="HASH", help="Delete all entries for a commit"
    )
    parser.add_argument(
        "--migrate",
        metavar="QDRANT_DB_PATH",
        type=Path,
        help="One-off migration: copy all entries from an old Qdrant DB into --db",
    )

    # Filters used by --search and --list
    parser.add_argument("--subsystem", help="Filter by subsystem")
    parser.add_argument("--type", dest="type_", help="Filter by type (for --list)")

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

    # Maintenance commands that don't need the vector store
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

    if args.migrate:
        cmd_migrate(client, store, args.migrate.resolve())
        return

    if args.search:
        cmd_search(store, args.search, args.subsystem)
        return

    if args.list:
        cmd_list(client, args.subsystem, args.type_)
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
