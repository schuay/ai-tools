"""memory_mcp.py — MCP server exposing the V8 engineering memory DB.

Exposes two tools to Claude Code and other MCP clients:
  - search_v8_memory: semantic search with optional subsystem/type filters
  - list_memory_info: DB stats (entry count, available subsystems/types)

Configuration (environment variables):
  V8_MEMORY_DB_PATH   path to the Chroma DB directory (default: platformdirs user_data_dir/ai-tools/v8-memory)

Usage:
    memory-mcp                          # stdio (default, for Claude Code)
    uv run memory-mcp

Claude Code ~/.claude/mcp.json:
    {
      "mcpServers": {
        "v8-memory": {
          "command": "uv",
          "args": ["--directory", "/path/to/ai-tools", "run", "memory-mcp"],
          "env": { "V8_MEMORY_DB_PATH": "/path/to/chroma/db" }
        }
      }
    }
"""

import os

import chromadb
from fastmcp import FastMCP
from langchain_chroma import Chroma

from memorize import (
    COLLECTION,
    DEFAULT_DB,
    SUBSYSTEMS,
    TYPES,
    _FastEmbeddings,
    _build_filter,
)

# ── setup ─────────────────────────────────────────────────────────────────────

_db_path = os.environ.get("V8_MEMORY_DB_PATH") or str(DEFAULT_DB)

_client = chromadb.PersistentClient(path=_db_path)
_store = Chroma(
    client=_client, collection_name=COLLECTION, embedding_function=_FastEmbeddings()
)

mcp = FastMCP(
    "v8-memory",
    instructions=(
        "Search the V8 engineering knowledge base for insights about V8 internals. "
        "Use search_v8_memory when working on V8-related code to retrieve relevant "
        "context about subsystems, known gotchas, design decisions, and performance "
        f"considerations. Available subsystems: {', '.join(SUBSYSTEMS)}. "
        f"Available types: {', '.join(TYPES)}."
    ),
)

# ── tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def search_v8_memory(
    query: str,
    subsystem: str | None = None,
    type: str | None = None,
    limit: int = 5,
) -> str:
    """Search the V8 engineering knowledge base for relevant insights.

    Call this BEFORE writing or modifying V8 code, not only when stuck.
    Run 2-3 targeted queries from different angles rather than one broad one.
    Always pass subsystem= when the code context makes it clear — it cuts noise
    significantly. Prefer specific, problem-shaped queries over generic keywords.

    Good queries:
      search_v8_memory("osr entry requirements in Maglev", subsystem="maglev")
      search_v8_memory("write barrier elision conditions", subsystem="gc")
      search_v8_memory("feedback vector slot kinds for call ICs", subsystem="ic")
      search_v8_memory("Turbofan deopt on polymorphic call sites", subsystem="turbofan")
      search_v8_memory("MinorMS handling of large objects during scavenge", subsystem="scavenger")

    Bad queries: "GC", "compiler", "how does V8 work", "optimization"

    query: specific technical question or description of what you're looking for
    subsystem: optional filter — one of: gc, scavenger, minorms, ignition,
               maglev, turbofan, turboshaft, liftoff, wasm, ic, builtins,
               csa, torque, parser, api, runtime, sandbox, profiler, debug
    type: optional filter — one of: performance, correctness, design, gotcha,
          api-change, refactor
    limit: number of results to return (default 5, max 20)
    """
    limit = min(limit, 20)
    chroma_filter = _build_filter(subsystem, type)

    try:
        results = _store.similarity_search_with_score(
            query, k=limit, filter=chroma_filter
        )
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No relevant entries found."

    parts = []
    for doc, score in results:
        m = doc.metadata
        subs = (m.get("subsystems") or "").split(",")
        tags = ", ".join(s for s in subs if s) or "—"
        parts.append(
            f"[relevance {score:.2f} | {m.get('date', '')} | {tags} | {m.get('type', '')}]\n"
            f"{doc.page_content}\n"
            f"(source: {m.get('source_file', '')})"
        )
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def list_memory_info() -> str:
    """Return statistics about the V8 memory DB: entry count, subsystems, types, date range."""
    try:
        collection = _client.get_collection(COLLECTION)
    except Exception:
        return "Memory DB is empty (collection does not exist yet)."

    total = collection.count()
    if not total:
        return "Memory DB is empty."

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

    lines = [f"Total entries: {total}"]
    if dates:
        lines.append(f"Date range: {min(dates)} → {max(dates)}")
    if subsystem_counts:
        lines.append(
            "\nSubsystems: "
            + ", ".join(
                f"{s}({n})"
                for s, n in sorted(subsystem_counts.items(), key=lambda x: -x[1])
            )
        )
    if type_counts:
        lines.append(
            "Types: "
            + ", ".join(
                f"{t}({n})" for t, n in sorted(type_counts.items(), key=lambda x: -x[1])
            )
        )
    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
