"""memory_mcp.py — MCP server exposing the V8 engineering memory DB.

Exposes two tools to Claude Code and other MCP clients:
  - search_v8_memory: semantic search with optional subsystem/type filters
  - list_memory_info: DB stats (entry count, available subsystems/types)

Configuration (environment variables):
  V8_MEMORY_DB_PATH   path to the Qdrant DB directory (required)

Usage:
    memory-mcp                          # stdio (default, for Claude Code)
    uv run memory-mcp

Claude Code ~/.claude/mcp.json:
    {
      "mcpServers": {
        "v8-memory": {
          "command": "uv",
          "args": ["--directory", "/path/to/ai-tools", "run", "memory-mcp"],
          "env": { "V8_MEMORY_DB_PATH": "/path/to/db" }
        }
      }
    }
"""

import os
import sys
from pathlib import Path

from fastmcp import FastMCP
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from memorize import COLLECTION, SUBSYSTEMS, TYPES, _FastEmbeddings

# ── setup ─────────────────────────────────────────────────────────────────────

_db_path = os.environ.get("V8_MEMORY_DB_PATH", "")
if not _db_path:
    print("V8_MEMORY_DB_PATH is not set", file=sys.stderr)
    sys.exit(1)

_client = QdrantClient(path=_db_path)
_store = QdrantVectorStore(
    client=_client, collection_name=COLLECTION, embedding=_FastEmbeddings()
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

    query: natural language description of what you're looking for
    subsystem: optional filter — one of: gc, scavenger, minorms, ignition,
               maglev, turbofan, turboshaft, liftoff, wasm, ic, builtins,
               csa, torque, parser, api, runtime, sandbox, profiler, debug
    type: optional filter — one of: performance, correctness, design, gotcha,
          api-change, refactor
    limit: number of results to return (default 5, max 20)
    """
    limit = min(limit, 20)
    must = []
    if subsystem:
        must.append(
            FieldCondition(key="metadata.subsystems", match=MatchValue(value=subsystem))
        )
    if type:
        must.append(FieldCondition(key="metadata.type", match=MatchValue(value=type)))
    qdrant_filter = Filter(must=must) if must else None

    try:
        results = _store.similarity_search_with_score(
            query, k=limit, filter=qdrant_filter
        )
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No relevant entries found."

    parts = []
    for doc, score in results:
        m = doc.metadata
        tags = ", ".join(m.get("subsystems", [])) or "—"
        parts.append(
            f"[relevance {score:.2f} | {m.get('date', '')} | {tags} | {m.get('type', '')}]\n"
            f"{doc.page_content}\n"
            f"(source: {m.get('source_file', '')})"
        )
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def list_memory_info() -> str:
    """Return statistics about the V8 memory DB: entry count, subsystems, types, date range."""
    if not _client.collection_exists(COLLECTION):
        return "Memory DB is empty (collection does not exist yet)."

    info = _client.get_collection(COLLECTION)
    total = info.points_count
    if not total:
        return "Memory DB is empty."

    subsystem_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    dates: list[str] = []
    offset = None

    while True:
        points, offset = _client.scroll(
            COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=200,
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
