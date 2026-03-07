# V8 Memory — System Prompt Guide

How to configure Claude Code, Gemini CLI, or any MCP-capable assistant to use
the `v8-memory` knowledge base effectively.

---

## How the tool gets invoked

The assistant sees two signals when deciding whether to call `search_v8_memory`:

1. **The tool docstring** — describes what the tool does and when to use it.
   Already tuned in `memory_mcp.py`.
2. **Your system prompt / CLAUDE.md** — the most reliable lever. Without it,
   LLMs reach for the memory tool reactively (only when stuck) rather than
   proactively (before starting work). The instructions below fix that.

---

## Claude Code — `~/.claude/CLAUDE.md`

Add this block to your global `~/.claude/CLAUDE.md` (or the repo-level
`CLAUDE.md` in the V8 checkout):

```markdown
## V8 knowledge base

A curated vector memory of V8 engineering insights is available via the
`search_v8_memory` MCP tool. It contains distilled knowledge about GC,
compilers (Ignition, Maglev, Turbofan), Wasm tiers (Liftoff, Turboshaft),
the IC system, builtins, CSA/Torque, and more — extracted from real V8
commits and reviewed for lasting relevance.

### When to query

Query `search_v8_memory` at the **start** of any V8 task — before reading
files, before writing code, before proposing a design. Do not wait until
you are stuck. The goal is to surface relevant context you might not know
to look for.

Query whenever you are:
- About to modify code in a V8 subsystem
- Trying to understand why something is implemented a certain way
- Checking for known gotchas or correctness constraints
- Making a design or performance decision in a V8 context

### How to query well

Run **2–3 targeted queries** from different angles. One broad query is
rarely enough.

Prefer specific, problem-shaped questions over keywords:
- GOOD: `search_v8_memory("write barrier elision conditions", subsystem="gc")`
- GOOD: `search_v8_memory("OSR entry requirements in Maglev", subsystem="maglev")`
- GOOD: `search_v8_memory("feedback vector slot kinds for call ICs", subsystem="ic")`
- BAD:  `search_v8_memory("GC")`
- BAD:  `search_v8_memory("compiler optimizations")`

Always pass `subsystem=` when the context makes it clear — it significantly
reduces noise. Available subsystems: gc, scavenger, minorms, ignition,
maglev, turbofan, turboshaft, liftoff, wasm, ic, builtins, csa, torque,
parser, api, runtime, sandbox, profiler, debug.

Use `type=` to narrow by category when relevant:
performance, correctness, design, gotcha, api-change, refactor.

### How to use results

After retrieving results, explicitly state:
- Which entries are relevant to the current task
- How they inform your approach or change what you would otherwise do
- If nothing relevant was found, say so — do not silently ignore the step

Do not treat memory results as authoritative truth. V8 evolves fast; an
insight may be stale. Cross-check against current source when it matters.
```

---

## Gemini CLI — system prompt

Gemini CLI reads a system prompt from `~/.gemini/system_prompt.md` (or
`--system-prompt` flag). Add:

```markdown
## V8 knowledge base

A `search_v8_memory` tool is available via MCP. It contains curated V8
engineering insights covering GC, Ignition, Maglev, Turbofan, Liftoff,
Turboshaft, Wasm, the IC system, builtins, CSA/Torque, and more.

**Always query it at the start of a V8 task** — before reading files or
writing code. Run 2–3 focused queries, not one broad one.

Query style:
- Specific and technical: "MinorMS large object handling during scavenge"
- Subsystem-scoped: pass subsystem="gc" / "maglev" / "ic" etc. whenever clear
- Problem-shaped: "why does Turbofan deopt on polymorphic receivers"
- Not: "GC", "V8 internals", "how compilers work"

After querying, summarise what you found and explain how it affects your
approach. If results are potentially stale, note that and verify in source.
```

---

## Explicit invocation (most reliable)

For tasks where you know the memory is likely relevant, ask for it directly
in your first message. This works regardless of system prompt configuration:

> "Before answering, search v8-memory for anything relevant to
> `[subsystem / topic]`, then proceed."

Or more specifically:

> "Search v8-memory for gotchas around write barriers in Maglev, then
> help me add a new IR instruction that allocates on the heap."

Explicit invocation is the most reliable pattern and worth using whenever
the task is clearly domain-specific.

---

## MCP server setup

Add to `~/.claude/mcp.json` (Claude Code) or equivalent:

```json
{
  "mcpServers": {
    "v8-memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/ai-tools", "run", "memory-mcp"],
      "env": {
        "V8_MEMORY_DB_PATH": "/path/to/v8-memory-db"
      }
    }
  }
}
```

`V8_MEMORY_DB_PATH` must point to the same directory used by `memorize --db`.
