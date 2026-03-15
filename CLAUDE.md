# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Install locally (all CLI commands into PATH via ~/.local/bin/)
uv tool install .

# Upgrade after pulling changes
uv tool upgrade ai-tools

# Run tests
pytest src/tests/

# Run a single test file
pytest src/tests/test_tools.py

# Run a single test
pytest src/tests/test_tools.py::test_name

# Lint / format
ruff check src/
ruff format src/
```

## Architecture

Three files, three responsibilities, almost no coupling between them (see `CONTEXT.md` for details):

- **`src/graph.py`** — LangGraph agent factory. `make_agent(model, checkpointer, ...)` configures tools, system prompt, and middleware. Also exports bare `agent` for LangGraph Studio. Handles Gemini schema fixups (missing `items` in array schemas).

- **`src/session.py`** — All logic. Manages agent routing, LangGraph streaming, interrupt handling, and manual conversation history. The `SessionIO` protocol (`write`, `set_status`) is its only interface to the UI. Zero Textual imports.

- **`src/cli.py`** — Pure Textual TUI. Implements `SessionIO`. Runs the session on a Textual worker thread; all UI interactions marshal back to the main thread via `call_from_thread`.

### Threading model

Session runs on a single worker thread. Three `threading.Event` pairs coordinate with the main thread:
- `_input_event`/`_input_value` — worker blocks waiting for user input
- `_steer_event`/`_steer_value` — mid-stream interruption by user
- `_stop` — shutdown; sets both others to unblock the worker immediately

### Agents

11 pre-configured agents in `session.py:AGENTS` (OpenAI GPT-5, Gemini Flash/Pro, Claude Haiku/Sonnet/Opus, DeepSeek). Each gets its own `MemorySaver`. Adding an agent is one dict entry with `model_id`, `description`, and any kwargs forwarded to `init_chat_model`.

Routing per turn: exact substring match (longest first), then a cheap Gemini Flash LLM call against descriptions. A fresh `thread_id` (uuid4) gives LangGraph a clean checkpoint namespace each turn.

Cross-turn history is a flat `[{"role", "content"}]` list prepended to each LangGraph input — this is how context is shared across agents and turns, since each agent's `MemorySaver` only tracks intra-turn state.

### Tools (`src/tools/`)

`standard_tools(web, git, fs, shell)` returns composable tool lists. Key tools:
- **`edit_file`** — 3-pass fuzzy matching (exact → strip trailing whitespace → sliding-window SequenceMatcher ≥0.85)
- **`run_d8`** — V8 JavaScript engine shell (parameter named `d8_args` to avoid Pydantic v1 collision)
- **`ask_user`** — LangGraph interrupt for mid-turn clarification

### Gemini schema compatibility

Gemini rejects tool schemas with missing `items` on array types. `graph.py` has schema-fixup logic for this. Tests in `src/tests/test_tool_schemas.py` validate all tool schemas pass Gemini's requirements.

## Required environment variables

```sh
export GOOGLE_API_KEY=...    # Gemini (required)
export TAVILY_API_KEY=...    # Web search (required)
export OPENAI_API_KEY=...    # Optional
export DEEPSEEK_API_KEY=...  # Optional
export ANTHROPIC_API_KEY=... # Optional
```

MCP servers can be configured in `~/.config/ai-tools/mcp.json`.
