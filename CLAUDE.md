# CLAUDE.md

Guidance for Claude Code working in this repo.

## Commands

```sh
# Install all CLI commands into ~/.local/bin/
uv tool install .

# After pulling changes
uv tool upgrade ai-tools

# Tests / lint
uv run pytest src/tests/
uv run pytest src/tests/test_tools.py::TestFind::test_exact_match
uv run ruff check src/
uv run ruff format src/
```

Always use `uv` to run Python (per user preference).

## CLI entry points (from `pyproject.toml`)

| Command       | Module             | Purpose                                              |
| ------------- | ------------------ | ---------------------------------------------------- |
| `agent`       | `cli.py`           | Multi-agent interactive REPL (the main tool)         |
| `qq`          | `qq.py`            | One-shot quick query; piped stdin handled inline     |
| `analyze`     | `analyze.py`       | Headless deep-agent perf-trace analysis              |
| `commitmsg`   | `commitmsg.py`     | Generate a commit message for staged/all changes     |
| `repowatcher` | `repowatcher.py`   | Filter+analyse new commits on a branch (daemon mode) |
| `memorize`    | `memorize.py`      | Curate analyses into a Chroma vector DB              |
| `memory-mcp`  | `memory_mcp.py`    | Expose the memory DB to MCP clients                  |

## Architecture (the `agent` REPL)

Three top-level files, near-zero coupling:

- **`src/graph.py`** — agent factory. `make_agent(model, checkpointer, name, agents, interrupt_on, extra_tools, system_prompt, extra_system_prompt, extra_middleware)` returns a `create_agent(...)` graph wired up with the standard tool set, `SubAgentMiddleware` (general-purpose subagent only), `SummarizationMiddleware`, `AnthropicPromptCachingMiddleware`, `PatchToolCallsMiddleware`, `ModelRetryMiddleware` (rate-limit backoff), `TodoListMiddleware`, optional `HumanInTheLoopMiddleware`. Schema fixups (strip `additionalProperties`/`$schema`/`title`/`description`, ensure `items` on arrays) keep Gemini happy. `--trace` enables stderr dumps of system prompt + each model call. Bare `agent = make_agent()` is exported for `langgraph dev`.

- **`src/session.py`** — all session logic. `Session` owns the agent registry (`AGENTS` class dict), routing, manual cross-turn history, LangGraph streaming, slash commands (`/help`, `/clear`, `/save`, `/load`, `/sessions`), HITL approval flow, MCP per-turn session setup, and persistence. `_LoggingIO` mirrors writes to `~/.local/state/ai-tools/log/<ts>.log`. Sessions auto-save to `~/.local/share/ai-tools/sessions/auto_<ts>.{json,db}` on shutdown (10 most recent kept).

- **`src/cli.py`** — terminal UI built on `prompt_toolkit` + Rich. `TerminalIO` implements `SessionIO`, with a `_TeeFile` that captures Rich output so tool results can be re-rendered when the user toggles collapse. Bottom toolbar shows live status. Keys: Enter submits, Alt+Enter / Ctrl-J newline, Esc interrupts current turn, Ctrl-O toggles tool-output collapse, Up/Down history navigation.

(`CONTEXT.md` is older and refers to a pre-prompt-toolkit Textual UI — treat CLAUDE.md as authoritative.)

### Threading and event coordination

The session runs on a worker thread spawned by `cli.main()`. Four `threading.Event`s coordinate with the prompt-toolkit thread:

- `_input_event` / `_input_value` — worker blocks here when it needs user text.
- `_steer_event` / `_steer_value` — set by `submit()` mid-turn to redirect the agent.
- `_interrupt_event` — set by Esc; raises `_Interrupted` between chunks.
- `_stop` — shutdown signal; sets the others to unblock the worker.

Plus a persistent `asyncio` event loop in a daemon thread (`Session._loop`), used only on the MCP path so async HTTP clients keep their connection pools alive across turns.

### Streaming pipeline (`_stream`)

Producer thread iterates `agent.stream(...)` (or, for MCP, `astream` on the persistent loop) and pushes chunks through a `queue.Queue`. The consumer polls every 50 ms so `_stop`/`_steer`/`_interrupt` events surface promptly. Token usage is summed from `AIMessageChunk.usage_metadata` and reported at end of turn.

### Agents

11 entries in `Session.AGENTS`. Names are routing keys; each provides `model_id` + `description` + extra kwargs forwarded verbatim to `init_chat_model`. Current roster:

- OpenAI: `gpt` (5.2), `turbo` (5.2-pro), `mini` (5-mini), `nano` (5-nano)
- Google: `lite` / `flash` (gemini-3-flash, minimal/medium thinking), `gemini` (3.1-pro)
- Anthropic: `haiku`, `sonnet`, `opus`
- DeepSeek: `seek` (chat). `deepseek-reasoner` is excluded — no tool-calling.

Default agent is `flash`; routing model is `lite`. An entry is omitted at startup if its provider env var is unset (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`).

### Routing

Per turn: 1) substring match on `<name>:` prefix (longest first to avoid `deepseek` matching `deepseek-v3`); 2) ask the lite router whether a name was mentioned, expecting one of the names or `?`; 3) fall back to `_last_agent` or `DEFAULT_AGENT`. Routing is name-based only — never by topic.

### Cross-turn history

A flat `list[{"role", "content"}]` is maintained on the session. Each agent has a per-agent offset (`_agent_history_offset`) so it only sees the slice it hasn't seen yet — its own prior turns are already in its LangGraph thread (one stable `thread_id` per agent name). Assistant content is tagged `[agent-name]: ...` so cross-agent context preserves attribution.

### Persistence

A shared **in-memory SQLite via URI mode** backs `SqliteSaver` (sync) and `AsyncSqliteSaver` (used on the MCP path). `/save` snapshots both the JSON metadata and the SQLite DB to `~/.local/share/ai-tools/sessions/<name>.{json,db}`. `/load` restores both.

### MCP

Config lives at `~/.config/ai-tools/mcp.json` (passed verbatim to `MultiServerMCPClient`). Per-turn pattern: open all sessions, load tool sets, inject server `instructions` strings into the system prompt, then build a fresh agent for that turn with all MCP tools registered in the `ToolNode`.

**Lazy tool loading** (`tools/lazy_mcp.py`) keeps schemas out of the model context until the LLM calls `search_mcp_tools(query)`. The companion `LazyMCPMiddleware` filters `request.tools` and re-seeds the unlocked set from prior `search_mcp_tools` results in history each turn.

### Tools (`src/tools/`)

`standard_tools(web, git, fs, shell)` returns:
- always: `web_fetch` (and `web_search` if `TAVILY_API_KEY` is set)
- if `git=True` and cwd is in a git repo: `git_grep/show/blame/log/diff/status`, `read_around`
- if `fs=True`: `read_file`, `grep_files`, `list_dir`, `edit_file`, `write_file`
- if `shell=True`: `run_shell`

Notable:
- **`edit_file`** — 3-pass fuzzy match: exact → strip-trailing-whitespace → sliding-window `SequenceMatcher` ≥ 0.85.
- **`run_d8`** (`tools/shell.py`) — used only by `analyze.py` as `extra_tools`, never via `standard_tools`. The `d8_args` parameter name avoids the Pydantic v1 `ValidatedFunction` `v__args` ghost-field bug (see `tests/test_tool_schemas.py`).
- **`ask_user`** — LangGraph `interrupt()` for clarification; only injected when `interrupt_on` is set (i.e. interactive).
- HITL approval is enabled by default for `edit_file`/`write_file` (`Session.INTERRUPT_ON`); approval prompts show a unified diff first.

## Required environment variables

```sh
export GOOGLE_API_KEY=...    # always (router uses lite)
export TAVILY_API_KEY=...    # for web_search
export OPENAI_API_KEY=...    # optional
export DEEPSEEK_API_KEY=...  # optional
export ANTHROPIC_API_KEY=... # optional
```

`~/.config/ai-tools/mcp.json` is optional MCP server config.
