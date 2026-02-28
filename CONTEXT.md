# Architecture

Three files, three responsibilities, almost no coupling between them.

**`graph.py`** defines the LangGraph agent: tools, system prompt, and a `make_agent(model, checkpointer)` factory. It knows nothing about sessions, routing, or UI. It also exports a bare `agent = make_agent()` for LangGraph Studio / `langgraph dev`, which injects its own checkpointer at runtime.

**`session.py`** owns all the logic. It talks to the UI exclusively through the two-method `SessionIO` protocol (`write`, `set_status`). The UI calls back via two methods: `submit(text)` and `stop()`. That's the entire interface — `session.py` has zero Textual imports.

**`cli.py`** is pure Textual. `AgentApp` implements `SessionIO` by wrapping `_append` and `_set_placeholder` in `call_from_thread`, which safely marshals calls from the worker onto the event loop. All input flows through `on_input_submitted` → `session.submit()`. Shutdown flows through `on_unmount` → `session.stop()`.

---

## Threading model

The session runs on a single Textual worker thread (`run_worker(..., thread=True)`). Textual's event loop stays on the main thread. They share no mutable state except through three `threading.Event` pairs:

- **`_input_event` / `_input_value`** — for when the session is actively waiting for user input (interrupt responses, post-completion prompts). The worker blocks on `_input_event.wait()`; the main thread sets it via `submit()`.
- **`_steer_event` / `_steer_value`** — for mid-stream interruptions. The worker checks `_steer_event.is_set()` between each LangGraph chunk; the main thread sets it via `submit()` when the session isn't already waiting.
- **`_stop`** — the shutdown signal. `stop()` sets it and also sets both other events to immediately unblock whatever the worker is stuck on.

`submit()` routes to one of the two first pairs based on `_waiting_for_input`. The boolean flag itself is only written by the worker thread, only read by the UI thread, so there's no race worth worrying about here — if the flag is momentarily stale the worst outcome is a submit going to steer instead of input, which is harmless.

---

## Agent setup

All agents are instantiated eagerly at `Session.__init__` time via `_build_agents()`. Each gets its own `MemorySaver` checkpointer. The `AGENTS` class dict is the single source of truth: `model_id` and `description` are metadata keys stripped before the rest is forwarded verbatim to `init_chat_model`. Adding an agent is one dict entry.

Each agent is a LangGraph compiled graph produced by `create_deep_agent` (from the `deepagents` library), which wraps a ReAct-style tool-calling loop around the given model with the three tools (`git_show`, `read_around`, `ask_user`) and a `FilesystemBackend` for virtual file access into the V8 repo.

Note: `deepseek-reasoner` (R1) is excluded — it doesn't support tool calling, which this agent loop requires.

---

## The main loop and how agents are called

`run()` is the top-level loop. One user message per iteration:

```
route → stream (possibly resume several times) → append to history → wait for next message
```

**Routing** happens in `_setup_turn`. First an exact substring match against agent names is tried (longest-first to avoid `deepseek` matching before `deepseek-v3`). If no explicit name is found, a cheap LLM call (gemini-flash) picks one from the descriptions. A fresh `thread_id` is generated here via `uuid4()`, giving LangGraph a clean checkpoint namespace for this turn.

**Streaming** is `_stream`, which drives one `agent.stream()` call in `stream_mode=["messages", "updates"]`: `messages` yields individual LangGraph message chunks as they arrive from the LLM; `updates` yields graph state deltas, which is where interrupts surface. On each chunk it checks `_stop` and `_steer_event` before doing anything else.

**The resume loop** in `_run_turn` is what makes multi-step tool calling work. `_stream` returns one of three outcomes: `(steered, None, parts)`, `(False, Command(...), parts)`, or `(False, None, parts)`. The middle case means LangGraph hit an interrupt — the `ask_user` tool or a HITL gate — and the session pauses to collect user input, then feeds it back as a `Command(resume=...)`. `_run_turn` loops, calling `_stream` again with the Command as input, until LangGraph exhausts the graph (no more interrupt, no steer).

**History** is maintained manually as a flat list of `{"role", "content"}` dicts. At the end of each completed turn the user message and the full agent response (tagged `[agent-name]: ...`) are appended. The next turn prepends this entire list to the LangGraph input messages, giving every agent full conversation context regardless of which model handled previous turns. LangGraph's per-agent `MemorySaver` only tracks intra-turn state (tool call chains, intermediate messages). Cross-turn, cross-agent continuity is the session's job.

**Steering** short-circuits the current turn entirely. When `_steer_event` fires mid-stream, `_stream` flushes its buffer and returns `steered=True`. `_run_turn` surfaces this to `run()`, which discards any partial response and re-enters the loop with the steer message as the new user message — bypassing history for that aborted turn.

---

## Shutdown

`_Stopped` is a private exception used purely for unwinding the call stack when `stop()` is called. It's raised in `_wait_input` (after `_input_event` is triggered) and in `_stream` (after `_stop` is detected on the next chunk boundary). It propagates naturally through `_handle_interrupt` → `_run_turn` → `run()`, where it's caught and silently swallowed, allowing the worker thread to exit cleanly.
