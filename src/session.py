"""
Multi-agent conversation session.

All agent logic lives here: registry, routing, conversation history,
LangGraph streaming, and interrupt handling.

The UI supplies a SessionIO implementation and calls:
    session.submit(text)   — user pressed Enter
    session.run()          — blocks; run on a worker thread
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import time
import traceback
from datetime import datetime
from pathlib import Path

from platformdirs import user_log_path
from threading import Event, Thread
from typing import Protocol


def _provider_key_var(model_id: str) -> str | None:
    """Return the env-var name required by this model_id's provider, or None."""
    if model_id.startswith("openai:"):
        return "OPENAI_API_KEY"
    if model_id.startswith("anthropic:"):
        return "ANTHROPIC_API_KEY"
    if model_id.startswith("google_genai:"):
        return "GOOGLE_API_KEY"
    if model_id.startswith("deepseek:") or model_id.startswith("deepseek-"):
        return "DEEPSEEK_API_KEY"
    return None


class _Stopped(Exception):
    """Raised internally to unwind the call stack when stop() is called."""


class _Interrupted(Exception):
    """Raised when the user presses Esc to abort the current agent turn."""


from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import make_agent
from tools.fs import preview_diff, preview_write


# ── IO protocol ──────────────────────────────────────────────────────────────


class SessionIO(Protocol):
    """What the Session needs from the UI. Called from the worker thread."""

    def write(self, text: str, style: str | None = None) -> None: ...
    def set_status(self, text: str) -> None: ...


_LOG_DIR = user_log_path("ai-tools")


class _LoggingIO:
    """Wraps a SessionIO and mirrors write() calls to a timestamped log file."""

    def __init__(self, inner: SessionIO) -> None:
        self._inner = inner
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._f = (_LOG_DIR / f"{ts}.log").open("w", buffering=1, encoding="utf-8")

    def write(self, text: str, style: str | None = None) -> None:
        self._inner.write(text, style)
        if not self._f.closed:
            self._f.write(text + "\n")

    def set_status(self, text: str) -> None:
        self._inner.set_status(text)

    def log(self, text: str) -> None:
        """Write a line directly to the log (e.g. user input) without sending to UI."""
        if not self._f.closed:
            self._f.write(text + "\n")

    def close(self) -> None:
        self._f.close()


# ── streaming helper ─────────────────────────────────────────────────────────


class _LineBuffer:
    """
    Accumulates streaming text and flushes to SessionIO one line at a time.
    Holds the current incomplete line until a newline arrives or flush() is called.
    """

    def __init__(self, io: SessionIO) -> None:
        self._io = io
        self._buf = ""
        self._style: str | None = None

    def push(self, text: str, style: str | None = None) -> None:
        self._style = style
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._io.write(line, style=style)

    def flush(self) -> None:
        if self._buf:
            self._io.write(self._buf, style=self._style)
            self._buf = ""


# ── session ──────────────────────────────────────────────────────────────────


class Session:
    """
    Manages a multi-agent conversation: routing, history, streaming, interrupts.

    Override class-level attributes to customise agents and routing without
    touching any other code:

        class MySession(Session):
            AGENTS = {
                "my-agent": {
                    "model_id": "openai:gpt-4o",
                    "description": "GPT-4o for everything",
                },
            }
            DEFAULT_AGENT = "my-agent"
    """

    # ── agent registry ───────────────────────────────────────────────────────
    # Each entry: model_id (required), description (required for router prompt),
    # plus any extra kwargs forwarded to init_chat_model.

    AGENTS: dict[str, dict] = {
        "gpt": {
            "model_id": "openai:gpt-5.2",
            "reasoning": {"effort": "medium", "summary": "auto"},
            "description": "OpenAI GPT-5.2",
        },
        "turbo": {
            "model_id": "openai:gpt-5.2-pro",
            "reasoning": {"effort": "high", "summary": "auto"},
            "description": "OpenAI GPT-5.2 Pro — higher reasoning effort",
        },
        "mini": {
            "model_id": "openai:gpt-5-mini",
            "reasoning": {"effort": "medium", "summary": "auto"},
            "description": "OpenAI GPT-5 mini — faster, more cost-efficient",
        },
        "nano": {
            "model_id": "openai:gpt-5-nano",
            "reasoning": {"effort": "low", "summary": "auto"},
            "description": "OpenAI GPT-5 nano",
        },
        "seek": {
            "model_id": "deepseek-chat",
            "description": "DeepSeek chat",
        },
        "lite": {
            "model_id": "google_genai:gemini-3-flash-preview",
            "include_thoughts": True,
            "thinking_level": "minimal",
            "description": "Gemini Flash — minimal thinking",
        },
        "flash": {
            "model_id": "google_genai:gemini-3-flash-preview",
            "include_thoughts": True,
            "thinking_level": "medium",
            "description": "Gemini Flash — medium thinking",
        },
        "gemini": {
            "model_id": "google_genai:gemini-3.1-pro-preview",
            "include_thoughts": True,
            "thinking_level": "high",
            "max_retries": 6,
            "description": "Gemini Pro",
        },
        "haiku": {
            "model_id": "anthropic:claude-haiku-4-5-20251001",
            "description": "Claude Haiku — fastest",
        },
        "sonnet": {
            "model_id": "anthropic:claude-sonnet-4-6",
            "description": "Claude Sonnet",
        },
        "opus": {
            "model_id": "anthropic:claude-opus-4-6",
            "description": "Claude Opus",
        },
    }
    DEFAULT_AGENT = "flash"
    ROUTER_AGENT_NAME = "nano"

    # Keys in AGENTS entries that are not forwarded to init_chat_model.
    _METADATA_KEYS = frozenset({"model_id", "description"})

    def __init__(self, io: SessionIO, prompt: str) -> None:
        self._io = _LoggingIO(io)
        self._prompt = prompt

        self._agents = self._build_agents()
        router_cfg = self.AGENTS[self.ROUTER_AGENT_NAME]
        router_kwargs = {k: v for k, v in router_cfg.items() if k not in self._METADATA_KEYS}
        self._router = init_chat_model(router_cfg["model_id"], **router_kwargs)
        self._history: list[dict] = []
        self._last_agent: str | None = None
        self._agent_history_offset: dict[str, int] = {}

        # Thread synchronisation between the worker (session) and main (UI) threads.
        self._input_event = Event()
        self._input_value = ""
        self._waiting_for_input = False
        self._steer_event = Event()
        self._steer_value = ""
        self._stop = Event()
        self._interrupt_event = Event()

    # ── public API ───────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the worker thread to exit. Called from the UI thread."""
        self._stop.set()
        self._input_event.set()  # unblocks _wait_input
        self._steer_event.set()  # unblocks _stream on next chunk check
        self._io.close()

    def interrupt(self) -> None:
        """Abort the current agent turn. Called from the UI thread (Esc key)."""
        self._interrupt_event.set()

    def submit(self, text: str) -> None:
        """Called from the UI thread when the user presses Enter."""
        self._io.log(f"> {text}")
        if self._waiting_for_input:
            self._input_value = text
            self._input_event.set()
        else:
            self._steer_value = text
            self._steer_event.set()

    def run(self) -> None:
        """Main loop. Blocks; run on a dedicated worker thread."""
        for name, agent_cfg in self.AGENTS.items():
            if name not in self._agents:
                continue
            marker = " (default)" if name == self.DEFAULT_AGENT else ""
            self._io.write(f"  {name}: {agent_cfg['description']}{marker}", style="dim")
        if self._mcp_server_names:
            self._io.write(f"  mcp: {', '.join(self._mcp_server_names)}", style="dim")
        try:
            user_msg = self._prompt or self._wait_input("> ")
        except _Stopped:
            return
        force_agent: str | None = None
        try:
            while True:
                self._interrupt_event.clear()  # discard stale Esc presses from idle period
                try:
                    steered, response = self._run_turn(user_msg, force_agent)
                except _Stopped:
                    raise
                except _Interrupted:
                    self._history.append({"role": "user", "content": user_msg})
                    self._history.append(
                        {"role": "assistant", "content": "[interrupted]"}
                    )
                    self._agent_history_offset[self._last_agent] = len(self._history)
                    self._io.write("[interrupted]", style="bold yellow")
                    user_msg = self._wait_input("> ")
                    force_agent = None
                    continue
                except Exception as e:
                    self._history.append({"role": "user", "content": user_msg})
                    self._history.append(
                        {"role": "assistant", "content": f"[error: {type(e).__name__}]"}
                    )
                    self._agent_history_offset[self._last_agent] = len(self._history)
                    self._io.write(
                        f"[error] {type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                        style="bold red",
                    )
                    user_msg = self._wait_input("> ")
                    force_agent = None
                    continue
                if steered:
                    self._history.append({"role": "user", "content": user_msg})
                    self._history.append({"role": "assistant", "content": "[steered]"})
                    self._agent_history_offset[self._last_agent] = len(self._history)
                    user_msg = self._steer_value
                    force_agent = self._last_agent  # stay on the active agent
                    continue
                self._history.append({"role": "user", "content": user_msg})
                self._history.append({"role": "assistant", "content": response})
                self._agent_history_offset[self._last_agent] = len(self._history)
                user_msg = self._wait_input("> ")
                force_agent = None
        except _Stopped:
            pass

    # ── agent construction ───────────────────────────────────────────────────

    # Tools that require explicit user approval before execution.
    INTERRUPT_ON: dict[str, bool] = {
        "edit_file": True,
        "write_file": True,
    }

    def _build_agent(
        self,
        name: str,
        cfg: dict,
        all_agents: dict,
        extra_tools: list | None = None,
        checkpointer=None,
    ) -> object:
        kwargs = {k: v for k, v in cfg.items() if k not in self._METADATA_KEYS}
        return make_agent(
            model=init_chat_model(cfg["model_id"], **kwargs),
            checkpointer=checkpointer,
            name=name,
            agents=all_agents,
            interrupt_on=self.INTERRUPT_ON,
            extra_tools=extra_tools,
        )

    def _init_mcp(self) -> tuple:
        """Load MCP config and create a MultiServerMCPClient, or return (None, []).

        The client is stateless — it holds connection configs, not live connections.
        Actual sessions are opened per-turn in _run_turn() using client.session(),
        which is the pattern recommended by langchain-mcp-adapters for stateful servers:
          https://docs.langchain.com/oss/python/langchain/mcp#stateful-sessions

        For stdio transport: each session spawns a subprocess that lives for the
        duration of one agent turn (all tool calls in that turn share the subprocess).
        Between turns the subprocess is killed and restarted. Cross-turn state is NOT
        preserved; if you need that, a full async refactor is required (persistent loop).

        For HTTP/SSE transport: the server process is long-lived regardless; the
        per-turn session is just an HTTP connection that is re-established each turn.
        GDB state lives in the server process and survives across turns.
        """
        from tools.mcp import load_config

        config = load_config()
        if not config:
            return None, []
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            return MultiServerMCPClient(config), list(config.keys())
        except ImportError:
            self._io.write("[mcp] langchain-mcp-adapters not installed", style="bold red")
            return None, []
        except Exception as e:
            self._io.write(f"[mcp] failed to init client: {e}", style="bold red")
            return None, []

    def _build_agents(self) -> dict:
        available = {
            name: cfg
            for name, cfg in self.AGENTS.items()
            if not (key := _provider_key_var(cfg["model_id"])) or os.environ.get(key)
        }
        # Store available configs and checkpointers separately from the compiled agents.
        # Checkpointers must survive agent rebuilds: per-turn MCP sessions require us to
        # recompile the agent each turn (to inject session-bound tools), but the
        # MemorySaver instance must be shared so LangGraph thread history is preserved.
        # All deepagents middleware (TodoList, Summarization, etc.) stores its state
        # through LangGraph's checkpoint, not in middleware instance vars, so rebuilding
        # the compiled agent is safe.
        #
        # Optimisation opportunity: only rebuild when MCP tools are configured; for
        # non-MCP turns the pre-built agent from this dict could be used directly.
        self._available_agents = available
        self._checkpointers = {name: MemorySaver() for name in available}
        self._mcp_client, self._mcp_server_names = self._init_mcp()
        return {
            name: self._build_agent(
                name, cfg, all_agents=available, checkpointer=self._checkpointers[name]
            )
            for name, cfg in available.items()
        }

    # ── turn orchestration ───────────────────────────────────────────────────

    def _run_turn(
        self, user_msg: str, force_agent: str | None = None
    ) -> tuple[bool, str]:
        """
        Route user_msg, run the agent to completion.
        Returns (steered, response_for_history).
        If steered, response is empty and self._steer_value holds the new message.
        """
        agent_name, agent, config = self._setup_turn(user_msg, force_agent)

        # Inject only the cross-agent history this agent hasn't seen yet.
        # Its own prior turns are already in its LangGraph thread (tool calls included).
        offset = self._agent_history_offset.get(agent_name, 0)
        messages = [
            {"role": m["role"], "content": m["content"]} for m in self._history[offset:]
        ]
        messages.append({"role": "user", "content": user_msg})
        current_input: dict | Command = {"messages": messages}
        response_parts: list[str] = []

        while True:
            self._steer_event.clear()

            if self._mcp_client:
                # Per-turn stateful MCP sessions (docs pattern):
                #   async with client.session(name) as session:
                #       tools = await load_mcp_tools(session)
                #       agent = create_agent(..., tools)
                #
                # The agent is rebuilt each turn so it carries the session-bound tools.
                # The checkpointer is shared (self._checkpointers[agent_name]) so the
                # LangGraph thread history (messages, tool calls) survives across turns.
                #
                # For stdio servers: the subprocess lives for the whole turn (one
                # asyncio.run()), then dies. Cross-turn state is NOT preserved.
                # For HTTP servers: the server process is long-lived; per-turn sessions
                # are just reconnections and GDB state survives across turns.
                #
                # Optimisation: if MCP tools are unlikely to be called (e.g. a pure
                # text conversation), skipping the session setup would save latency.
                # For now we always rebuild; a future version could check the user
                # message for signals before opening sessions.
                cfg = self._available_agents[agent_name]
                checkpointer = self._checkpointers[agent_name]
                available = self._available_agents

                async def _mcp_runner(chunk_q, _cur=current_input) -> None:
                    from langchain_mcp_adapters.tools import load_mcp_tools

                    async with contextlib.AsyncExitStack() as stack:
                        sessions = {
                            name: await stack.enter_async_context(
                                self._mcp_client.session(name)
                            )
                            for name in self._mcp_server_names
                        }
                        mcp_tools: list = []
                        for name, sess in sessions.items():
                            mcp_tools.extend(
                                await load_mcp_tools(sess, server_name=name)
                            )
                        turn_agent = self._build_agent(
                            agent_name,
                            cfg,
                            all_agents=available,
                            extra_tools=mcp_tools,
                            checkpointer=checkpointer,
                        )
                        async for item in turn_agent.astream(
                            _cur,
                            config=config,
                            stream_mode=["messages", "updates"],
                            subgraphs=True,
                        ):
                            chunk_q.put(("chunk", item))

                steered, resume, parts = self._stream(
                    None, config, current_input, async_agent_runner=_mcp_runner
                )
            else:
                steered, resume, parts = self._stream(agent, config, current_input)

            response_parts.extend(parts)

            if steered:
                self._io.write(f"[steered] {self._steer_value}", style="bold cyan")
                return True, ""
            if resume is not None:
                current_input = resume
                continue

            return False, f"[{agent_name}]: {''.join(response_parts)}"

    def _setup_turn(
        self, user_msg: str, force_agent: str | None = None
    ) -> tuple[str, object, dict]:
        """Route, show agent header, return (name, agent, langgraph_config)."""
        if force_agent:
            name = force_agent
        else:
            self._io.set_status("routing…")
            name = self._route(user_msg)
        self._last_agent = name  # always track, including force_agent path
        self._io.write(f"[{name}]", style="bold blue")
        self._io.set_status("Agent is running…")
        return (
            name,
            self._agents[name],
            {"configurable": {"thread_id": name}},  # stable per-agent ID
        )

    # ── streaming ────────────────────────────────────────────────────────────

    def _stream(
        self,
        agent,
        config: dict,
        current_input: dict | Command,
        *,
        async_agent_runner=None,
    ) -> tuple[bool, Command | None, list[str]]:
        """
        Drive one agent.stream() (or astream()) call.
        Returns (steered, resume_command_or_None, accumulated_text_parts).

        The generator runs in a producer thread so that the worker thread can
        check the interrupt/stop/steer events every POLL_MS milliseconds instead
        of only between chunks (which could be 10s+ apart during LLM thinking or
        tool execution).

        async_agent_runner: if provided, the producer runs it via asyncio.run()
        instead of the default agent.stream() path. Used for MCP stateful sessions
        (see _run_turn). In that case `agent` may be None.
        """
        POLL_MS = 0.05  # 50 ms

        chunk_q: queue.Queue = queue.Queue()

        if async_agent_runner is not None:
            # MCP path: the runner opens sessions, rebuilds the agent with
            # session-bound tools, and iterates astream(). All of this happens
            # inside a single asyncio.run() so the MCP subprocess (stdio) or
            # connection (HTTP) stays alive for the entire agent turn.
            def _producer() -> None:
                try:
                    asyncio.run(async_agent_runner(chunk_q))
                except Exception as exc:
                    chunk_q.put(("error", exc))
                finally:
                    chunk_q.put(("done", None))
        else:
            def _producer() -> None:
                try:
                    for item in agent.stream(
                        current_input,
                        config=config,
                        stream_mode=["messages", "updates"],
                        subgraphs=True,
                    ):
                        chunk_q.put(("chunk", item))
                except Exception as exc:
                    chunk_q.put(("error", exc))
                finally:
                    chunk_q.put(("done", None))

        Thread(target=_producer, daemon=True).start()

        text_parts: list[str] = []
        current_block: str | None = None
        current_ns: tuple = ()
        seen_tool_ids: set[str] = set()
        tool_call_args: dict[str, str] = {}
        buf = _LineBuffer(self._io)
        last_activity = time.monotonic()
        shown_wait_secs = 0

        while True:
            # Check control events before blocking on the next chunk.
            if self._stop.is_set():
                buf.flush()
                raise _Stopped()
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()
                buf.flush()
                raise _Interrupted()
            if self._steer_event.is_set():
                buf.flush()
                return True, None, text_parts

            try:
                kind, item = chunk_q.get(timeout=POLL_MS)
            except queue.Empty:
                elapsed = int(time.monotonic() - last_activity)
                if elapsed >= 3 and elapsed != shown_wait_secs:
                    self._io.set_status(f"Waiting… ({elapsed}s)")
                    shown_wait_secs = elapsed
                continue

            last_activity = time.monotonic()
            shown_wait_secs = 0

            if kind == "done":
                break
            if kind == "error":
                raise item  # type: ignore[misc]

            namespace, mode, data = item

            if namespace != current_ns:
                current_ns = namespace
                if namespace:
                    buf.flush()
                    current_block = None
                    # namespace entries look like "node_name:uuid"; strip the uuid
                    label = " > ".join(ns.split(":")[0] for ns in namespace)
                    self._io.write(f"  [{label}]", style="dim")

            if mode == "updates" and "__interrupt__" in data:
                buf.flush()
                return (
                    False,
                    self._handle_interrupt(data["__interrupt__"][0].value),
                    text_parts,
                )

            if mode != "messages":
                continue

            chunk, _meta = data

            if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    tool_id = tc.get("id")
                    if tool_id:
                        tool_call_args[tool_id] = tool_call_args.get(
                            tool_id, ""
                        ) + tc.get("args", "")
                    if tc.get("name") and tool_id and tool_id not in seen_tool_ids:
                        seen_tool_ids.add(tool_id)
                        buf.flush()
                        current_block = None
                        self._io.write(f"[tool] {tc['name']}", style="dim")
                continue

            if isinstance(chunk, ToolMessage):
                buf.flush()
                raw_args = tool_call_args.get(chunk.tool_call_id, "")
                if raw_args:
                    try:
                        parsed = json.loads(raw_args)
                        args_str = ", ".join(f"{k}={v!r}" for k, v in parsed.items())
                    except json.JSONDecodeError:
                        args_str = raw_args
                    self._io.write(f"  ({args_str})", style="dim")
                self._io.write(
                    f"  → {str(chunk.content)[:120].replace(chr(10), ' ')}…",
                    style="dim",
                )
                continue

            if not isinstance(chunk, AIMessageChunk) or not chunk.content:
                continue

            content_blocks = (
                chunk.content
                if isinstance(chunk.content, list)
                else [{"type": "text", "text": chunk.content}]
            )
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                match block.get("type"):
                    case "thinking":
                        if text := block.get("thinking", ""):
                            if current_block != "thinking":
                                buf.flush()
                                self._io.write("[thinking]", style="dim")
                                current_block = "thinking"
                            buf.push(text, style="dim")
                    case "reasoning":
                        # OpenAI reasoning summary: "summary" is a list of
                        # {"type": "summary_text", "text": "..."} objects.
                        raw = block.get("summary") or block.get("reasoning") or ""
                        if isinstance(raw, list):
                            text = " ".join(
                                item.get("text", "")
                                for item in raw
                                if isinstance(item, dict)
                            )
                        else:
                            text = raw
                        if text:
                            if current_block != "reasoning":
                                buf.flush()
                                self._io.write("[reasoning]", style="dim")
                                current_block = "reasoning"
                            buf.push(text, style="dim")
                    case "text":
                        if text := block.get("text", ""):
                            if current_block in ("thinking", "reasoning"):
                                buf.flush()
                            current_block = "text"
                            text_parts.append(text)
                            buf.push(text)

        buf.flush()
        return False, None, text_parts

    # ── routing ──────────────────────────────────────────────────────────────

    def _route(self, query: str) -> str:
        """Switch agent only if one is mentioned by name; otherwise stay put."""
        q = query.lower()
        # Fast path: exact name substring match (longest first to avoid prefix collisions).
        for name in sorted(self._agents, key=len, reverse=True):
            if name in q:
                self._io.write(f"[routing → {name}]", style="dim")
                return name

        try:
            resp = self._router.invoke(
                [
                    SystemMessage(content=self._router_prompt()),
                    HumanMessage(content=query),
                ]
            )
            content = resp.content
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            name = content.strip().lower()
            if name in self._agents:
                self._io.write(f"[routing → {name}]", style="dim")
                return name
        except Exception as e:
            self._io.write(f"[routing error: {e}]", style="dim")

        fallback = self._last_agent or self.DEFAULT_AGENT
        return fallback

    def _router_prompt(self) -> str:
        names = ", ".join(self._agents)
        return (
            f"Available agents: {names}.\n\n"
            f"Does the message explicitly reference one of these agents by name "
            f"(including informal variants, e.g. 'use claude', 'switch to gemini', "
            f"'ask opus', 'let flash handle this')?\n\n"
            f"If yes, reply with ONLY that agent name.\n"
            f"If no agent is mentioned by name, reply with ONLY '?'.\n"
            f"Do NOT route based on topic or what the agent is good at — only on "
            f"whether an agent is explicitly named."
        )

    # ── interrupt handling ───────────────────────────────────────────────────

    def _handle_interrupt(self, value: object) -> Command:
        if isinstance(value, dict) and "question" in value:
            self._io.write(f"[agent asks] {value['question']}", style="bold yellow")
            return Command(resume=self._wait_input("> "))

        if isinstance(value, dict) and "action_requests" in value:
            decisions = [
                self._handle_hitl_action(req, cfg)
                for req, cfg in zip(value["action_requests"], value["review_configs"])
            ]
            return Command(resume={"decisions": decisions})

        self._io.write(f"[interrupt] {value!r}", style="yellow")
        self._wait_input("Press Enter to continue… ")
        return Command(resume=None)

    def _show_diff(self, diff: str) -> None:
        for line in diff.splitlines():
            if line.startswith("+"):
                self._io.write(line, style="green")
            elif line.startswith("-"):
                self._io.write(line, style="red")
            else:
                self._io.write(line, style="dim")

    def _handle_hitl_action(self, action_req: dict, review_cfg: dict) -> dict:
        name, args = action_req["name"], action_req["args"]
        allowed = review_cfg["allowed_decisions"]
        opts = "/".join(allowed)

        self._io.write(f"[approve?] tool={name}", style="bold yellow")
        if name == "edit_file":
            self._show_diff(preview_diff(args["path"], args["search"], args["replace"]))
        elif name == "write_file":
            self._show_diff(preview_write(args["path"], args["content"]))
        else:
            self._io.write(json.dumps(args, indent=2), style="dim")

        while True:
            choice = self._wait_input(f"[{opts}]> ").lower()
            if choice in allowed:
                break
            self._io.write(f"  choose one of: {opts}")

        match choice:
            case "approve":
                return {"type": "approve"}
            case "reject":
                msg = self._wait_input("rejection message (optional): ")
                return {"type": "reject", **({"message": msg} if msg else {})}
            case "edit":
                self._io.write(f"  current args: {json.dumps(args)}")
                raw = self._wait_input("new args (JSON): ")
                try:
                    return {
                        "type": "edit",
                        "edited_action": {"name": name, "args": json.loads(raw)},
                    }
                except json.JSONDecodeError:
                    self._io.write("  invalid JSON — approving as-is")
                    return {"type": "approve"}

    # ── input synchronisation ────────────────────────────────────────────────

    def _wait_input(self, prompt: str = "> ") -> str:
        """Block the worker thread until submit() is called. Returns the submitted text."""
        self._waiting_for_input = True
        self._input_event.clear()
        self._io.set_status(prompt)
        self._input_event.wait()
        self._waiting_for_input = False
        if self._stop.is_set():
            raise _Stopped()
        return self._input_value
