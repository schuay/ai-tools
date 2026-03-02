"""
Multi-agent conversation session.

All agent logic lives here: registry, routing, conversation history,
LangGraph streaming, and interrupt handling.

The UI supplies a SessionIO implementation and calls:
    session.submit(text)   — user pressed Enter
    session.run()          — blocks; run on a worker thread
"""

from __future__ import annotations

import json
import uuid
from threading import Event
from typing import Protocol
import traceback

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


# ── IO protocol ──────────────────────────────────────────────────────────────


class SessionIO(Protocol):
    """What the Session needs from the UI. Called from the worker thread."""

    def write(self, text: str, style: str | None = None) -> None: ...
    def set_status(self, text: str) -> None: ...


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
            "description": "OpenAI GPT — strong general reasoning",
        },
        "gpt-pro": {
            "model_id": "openai:gpt-5.2-pro",
            "description": "OpenAI GPT — Version of GPT-5.2 that produces smarter and more precise responses.",
        },
        "gpt-mini": {
            "model_id": "openai:gpt-5-mini",
            "description": "OpenAI GPT — GPT-5 mini is a faster, more cost-efficient version of GPT-5. It's great for well-defined tasks and precise prompts.",
        },
        "deepseek": {
            "model_id": "deepseek-chat",
            "description": "DeepSeek chat — good for code, concise answers",
        },
        "deepseek-v3": {
            "model_id": "deepseek:deepseek-v3.2-speciale",
            "description": "DeepSeek v3 special — alternative capable model",
        },
        "gemini-flash": {
            "model_id": "google_genai:gemini-3-flash-preview",
            "include_thoughts": True,
            "description": "Gemini Flash — fast and cheap for simple queries",
        },
        "gemini-pro": {
            "model_id": "google_genai:gemini-3.1-pro-preview",
            "include_thoughts": True,
            "max_retries": 6,
            "description": "Gemini Pro — thorough analysis",
        },
        "claude-haiku": {
            "model_id": "anthropic:claude-haiku-4-5-20251001",
            "description": "The fastest model with near-frontier intelligence",
        },
        "claude-sonnet": {
            "model_id": "anthropic:claude-sonnet-4-6",
            "description": "Claude Sonnet — very strong for code and nuance",
        },
        "claude-opus": {
            "model_id": "anthropic:claude-opus-4-6",
            "description": "Claude Opus — most powerful Claude model",
        },
    }
    DEFAULT_AGENT = "gemini-flash"
    ROUTER_MODEL_ID = "google_genai:gemini-3-flash-preview"

    # Keys in AGENTS entries that are not forwarded to init_chat_model.
    _METADATA_KEYS = frozenset({"model_id", "description"})

    def __init__(self, io: SessionIO, prompt: str) -> None:
        self._io = io
        self._prompt = prompt

        self._agents = self._build_agents()
        self._router = init_chat_model(self.ROUTER_MODEL_ID)
        self._history: list[dict] = []
        self._last_agent: str | None = None

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

    def interrupt(self) -> None:
        """Abort the current agent turn. Called from the UI thread (Esc key)."""
        self._interrupt_event.set()

    def submit(self, text: str) -> None:
        """Called from the UI thread when the user presses Enter."""
        if self._waiting_for_input:
            self._input_value = text
            self._input_event.set()
        else:
            self._steer_value = text
            self._steer_event.set()

    def run(self) -> None:
        """Main loop. Blocks; run on a dedicated worker thread."""
        user_msg = self._prompt
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
                    self._history.append({"role": "assistant", "content": "[interrupted]"})
                    self._io.write("[interrupted]", style="bold yellow")
                    user_msg = self._wait_input("> ")
                    force_agent = None
                    continue
                except Exception as e:
                    self._history.append({"role": "user", "content": user_msg})
                    self._history.append({"role": "assistant", "content": f"[error: {type(e).__name__}]"})
                    self._io.write(f"[error] {type(e).__name__}: {str(e)}\n{traceback.format_exc()}", style="bold red")
                    user_msg = self._wait_input("> ")
                    force_agent = None
                    continue
                if steered:
                    self._history.append({"role": "user", "content": user_msg})
                    self._history.append({"role": "assistant", "content": "[steered]"})
                    user_msg = self._steer_value
                    force_agent = self._last_agent  # stay on the active agent
                    continue
                self._history.append({"role": "user", "content": user_msg})
                self._history.append({"role": "assistant", "content": response})
                user_msg = self._wait_input("> ")
                force_agent = None
        except _Stopped:
            pass

    # ── agent construction ───────────────────────────────────────────────────

    def _build_agents(self) -> dict:
        agents = {}
        for name, cfg in self.AGENTS.items():
            kwargs = {k: v for k, v in cfg.items() if k not in self._METADATA_KEYS}
            agents[name] = make_agent(
                model=init_chat_model(cfg["model_id"], **kwargs),
                checkpointer=MemorySaver(),
            )
        return agents

    # ── turn orchestration ───────────────────────────────────────────────────

    def _run_turn(self, user_msg: str, force_agent: str | None = None) -> tuple[bool, str]:
        """
        Route user_msg, run the agent to completion.
        Returns (steered, response_for_history).
        If steered, response is empty and self._steer_value holds the new message.
        """
        agent_name, agent, config = self._setup_turn(user_msg, force_agent)

        messages = [{"role": m["role"], "content": m["content"]} for m in self._history]
        messages.append({"role": "user", "content": user_msg})
        current_input: dict | Command = {"messages": messages}
        response_parts: list[str] = []

        while True:
            self._steer_event.clear()
            steered, resume, parts = self._stream(agent, config, current_input)
            response_parts.extend(parts)

            if steered:
                self._io.write(f"[steered] {self._steer_value}", style="bold cyan")
                return True, ""
            if resume is not None:
                current_input = resume
                continue

            return False, f"[{agent_name}]: {''.join(response_parts)}"

    def _setup_turn(self, user_msg: str, force_agent: str | None = None) -> tuple[str, object, dict]:
        """Route, show agent header, return (name, agent, langgraph_config)."""
        if force_agent:
            name = force_agent
        else:
            self._io.set_status("routing…")
            name = self._route(user_msg)
            self._last_agent = name
        self._io.write(f"[{name}]", style="bold blue")
        self._io.set_status("Agent is running…")
        return (
            name,
            self._agents[name],
            {"configurable": {"thread_id": str(uuid.uuid4())}},
        )

    # ── streaming ────────────────────────────────────────────────────────────

    def _stream(
        self, agent, config: dict, current_input: dict | Command
    ) -> tuple[bool, Command | None, list[str]]:
        """
        Drive one agent.stream() call.
        Returns (steered, resume_command_or_None, accumulated_text_parts).
        """
        text_parts: list[str] = []
        current_block: str | None = None
        current_ns: tuple = ()
        seen_tool_ids: set[str] = set()
        tool_call_args: dict[str, str] = {}
        buf = _LineBuffer(self._io)

        for namespace, mode, data in agent.stream(
            current_input, config=config, stream_mode=["messages", "updates"], subgraphs=True,
        ):
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
                    case "text":
                        if text := block.get("text", ""):
                            if current_block == "thinking":
                                buf.flush()
                            current_block = "text"
                            text_parts.append(text)
                            buf.push(text)

        buf.flush()
        return False, None, text_parts

    # ── routing ──────────────────────────────────────────────────────────────

    def _route(self, query: str) -> str:
        """
        Pick the best agent for this query.
        Explicit name mention wins over the router LLM.
        """
        q = query.lower()
        # Sort longest-first so "deepseek-r" is matched before "deepseek".
        for name in sorted(self.AGENTS, key=len, reverse=True):
            if name in q:
                self._io.write(f"[routing → {name} (explicit)]", style="dim")
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
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            raw = content.strip()
            normalized = raw.lower()
            # "?" means the router is uncertain → stay on the last active agent.
            if normalized == "?":
                fallback = self._last_agent or self.DEFAULT_AGENT
                self._io.write(f"[routing → {fallback} (continuing)]", style="dim")
                return fallback
            if normalized in self.AGENTS:
                self._io.write(f"[routing → {normalized}]", style="dim")
                return normalized
            # Fuzzy fallback: first agent name contained in the response.
            for name in sorted(self.AGENTS, key=len, reverse=True):
                if name in normalized:
                    self._io.write(f"[routing → {name} (fuzzy: {raw!r})]", style="dim")
                    return name
            fallback = self._last_agent or self.DEFAULT_AGENT
            self._io.write(
                f"[routing → {fallback} (no match: {raw!r})]", style="dim"
            )
            return fallback
        except Exception as e:
            fallback = self._last_agent or self.DEFAULT_AGENT
            self._io.write(
                f"[routing → {fallback} (error: {e})]", style="dim"
            )
        return self._last_agent or self.DEFAULT_AGENT

    def _router_prompt(self) -> str:
        agent_list = ", ".join(self.AGENTS)
        descriptions = "\n".join(
            f"- {name}: {cfg['description']}" for name, cfg in self.AGENTS.items()
        )
        return (
            f"Route the user query to one of these agents: {agent_list}.\n"
            f"{descriptions}\n"
            f"Reply with ONLY the agent name.\n"
            f"Reply with exactly '?' if the query is a continuation or follow-up with no "
            f"clear agent preference (e.g. 'continue', 'go on', 'ok', 'yes', bare questions "
            f"that build on prior context). When in doubt, prefer '?'."
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

    def _handle_hitl_action(self, action_req: dict, review_cfg: dict) -> dict:
        name, args = action_req["name"], action_req["args"]
        allowed = review_cfg["allowed_decisions"]
        opts = "/".join(allowed)

        self._io.write(f"[approve?] tool={name}", style="bold yellow")
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
