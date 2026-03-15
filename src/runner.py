"""Headless agent runner.

Provides run_once(prompt, model, extra_tools) -> str:
  - Loads MCP tools from ~/.config/ai-tools/mcp.json if configured
  - Builds the full LangGraph agent via graph.make_agent()
  - Streams astream() to completion, printing progress to stderr
  - Handles ask_user interrupts automatically (resumes with best-judgment reply)
  - Returns the collected response text

Used by analyze.py for batch automation. session.py keeps its own streaming
loop because it needs threading, interrupt events, and HITL tool approval.
"""

import contextlib
import sys

from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import make_agent


# ── MCP helpers ───────────────────────────────────────────────────────────────


def _load_mcp_config() -> dict:
    try:
        from tools.mcp import load_config

        return load_config()
    except Exception:
        return {}


# ── streaming ─────────────────────────────────────────────────────────────────


async def _astream_to_text(agent, prompt: str, *, config: dict, verbose: bool) -> str:
    """Drive agent.astream() to completion, handle interrupts, return text."""
    text_parts: list[str] = []
    seen_tool_ids: set[str] = set()
    current_input: dict | Command = {"messages": [{"role": "user", "content": prompt}]}

    while True:
        async for item in agent.astream(
            current_input,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            namespace, mode, data = item

            # ── interrupt (ask_user) ──────────────────────────────────────────
            if mode == "updates" and data.get("__interrupt__"):
                value = data["__interrupt__"][0].value
                if verbose and isinstance(value, dict) and "question" in value:
                    print(
                        f"[agent asks] {value['question']} → proceeding autonomously",
                        file=sys.stderr,
                    )
                current_input = Command(
                    resume="Proceed with your best judgment — no human is available."
                )
                break  # restart astream with resume command

            if mode != "messages":
                continue

            chunk, _meta = data

            # ── tool call / result display ────────────────────────────────────
            if verbose and isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    if tc.get("name") and tc.get("id") not in seen_tool_ids:
                        seen_tool_ids.add(tc.get("id"))
                        print(f"[tool] {tc['name']}", file=sys.stderr)
                continue

            if verbose and isinstance(chunk, ToolMessage):
                snippet = str(chunk.content)[:120].replace("\n", " ")
                print(f"  → {snippet}", file=sys.stderr)
                continue

            if not isinstance(chunk, AIMessageChunk) or not chunk.content:
                continue

            # ── text / thinking blocks ────────────────────────────────────────
            blocks = (
                chunk.content
                if isinstance(chunk.content, list)
                else [{"type": "text", "text": chunk.content}]
            )
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                match block.get("type"):
                    case "thinking":
                        if verbose and (text := block.get("thinking", "")):
                            print(f"[thinking] {text[:80]}…", file=sys.stderr)
                    case "text":
                        if text := block.get("text", ""):
                            text_parts.append(text)
                            if verbose:
                                print(text, end="", flush=True, file=sys.stderr)
        else:
            # for-loop completed without a break → astream finished normally
            break

    if verbose:
        print(file=sys.stderr)  # final newline after streamed text

    return "".join(text_parts)


# ── public API ────────────────────────────────────────────────────────────────


async def run_once(
    prompt: str,
    model,
    *,
    extra_tools: list | None = None,
    system_prompt: str | None = None,
    verbose: bool = True,
) -> str:
    """Run the full agent once and return the response text.

    prompt:        user message / task description
    model:         a LangChain chat model instance
    extra_tools:   additional tools beyond the standard set (e.g. run_shell)
    system_prompt: override the default V8-engineer system prompt
    verbose:       print tool calls and streamed text to stderr (default True)
    """
    checkpointer = MemorySaver()
    mcp_config = _load_mcp_config()
    config = {"configurable": {"thread_id": "run"}}

    async def _execute(mcp_tools: list) -> str:
        agent = make_agent(
            model=model,
            checkpointer=checkpointer,
            extra_tools=(extra_tools or []) + mcp_tools,
            system_prompt=system_prompt,
            # interrupt_on=None → no HITL, ask_user not included in tools
        )
        return await _astream_to_text(agent, prompt, config=config, verbose=verbose)

    if mcp_config:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools

            client = MultiServerMCPClient(mcp_config)
            if verbose:
                print(f"[mcp] {', '.join(mcp_config)}", file=sys.stderr)
            async with contextlib.AsyncExitStack() as stack:
                sessions = {
                    name: await stack.enter_async_context(client.session(name))
                    for name in mcp_config
                }
                mcp_tools: list = []
                for name, sess in sessions.items():
                    mcp_tools.extend(await load_mcp_tools(sess, server_name=name))
                return await _execute(mcp_tools)
        except ImportError:
            if verbose:
                print("[mcp] langchain-mcp-adapters not installed", file=sys.stderr)
        except Exception as e:
            if verbose:
                print(f"[mcp] {e}", file=sys.stderr)

    return await _execute([])
