import logging
import os
import random
import threading
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools._text import trim_to_context
from tools.fs import edit_file, grep_files, list_dir, read_file, write_file
from tools.git import (
    REPO_ROOT,
    git_blame,
    git_commit_meta,
    git_commits_since,
    git_commits_since_date,
    git_diff,
    git_fetch,
    git_grep,
    git_log,
    git_resolve,
    git_show,
    git_status,
    in_git_repo,
    read_around,
)
from tools.shell import run_shell
from tools.web import web_fetch, web_search


def extract_text(content: object) -> str:
    """Extract plain text from an LLM response content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


_MAX_TOOL_OUTPUT = 50_000


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        k in msg
        for k in ("429", "quota", "rate limit", "resource_exhausted", "exhausted")
    )


def _invoke_with_backoff(fn, *args, max_retries: int = 8, **kwargs):
    """Call fn(*args, **kwargs), retrying with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit(e):
                raise
            wait = (2**attempt) + random.uniform(0, 1)
            logging.warning(
                "Rate limited — retrying in %.0fs (attempt %d/%d)",
                wait,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait)
    raise RuntimeError(f"Rate limit persisted after {max_retries} retries")


def _token_count(msg: AIMessage) -> int:
    m = getattr(msg, "usage_metadata", None)
    if isinstance(m, dict):
        return m.get("total_tokens", 0)
    return 0


def invoke_with_tools(
    model,
    tools: list,
    system_prompt: str,
    user_prompt: str,
    *,
    stop_event: threading.Event | None = None,
) -> tuple[str, int]:
    """Run a model.invoke loop, executing tool calls until a final text response.

    Returns (text, total_tokens). Retries on rate-limit errors with exponential
    backoff. Raises InterruptedError if stop_event is set between turns.
    """
    bound = model.bind_tools(tools) if tools else model
    tool_map = {fn.__name__: fn for fn in tools}
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    total_tokens = 0
    turn = 0
    while True:
        if stop_event and stop_event.is_set():
            raise InterruptedError("stop requested")
        logging.debug(
            "turn=%d  history=%d chars",
            turn,
            sum(len(str(m.content)) for m in messages),
        )
        response: AIMessage = _invoke_with_backoff(bound.invoke, messages)
        turn_tokens = _token_count(response)
        total_tokens += turn_tokens
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            logging.debug("turn=%d  tokens=%d  → done", turn, turn_tokens)
            return extract_text(response.content).strip(), total_tokens
        calls = ", ".join(tc["name"] for tc in response.tool_calls)
        logging.debug("turn=%d  tokens=%d  → calls: %s", turn, turn_tokens, calls)
        for tc in response.tool_calls:
            try:
                result = str(tool_map[tc["name"]](**tc["args"]))
            except Exception as e:
                result = f"Error: {e}"
            if len(result) > _MAX_TOOL_OUTPUT:
                result = (
                    result[:_MAX_TOOL_OUTPUT]
                    + f"\n… (truncated at {_MAX_TOOL_OUTPUT} chars)"
                )
            logging.debug("  %s → %d chars", tc["name"], len(result))
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        turn += 1


def standard_tools(
    *,
    web: bool = True,
    git: bool = True,
    fs: bool = False,
    shell: bool = False,
) -> list:
    """Return the standard tool set for an agent, based on context and flags.

    web:   include web_fetch (always) and web_search (if TAVILY_API_KEY is set)
    git:   include git read tools (only if the cwd is inside a git repo)
    fs:    include filesystem tools: read_file, grep_files, list_dir,
           edit_file, write_file
    shell: include run_shell for arbitrary command execution
    """
    tools: list = []

    if web and os.environ.get("TAVILY_API_KEY"):
        tools.append(web_search)
    if web:
        tools.append(web_fetch)

    if git and in_git_repo():
        tools += [
            git_grep,
            git_show,
            git_blame,
            git_log,
            git_diff,
            git_status,
            read_around,
        ]

    if fs:
        tools += [read_file, grep_files, list_dir, edit_file, write_file]

    if shell:
        tools.append(run_shell)

    return tools


__all__ = [
    "REPO_ROOT",
    "edit_file",
    "extract_text",
    "grep_files",
    "git_blame",
    "git_commit_meta",
    "git_commits_since",
    "git_commits_since_date",
    "git_fetch",
    "git_grep",
    "git_log",
    "git_resolve",
    "git_show",
    "in_git_repo",
    "invoke_with_tools",
    "list_dir",
    "read_file",
    "read_around",
    "run_shell",
    "standard_tools",
    "trim_to_context",
    "web_fetch",
    "web_search",
    "write_file",
]
