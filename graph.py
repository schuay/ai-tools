"""Importable agent definition — used by both main.py (CLI) and langgraph dev (Studio)."""

import subprocess

from langchain.chat_models import init_chat_model
from langgraph.types import interrupt

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

V8_REPO = "/home/jakob/src/v8"


# ── tools ───────────────────────────────────────────────────────────────────


def trim_to_context(full_text: str, line: int | None, context: int = 20):
    if line is None:
        return full_text

    lines = full_text.splitlines(keepends=True)
    start = max(0, line - 1 - context)
    end = min(len(lines), line - 1 + context + 1)
    return "".join(
        f"{i + 1:>6}  {'>>>' if i + 1 == line else '   '}  {lines[i]}"
        for i in range(start, end)
    )


def git_show(commit_hash: str) -> str:
    """Show the diff and metadata for a git commit in the v8 repository."""
    result = subprocess.run(
        ["git", "show", commit_hash],
        cwd=V8_REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"Error: {result.stderr.strip()}"
    return result.stdout


def read_around(file_path: str, line: int, context: int = 20) -> str:
    """Read lines around a given line number in a file inside the v8 repository.

    file_path: path relative to the v8 repo root
    line: 1-based line number to centre on
    context: number of lines to show before and after
    """
    import os

    full_path = os.path.join(V8_REPO, file_path)
    try:
        with open(full_path, "r", errors="replace") as f:
            return trim_to_context(f.read(), line, context)
    except OSError as e:
        return f"Error reading {full_path}: {e}"


def git_show_file(
    commit_hash: str, file_path: str, line: int | None = None, context: int = 20
) -> str:
    """Show the content of a file as it existed at a given commit in the v8 repository.

    commit_hash: the git commit hash
    file_path: path relative to the v8 repo root
    line: if given, centre the output on this 1-based line number and show `context` lines around it
    context: lines to show before and after `line` (default 20); ignored when line is not given
    """
    result = subprocess.run(
        ["git", "show", f"{commit_hash}:{file_path}"],
        cwd=V8_REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"Error: {result.stderr.strip()}"

    if line is None:
        return result.stdout

    return trim_to_context(result.stdout, line, context)


def git_blame(
    file_path: str,
    commit_hash: str | None = None,
    line: int | None = None,
    context: int = 20,
) -> str:
    """Show git blame for a file in the v8 repository.

    file_path: path relative to the v8 repo root
    commit_hash: if given, show blame as of that commit; defaults to HEAD
    line: if given, centre the output on this 1-based line number and show `context` lines around it
    context: lines to show before and after `line` (default 20); ignored when line is not given
    """
    cmd = ["git", "blame", "--date=short"]
    if commit_hash:
        cmd.append(commit_hash)
    cmd.append(file_path)

    result = subprocess.run(cmd, cwd=V8_REPO, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr.strip()}"

    if line is None:
        return result.stdout

    return trim_to_context(result.stdout, line, context)


def ask_user(question: str) -> str:
    """Ask the human a clarifying question when you are uncertain about something.

    Use this whenever you need more context, are unsure which aspect of the
    code to focus on, or want to confirm your interpretation before diving deep.
    """
    return interrupt({"question": question})


# ── system prompt ────────────────────────────────────────────────────────────

v8_instructions = """You are an expert V8 JavaScript engine developer.
Your job is to explain git commits from the V8 source repository clearly and concisely.

Always start by calling git_show to read the diff. Use read_around liberally to build
a thorough understanding of the surrounding code before drawing conclusions.

Use ask_user whenever there is something genuinely useful to learn from the human —
whether that's about their background, the focus they want, or something in the diff
that is ambiguous or surprising. The user is a V8 expert; treat them as a peer.

## Source of truth
Only code is ground truth. Comments, commit messages, and user input describe intent
but may be mistaken, stale, or imprecise. When they conflict with the code, trust
the code and note the discrepancy.

## Output format
1. One-sentence summary of what the commit does.
2. Motivation / why this change was made.
3. Technical details: which files/components are affected and how.
4. Any notable side-effects or follow-up considerations.
"""


# ── agent ────────────────────────────────────────────────────────────────────

# available models (passed in from callers; listed here for reference)
# init_chat_model("openai:gpt-5.2")
# init_chat_model("deepseek-chat")
# init_chat_model("deepseek-reasoner")
# init_chat_model("deepseek:deepseek-v3.2-speciale")
# init_chat_model("google_genai:gemini-3-flash-preview", include_thoughts=True)
# init_chat_model("google_genai:gemini-3.1-pro-preview", include_thoughts=True, max_retries=6)

_default_model = init_chat_model(
    "google_genai:gemini-3-flash-preview", include_thoughts=True
)


def _identity_section(name: str, all_agents: dict) -> str:
    """Build a system-prompt preamble that tells the agent who it is."""
    others = "\n".join(
        f"  - {n}: {cfg['description']}"
        for n, cfg in all_agents.items()
        if n != name
    )
    return (
        f"## Identity\n"
        f'Your name in this session is "{name}".\n\n'
        f"## Other agents in this session\n"
        f"{others}\n\n"
        f"When you see a message prefixed with an agent name followed by ':' or ',' "
        f'(e.g. "claude-sonnet: what about X?"), that is a routing directive — '
        f"the prefix names the intended recipient, not a subject of discussion. "
        f"Ignore the prefix and focus on the content.\n\n"
    )


def make_agent(model=None, checkpointer=None, name: str | None = None, agents: dict | None = None):
    identity = _identity_section(name, agents) if name and agents else ""
    return create_deep_agent(
        model=model or _default_model,
        tools=[git_show, git_show_file, git_blame, read_around, ask_user],
        backend=FilesystemBackend(root_dir=V8_REPO, virtual_mode=True),
        system_prompt=identity + v8_instructions,
        checkpointer=checkpointer,
    )


# langgraph dev supplies its own checkpointer — don't pass one here
agent = make_agent()
