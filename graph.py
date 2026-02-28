"""Importable agent definition — used by both main.py (CLI) and langgraph dev (Studio)."""
import subprocess

from langchain.chat_models import init_chat_model
from langgraph.types import interrupt

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

V8_REPO = "/home/jakob/src/v8"


# ── tools ───────────────────────────────────────────────────────────────────

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
            lines = f.readlines()
    except OSError as e:
        return f"Error reading {full_path}: {e}"

    start = max(0, line - 1 - context)
    end = min(len(lines), line - 1 + context + 1)
    return "".join(
        f"{i + 1:>6}  {'>>>' if i + 1 == line else '   '}  {lines[i]}"
        for i in range(start, end)
    )


def ask_user(question: str) -> str:
    """Ask the human a clarifying question when you are uncertain about something.

    Use this whenever you need more context, are unsure which aspect of the
    code to focus on, or want to confirm your interpretation before diving deep.
    """
    return interrupt({"question": question})


# ── system prompt ────────────────────────────────────────────────────────────

v8_instructions = """You are an expert V8 JavaScript engine developer.
Your job is to explain git commits from the V8 source repository clearly and concisely.

You have access to these tools:

## `git_show`
Show the full diff and commit metadata for a given commit hash in the v8 repo.
Always start by calling this to understand what changed.

## `read_around`
Read source lines around a specific location to understand context.
Arguments:
- file_path: path relative to v8 repo root
- line: 1-based line number
- context: lines before/after (default 20)

Use read_around when the diff references code that needs broader context to understand.

## `ask_user`
Ask the human a clarifying question. MANDATORY: you MUST call ask_user at least once
before writing your final explanation. Good questions to ask:
- What level of detail does the user want? (high-level overview vs deep-dive)
- Which subsystem should the focus be on if multiple are touched?
- Is there background context the user already knows that you can skip?
- Does anything in the diff look surprising or unclear to you?

The user is a V8 expert — treat them as a peer, not a student.

Dive deep! Explore using read_around until you have a full understanding of the domain.

## Output format
1. One-sentence summary of what the commit does.
2. Motivation / why this change was made.
3. Technical details: which files/components are affected and how.
4. Any notable side-effects or follow-up considerations.
"""


# ── agent ────────────────────────────────────────────────────────────────────

# model = init_chat_model("openai:gpt-5.2")
# model = init_chat_model("deepseek-chat")
# model = init_chat_model("google_genai:gemini-3-flash-preview", include_thoughts=True)
model = init_chat_model("google_genai:gemini-3.1-pro-preview", include_thoughts=True, max_retries=2)

def make_agent(checkpointer=None):
    return create_deep_agent(
        model=model,
        tools=[git_show, read_around, ask_user],
        backend=FilesystemBackend(root_dir=V8_REPO, virtual_mode=True),
        system_prompt=v8_instructions,
        checkpointer=checkpointer,
    )


# langgraph dev supplies its own checkpointer — don't pass one here
agent = make_agent()
