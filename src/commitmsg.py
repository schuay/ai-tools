"""commitmsg.py — generate a commit message for the current changes.

Reads `git diff HEAD`, understands the changes (using git tools for
additional context if needed), and prints a commit message.

Usage:
    python commitmsg.py
    uv run commitmsg.py
"""

import subprocess
import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools import git_blame, git_log, git_show, git_show_file, read_around

# ── model ─────────────────────────────────────────────────────────────────────

MODEL_ID = "google_genai:gemini-3-flash-preview"
MODEL_KWARGS = {"include_thoughts": True, "thinking_level": "medium"}

SYSTEM_PROMPT = """\
You are an expert software engineer writing git commit messages.

Given a git diff, write a commit message that accurately and concisely describes
the changes.

Format:
- Subject line: imperative mood, ≤72 characters, no trailing period
- Blank line (if body follows)
- Body (optional): explain WHY if the motivation isn't obvious from the diff alone;
  omit if the subject line is sufficient

Output ONLY the commit message — no commentary, no markdown fences, no preamble.

Use git tools (git_show_file, read_around, git_blame, git_log) when you need more
context to understand the purpose or intent of a change.\
"""

DIFF_INLINE_LIMIT = 20_000

# ── helpers ───────────────────────────────────────────────────────────────────


def _run_git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else f"Error: {r.stderr.strip()}"


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# ── core ──────────────────────────────────────────────────────────────────────


def run() -> str:
    diff = _run_git("diff", "HEAD")
    if not diff.strip():
        return "No changes to commit."

    if len(diff) > DIFF_INLINE_LIMIT:
        diff_content = (
            diff[:DIFF_INLINE_LIMIT]
            + f"\n… (diff truncated at {DIFF_INLINE_LIMIT} chars;"
            + " use git tools for full context)"
        )
    else:
        diff_content = diff

    tools = [git_show_file, read_around, git_blame, git_log, git_show]
    model = init_chat_model(MODEL_ID, **MODEL_KWARGS).bind_tools(tools)
    tool_map = {fn.__name__: fn for fn in tools}

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"<diff>\n{diff_content}\n</diff>"),
    ]

    while True:
        response: AIMessage = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return _extract_text(response.content).strip()

        for tc in response.tool_calls:
            try:
                result = tool_map[tc["name"]](**tc["args"])
            except Exception as e:
                result = f"Error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
