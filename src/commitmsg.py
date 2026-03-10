"""commitmsg.py — generate a commit message for the current changes.

Reads the relevant diff and prints a commit message suitable for use with
`git commit -m` or `git commit -am`:

    git commit -m  "$(commitmsg)"          # staged changes only (default)
    git commit -am "$(commitmsg -a)"       # all tracked modifications
    git commit -m  "$(commitmsg --full)"   # full message with optional body

Usage:
    commitmsg [-a | --all] [--full]
"""

import argparse
import subprocess
import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools import git_blame, git_log, git_show, read_around

# ── model ─────────────────────────────────────────────────────────────────────

MODEL_ID = "google_genai:gemini-3-flash-preview"
MODEL_KWARGS = {"include_thoughts": True, "thinking_level": "medium"}

SYSTEM_PROMPT_ONELINE = """\
You are an expert software engineer writing git commit messages.

Given a git diff, write a single-line commit message that accurately and
concisely describes the changes as one atomic work step.

Rules:
- Imperative mood, ≤72 characters, no trailing period
- No component/scope tags (no "feat:", "fix:", "component:" prefixes)
- One line only — no body, no blank lines

Output ONLY the subject line — no commentary, no markdown fences, no preamble.\
"""

SYSTEM_PROMPT_FULL = """\
You are an expert software engineer writing git commit messages.

Given a git diff, write a commit message that accurately and concisely describes
the changes.

Format:
- Subject line: imperative mood, ≤72 characters, no trailing period
- Blank line (if body follows)
- Body (optional): explain WHY if the motivation isn't obvious from the diff alone;
  omit if the subject line is sufficient

Output ONLY the commit message — no commentary, no markdown fences, no preamble.

Use git tools (git_show, read_around, git_blame, git_log) when you need more
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


def run(all_changes: bool = False, full: bool = False) -> str:
    # staged only: matches `git commit -m`
    # all tracked: matches `git commit -am` (which stages before committing)
    diff = _run_git("diff", "HEAD" if all_changes else "--staged")
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

    system_prompt = SYSTEM_PROMPT_FULL if full else SYSTEM_PROMPT_ONELINE
    tools = [git_show, read_around, git_blame, git_log]
    kwargs = MODEL_KWARGS if full else {**MODEL_KWARGS, "thinking_level": "minimal"}
    base_model = init_chat_model(MODEL_ID, **kwargs)
    model = base_model.bind_tools(tools) if full else base_model
    tool_map = {fn.__name__: fn for fn in tools}

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"<diff>\n{diff_content}\n</diff>"),
    ]

    while True:
        response: AIMessage = model.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return _extract_text(response.content).strip()

        for tc in response.tool_calls:
            try:
                result = tool_map[tc["name"]](**tc["args"])
            except Exception as e:
                result = f"Error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a git commit message for the current changes."
    )
    parser.add_argument(
        "-a",
        "--all",
        dest="all_changes",
        action="store_true",
        help="Include all tracked modifications (for use with git commit -am)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Emit a full message with optional body (default: one-line only)",
    )
    args = parser.parse_args()
    print(run(all_changes=args.all_changes, full=args.full))


if __name__ == "__main__":
    main()
