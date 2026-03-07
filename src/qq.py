"""qq — quick query CLI.

Reads a query from CLI arguments and optional context from stdin, then
answers using a fast model with optional tool access.

Usage:
    qq [-e] <query words...>
    <cmd> | qq [-e] <query words...>

Flags:
    -e    Echo stdin to stdout before the answer (useful in pipelines).
"""

import os
import re
import sys
from argparse import ArgumentParser

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools import (
    git_blame,
    git_log,
    git_show,
    read_around,
    web_fetch,
    web_search,
)
from tools.git import in_git_repo

# ── model ─────────────────────────────────────────────────────────────────────

MODEL_ID = "google_genai:gemini-3.1-flash-lite-preview"
MODEL_KWARGS = {"thinking_level": "minimal"}

SYSTEM_PROMPT = """\
You are a highly experienced senior V8 JavaScript engine engineer and general-purpose
technical researcher, embedded in a developer's terminal as a quick-answer tool.

You have deep working knowledge of V8 internals (parser/AST, Ignition, Maglev,
TurboFan, GC/heap, inline caches, object model, Wasm tiers, embedder API) and are
equally comfortable with general software engineering, systems programming, and
web research.

Answer directly and briefly — no preamble, no sign-off. Match depth to the question.

## Source of truth
Code is the only ground truth. Comments, commit messages, and descriptions convey
intent but may be stale or wrong. When they conflict with the code, trust the code.

## Tool use
Use tools only when you cannot answer confidently from your own knowledge:
- Use git tools to inspect actual code, history, or blame rather than guessing.
- Use web_search / web_fetch for current information, external docs, or bug reports.
- Do not call tools for facts you already know.\
"""


# ── core ──────────────────────────────────────────────────────────────────────


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


STDIN_INLINE_LIMIT = 8_000  # chars; beyond this, expose grep/read tools instead
STDIN_PREVIEW_LINES = 20


def _make_stdin_tools(data: str):
    """Return (grep_stdin, read_stdin) closures operating on an in-memory string."""
    lines = data.splitlines()

    def grep_stdin(pattern: str, context_lines: int = 2) -> str:
        """Search stdin content with a regex pattern.

        pattern: regex to search for (Python re syntax)
        context_lines: lines of context before/after each match (default 2)
        """
        out: list[str] = []
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                out.append(f"--- line {i + 1} ---")
                out.extend(f"{j + 1:>6}  {lines[j]}" for j in range(start, end))
        return "\n".join(out) if out else "No matches found."

    def read_stdin(start_line: int = 1, num_lines: int = 100) -> str:
        """Read a range of lines from stdin content.

        start_line: 1-based line to start from (default 1)
        num_lines: number of lines to return (default 100)
        """
        start = max(0, start_line - 1)
        end = min(len(lines), start + num_lines)
        return "\n".join(f"{i + 1:>6}  {lines[i]}" for i in range(start, end))

    return grep_stdin, read_stdin


def run(query: str, stdin_data: str) -> str:
    tools: list = [web_fetch]
    if os.environ.get("TAVILY_API_KEY"):
        tools = [web_search] + tools
    if in_git_repo():
        tools = [git_show, git_blame, git_log, read_around] + tools

    human_content = query
    if stdin_data:
        if len(stdin_data) <= STDIN_INLINE_LIMIT:
            human_content = f"{query}\n\n<stdin>\n{stdin_data.strip()}\n</stdin>"
        else:
            # Too large to inline: show a preview and expose search/read tools.
            grep_stdin, read_stdin = _make_stdin_tools(stdin_data)
            tools = [grep_stdin, read_stdin] + tools
            preview = "\n".join(stdin_data.splitlines()[:STDIN_PREVIEW_LINES])
            total_lines = stdin_data.count("\n") + 1
            human_content = (
                f"{query}\n\n"
                f"<stdin total_lines={total_lines}>\n"
                f"{preview}\n"
                f"... (use grep_stdin / read_stdin to explore the rest)\n"
                f"</stdin>"
            )

    model = init_chat_model(MODEL_ID, **MODEL_KWARGS).bind_tools(tools)
    tool_map = {fn.__name__: fn for fn in tools}

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    while True:
        response: AIMessage = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return _extract_text(response.content)

        for tc in response.tool_calls:
            try:
                result = tool_map[tc["name"]](**tc["args"])
            except Exception as e:
                result = f"Error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = ArgumentParser(description="Quick query CLI")
    parser.add_argument(
        "-e",
        "--echo",
        action="store_true",
        help="Echo stdin to stdout before the answer",
    )
    parser.add_argument("query", nargs="+")
    args = parser.parse_args()

    stdin_data = "" if sys.stdin.isatty() else sys.stdin.read()

    if args.echo and stdin_data:
        sys.stdout.write(stdin_data)
        if not stdin_data.endswith("\n"):
            sys.stdout.write("\n")

    print(run(" ".join(args.query), stdin_data))


if __name__ == "__main__":
    main()
