"""qq — quick query CLI.

Reads a query from CLI arguments and optional context from stdin, then
answers using a fast model with optional tool access.

Usage:
    qq [-e] <query words...>
    <cmd> | qq [-e] <query words...>
    qq [-e] <query words...> -        (then type input, Ctrl-D to finish)

Flags:
    -e    Echo stdin to stdout before the answer (useful in pipelines).
    -     Read from stdin explicitly (even when running interactively).
"""

import re
import sys
from argparse import ArgumentParser
from datetime import date

from langchain.chat_models import init_chat_model
from platformdirs import user_config_path

from tools import invoke_with_tools, standard_tools

# ── model ─────────────────────────────────────────────────────────────────────

MODEL_ID = "google_genai:gemini-3.1-flash-lite-preview"
MODEL_KWARGS = {"thinking_level": "minimal"}

SYSTEM_PROMPT_BASE = """\
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

_CONFIG_PATH = user_config_path("ai-tools") / "config.toml"


def _read_config() -> dict:
    try:
        import tomllib

        return tomllib.loads(_CONFIG_PATH.read_text())
    except FileNotFoundError:
        return {}


def _build_system_prompt() -> str:
    config = _read_config()
    context_lines: list[str] = [f"Today's date: {date.today().isoformat()}"]
    if location := config.get("location"):
        context_lines.append(f"Location: {location}")
    return SYSTEM_PROMPT_BASE + "\n\n## Context\n" + "\n".join(context_lines)


# ── core ──────────────────────────────────────────────────────────────────────


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
    tools = standard_tools()

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

    model = init_chat_model(MODEL_ID, **MODEL_KWARGS)
    text, _ = invoke_with_tools(model, tools, _build_system_prompt(), human_content)
    return text


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

    force_stdin = "-" in sys.argv
    stdin_data = sys.stdin.read() if (force_stdin or not sys.stdin.isatty()) else ""

    if args.echo and stdin_data:
        sys.stdout.write(stdin_data)
        if not stdin_data.endswith("\n"):
            sys.stdout.write("\n")

    query_words = [w for w in args.query if w != "-"]
    print(run(" ".join(query_words), stdin_data))


if __name__ == "__main__":
    main()
