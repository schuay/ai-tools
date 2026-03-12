"""analyze — automated performance-trace analysis.

For each input trace file, launches a deep agent with shell access and writes
a Markdown report alongside the input (or into --output-dir).

Usage:
    analyze [--model MODEL] [--thinking low|medium|high] [--output-dir DIR]
            [--system-prompt FILE] [--quiet] <file>...

The agent has access to:
  - run_d8     (execute d8 with given args, optional stdout/stderr redirection)
  - read_file, grep_files, list_dir, edit_file, write_file
  - web_search / web_fetch (if TAVILY_API_KEY is set)
  - git tools (if the cwd is a git repo)
  - MCP tools (if ~/.config/ai-tools/mcp.json is configured)
"""

import asyncio
import sys
from argparse import ArgumentParser
from pathlib import Path

from langchain.chat_models import init_chat_model

import runner
from tools.shell import run_d8

# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a performance engineer with deep expertise in Linux perf, V8/JavaScript
engine profiling, flame graphs, and systems-level performance analysis.

## Source of truth
Always inspect actual data before drawing conclusions. Run tools to get real
numbers rather than speculating. Prefer precise, evidence-backed statements.

## Working approach
1. Inspect the trace file: determine its format (perf.data, cpuprofile, JSON,
   text report, …) and choose the right analysis tools accordingly.
2. Identify hotspots: extract the top functions/call stacks by CPU time or
   sample count. Use read_file, grep_files, or run_d8 as appropriate.
3. Drill down: for significant hotspots, look up symbol addresses, source
   locations, and caller/callee chains as needed. Use run_d8 to re-run d8
   with profiling flags (e.g. --prof, --log-maps) and redirect output to files
   for further analysis.
4. Correlate: cross-reference with source code (read_file, grep_files, git
   tools) and external documentation (web_search) to understand *why* something
   is hot and what can be done about it.
5. Write findings: produce a complete Markdown report — no placeholders or
   deferred sections. Concrete file paths, function names, and line numbers
   where known.

## Output
Write the report using write_file to the path specified in the task. Structure:

  # Performance Analysis: <filename>
  ## Executive Summary
  ## Hotspots
  ## Root Cause Analysis
  ## Optimization Opportunities
  ## Recommended Next Steps
"""

# ── core ──────────────────────────────────────────────────────────────────────


def _build_prompt(trace_path: Path, output_path: Path) -> str:
    return (
        f"Analyze the performance trace at: {trace_path.resolve()}\n\n"
        f"Write the complete Markdown report to: {output_path.resolve()}\n\n"
        f"Use run_shell, read_file, grep_files, and any other tools needed to "
        f"perform a thorough analysis. Do not stop until the report is written."
    )


def _make_model(model_id: str, thinking: str):
    kwargs: dict = {}
    if "gemini" in model_id:
        kwargs["include_thoughts"] = True
        kwargs["thinking_level"] = thinking
        kwargs["max_retries"] = 6
    elif "claude" in model_id:
        budget = {"low": 2000, "medium": 8000, "high": 16000}[thinking]
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return init_chat_model(model_id, **kwargs)


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = ArgumentParser(description="Batch AI performance-trace analysis")
    parser.add_argument(
        "--model",
        default="google_genai:gemini-3.1-pro-preview",
        metavar="MODEL",
        help="LangChain model ID (default: google_genai:gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--thinking",
        choices=["low", "medium", "high"],
        default="high",
        help="Thinking/reasoning level (default: high)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write reports here (default: same directory as the input file)",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a file containing the system prompt (default: built-in prompt)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress streaming output; only show file-level progress",
    )
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    if args.system_prompt is not None:
        system_prompt = args.system_prompt.read_text(encoding="utf-8")
    else:
        system_prompt = SYSTEM_PROMPT

    model = _make_model(args.model, args.thinking)
    errors: list[str] = []

    for trace_path in args.files:
        if not trace_path.exists():
            print(f"[skip] {trace_path}: not found", file=sys.stderr)
            errors.append(str(trace_path))
            continue

        out_dir = args.output_dir or trace_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / (trace_path.stem + ".md")

        print(f"\n[analyze] {trace_path} → {output_path}", file=sys.stderr)

        prompt = _build_prompt(trace_path, output_path)
        try:
            result = asyncio.run(
                runner.run_once(
                    prompt,
                    model,
                    extra_tools=[run_d8],
                    system_prompt=system_prompt,
                    verbose=not args.quiet,
                )
            )
        except Exception as e:
            print(f"[error] {trace_path}: {e}", file=sys.stderr)
            errors.append(str(trace_path))
            continue

        # If the agent wrote the file itself (via write_file tool), we're done.
        # Otherwise, fall back to writing the collected response text.
        if output_path.exists():
            print(f"[done]  {output_path}", file=sys.stderr)
        else:
            output_path.write_text(result, encoding="utf-8")
            print(f"[wrote] {output_path}", file=sys.stderr)

    if errors:
        print(f"\n[failed] {len(errors)} file(s): {', '.join(errors)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
