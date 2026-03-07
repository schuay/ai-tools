"""repowatcher.py — watch a git repo's origin/main for new commits.

For each new commit, a fast LLM filter decides if it's interesting.
Interesting commits get a full in-depth analysis written to a .md file.

Usage:
    python src/repowatcher.py --repo PATH --output-dir PATH [options]
    uv run repowatcher --repo PATH --output-dir PATH [options]
"""

import argparse
import concurrent.futures
import json
import logging
import re
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import tools.git as _git_mod
from tools import (
    git_blame,
    git_commit_meta,
    git_commits_since,
    git_fetch,
    git_grep,
    git_log,
    git_resolve,
    git_show,
    read_around,
)

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "google_genai:gemini-3.1-pro-preview"
FILTER_MODEL = "google_genai:gemini-3-flash-preview"
MODEL_KWARGS = {"include_thoughts": True, "thinking_level": "medium"}

FILTER_SYSTEM = """\
You are a senior engineer triaging git commits to decide if they warrant deeper analysis.

A commit is INTERESTING if it:
- Introduces new features, APIs, or significant behaviour changes
- Fixes a notable bug (security, correctness, data-loss)
- Makes architectural or design decisions
- Has substantial impact on performance or reliability

A commit is NOT interesting if it:
- Only updates formatting, style, or whitespace
- Bumps a version number or lockfile with no logic change
- Adds/updates comments or documentation only
- Is a trivial typo fix or rename with no functional effect

Reply with EXACTLY one word: INTERESTING or SKIP.
No explanation, no punctuation, nothing else.\
"""

ANALYSIS_SYSTEM = """\
You are an expert software engineer performing an in-depth analysis of a git commit.

Produce a structured Markdown report with these sections:

## Summary
2–4 sentence overview of what changed and why.

## Changes
Bullet-point breakdown of the key changes. Group related changes.
Reference specific files and functions where helpful.

## Impact & Implications
What are the downstream effects? Who/what is affected?
Any risks, gotchas, or things reviewers should watch for?

## Commit Message Accuracy
Does the commit message accurately capture the intent and scope?
Note any gaps or misleading aspects.

Use the git tools (git_show, git_blame, git_log, git_grep, read_around)
to gather as much context as needed before writing your analysis.
Write clearly and precisely. Avoid padding.\
"""

# ── state management ──────────────────────────────────────────────────────────


class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict = {"processed": [], "daemon_tip": None}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
                # Normalise: ensure processed is a list (we keep as set internally)
            except Exception as e:
                logging.warning("Could not load state file %s: %s", path, e)

    @property
    def processed(self) -> set[str]:
        with self._lock:
            return set(self._data.get("processed", []))

    @property
    def daemon_tip(self) -> str | None:
        with self._lock:
            return self._data.get("daemon_tip")

    def mark_processed(self, commit_hash: str) -> None:
        with self._lock:
            processed = self._data.setdefault("processed", [])
            if commit_hash not in processed:
                processed.append(commit_hash)
            self._write_locked()

    def set_daemon_tip(self, commit_hash: str) -> None:
        with self._lock:
            self._data["daemon_tip"] = commit_hash
            self._write_locked()

    def _write_locked(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)


# ── LLM helpers ───────────────────────────────────────────────────────────────


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


def _run_agent(model, tools: list, system_prompt: str, user_prompt: str) -> str:
    tool_map = {fn.__name__: fn for fn in tools}
    bound = model.bind_tools(tools)
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    while True:
        response: AIMessage = bound.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return _extract_text(response.content).strip()
        for tc in response.tool_calls:
            try:
                result = tool_map[tc["name"]](**tc["args"])
            except Exception as e:
                result = f"Error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))


# ── filter & analysis ─────────────────────────────────────────────────────────

DIFF_LIMIT = 20_000


def is_interesting(diff: str, filter_model) -> bool:
    if len(diff) > DIFF_LIMIT:
        diff = diff[:DIFF_LIMIT] + f"\n… (truncated at {DIFF_LIMIT} chars)"
    response = filter_model.invoke(
        [
            SystemMessage(content=FILTER_SYSTEM),
            HumanMessage(content=f"<diff>\n{diff}\n</diff>"),
        ]
    )
    verdict = _extract_text(response.content).strip().upper()
    return verdict.startswith("INTERESTING")


def analyse_commit(commit_hash: str, meta: dict, analysis_model) -> str:
    tools = [git_show, git_blame, git_log, git_grep, read_around]
    user_prompt = (
        f"Analyse commit {commit_hash}.\n\n"
        f"Subject: {meta['subject']}\n"
        f"Author:  {meta['author']}\n"
        f"Date:    {meta['date']}\n\n"
        "Use git tools to gather the full context before writing your analysis."
    )
    return _run_agent(analysis_model, tools, ANALYSIS_SYSTEM, user_prompt)


# ── output ────────────────────────────────────────────────────────────────────


def _slug(subject: str, max_len: int = 40) -> str:
    s = subject.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s[:max_len].rstrip("-")


def write_output(output_dir: Path, meta: dict, analysis: str, repo: Path) -> Path:
    commit_hash = meta["hash"]
    short_hash = commit_hash[:8]
    # Parse date from git format "YYYY-MM-DD HH:MM:SS +ZZZZ"
    try:
        date_str = meta["date"].split()[0]
    except Exception:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    slug = _slug(meta["subject"])
    filename = f"{date_str}_{short_hash}_{slug}.md"
    out_path = output_dir / filename

    analyzed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frontmatter = (
        "---\n"
        f"commit: {commit_hash}\n"
        f"date: {date_str}\n"
        f"author: {meta['author']}\n"
        f"subject: {meta['subject']}\n"
        f"repo: {repo}\n"
        f"analyzed_at: {analyzed_at}\n"
        "---\n\n"
    )
    out_path.write_text(frontmatter + analysis + "\n")
    return out_path


# ── batch processing ──────────────────────────────────────────────────────────


def _process_one(
    commit_hash: str,
    repo: Path,
    output_dir: Path,
    state: State,
    filter_model,
    analysis_model,
) -> None:
    short = commit_hash[:8]
    logging.info("Evaluating %s …", short)
    diff = git_show(commit_hash, context=10_000)
    try:
        interesting = is_interesting(diff, filter_model)
    except Exception as e:
        logging.warning("Filter failed for %s: %s — skipping", short, e)
        return

    if not interesting:
        logging.info("%s → SKIP", short)
        state.mark_processed(commit_hash)
        return

    logging.info("%s → INTERESTING — analysing …", short)
    meta = git_commit_meta(commit_hash)
    try:
        analysis = analyse_commit(commit_hash, meta, analysis_model)
        out_path = write_output(output_dir, meta, analysis, repo)
        logging.info("%s → written: %s", short, out_path)
        state.mark_processed(commit_hash)
    except Exception as e:
        logging.warning("%s → analysis failed: %s — will retry", short, e)


def process_commits(
    commits: list[str],
    repo: Path,
    output_dir: Path,
    state: State,
    filter_model,
    analysis_model,
    workers: int = 1,
) -> None:
    already = state.processed
    pending = [c for c in commits if c not in already]
    if not pending:
        logging.info("All %d commit(s) already processed.", len(commits))
        return

    logging.info("Processing %d new commit(s) with %d worker(s).", len(pending), workers)
    if workers <= 1:
        for commit_hash in pending:
            _process_one(commit_hash, repo, output_dir, state, filter_model, analysis_model)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one, c, repo, output_dir, state, filter_model, analysis_model
                ): c
                for c in pending
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.warning("Worker error for %s: %s", futures[future][:8], e)


# ── modes ─────────────────────────────────────────────────────────────────────


def run_range(
    repo: Path,
    output_dir: Path,
    state: State,
    filter_model,
    analysis_model,
    from_hash: str,
    to_hash: str,
    workers: int = 1,
) -> None:
    from_ref = git_resolve(from_hash)
    to_ref = git_resolve(to_hash)
    commits = git_commits_since(from_ref, to_ref)
    logging.info(
        "Range %s..%s → %d commit(s)",
        from_hash[:8] if len(from_hash) > 8 else from_hash,
        to_hash[:8] if len(to_hash) > 8 else to_hash,
        len(commits),
    )
    process_commits(commits, repo, output_dir, state, filter_model, analysis_model, workers)


def run_daemon(
    repo: Path,
    output_dir: Path,
    state: State,
    filter_model,
    analysis_model,
    remote: str,
    branch: str,
    poll_seconds: int,
    workers: int = 1,
) -> None:
    stop_event = threading.Event()

    def _handle_signal(signum, frame):
        logging.info("Signal %s received — shutting down …", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    remote_ref = f"{remote}/{branch}"

    # First run: initialise daemon_tip to current tip without backfilling
    if state.daemon_tip is None:
        git_fetch(remote)
        tip = git_resolve(remote_ref)
        if tip.startswith("Error:"):
            logging.error("Cannot resolve %s: %s", remote_ref, tip)
            sys.exit(1)
        logging.info("First run — starting from %s (no backfill)", tip[:8])
        state.set_daemon_tip(tip)

    logging.info("Polling %s every %ds …", remote_ref, poll_seconds)
    while not stop_event.is_set():
        try:
            git_fetch(remote)
            tip = git_resolve(remote_ref)
            daemon_tip = state.daemon_tip

            if tip == daemon_tip:
                logging.debug("No new commits.")
            elif tip.startswith("Error:"):
                logging.warning("Could not resolve %s: %s", remote_ref, tip)
            else:
                commits = git_commits_since(daemon_tip, tip)
                if commits:
                    process_commits(
                        commits, repo, output_dir, state, filter_model, analysis_model, workers
                    )
                state.set_daemon_tip(tip)
        except Exception as e:
            logging.warning("Poll error: %s", e)

        stop_event.wait(poll_seconds)

    logging.info("Exited cleanly.")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch a git repo for new commits and analyse interesting ones."
    )
    parser.add_argument(
        "--repo", required=True, type=Path, help="Path to the git repository"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for analysis .md files",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Path to state JSON file (default: OUTPUT_DIR/state.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Analysis model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--branch", default="main", help="Branch to watch (default: main)"
    )
    parser.add_argument(
        "--remote", default="origin", help="Remote name (default: origin)"
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Poll interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of commits to analyse in parallel (default: 1)",
    )
    parser.add_argument(
        "--range",
        dest="range_spec",
        metavar="FROM..TO",
        help="One-shot mode: process commits in FROM..TO range",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        logging.error("%s is not a git repository", repo)
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_file = args.state_file or (output_dir / "state.json")
    state = State(state_file)

    # Point tools.git at the target repo
    _git_mod.REPO_ROOT = str(repo)

    filter_model = init_chat_model(FILTER_MODEL)
    analysis_model = init_chat_model(args.model, **MODEL_KWARGS)

    if args.range_spec:
        # One-shot range mode
        parts = args.range_spec.split("..")
        if len(parts) != 2:
            logging.error(
                "--range must be in FROM..TO format, got: %s", args.range_spec
            )
            sys.exit(1)
        from_ref, to_ref = parts
        run_range(
            repo, output_dir, state, filter_model, analysis_model, from_ref, to_ref,
            workers=args.workers,
        )
    else:
        # Daemon mode
        run_daemon(
            repo,
            output_dir,
            state,
            filter_model,
            analysis_model,
            remote=args.remote,
            branch=args.branch,
            poll_seconds=args.poll,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
