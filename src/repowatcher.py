"""repowatcher.py — watch a git repo's origin/main for new commits.

For each new commit, a fast LLM filter decides if it's interesting.
Interesting commits get a full in-depth analysis written to a .md file.

Usage:
    python src/repowatcher.py --repo PATH --output-dir PATH [options]
    uv run repowatcher --repo PATH --output-dir PATH [options]
"""

import argparse
import json
import logging
import queue
import re
import signal
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

import tools.git as _git_mod
from tools import (
    extract_text,
    git_blame,
    git_commit_meta,
    git_commits_since,
    git_commits_since_date,
    git_fetch,
    git_grep,
    git_log,
    git_resolve,
    git_show,
    invoke_with_tools,
    read_around,
)

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "google_genai:gemini-3.1-pro-preview"
FILTER_MODEL = "google_genai:gemini-3-flash-preview"
MODEL_KWARGS = {"include_thoughts": True, "thinking_level": "low"}

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
- Concerns only architectures loong64, ppc, s390, mips64, riscv
- Is a "Update V8 DEPS" commit
- Is a revert

Reply with EXACTLY one word: INTERESTING or SKIP.
No explanation, no punctuation, nothing else.\
"""

ANALYSIS_SYSTEM = """\
You are a V8 engine expert and longtime contributor. You have deep knowledge of V8's
internals: the Ignition interpreter, Maglev and Turbofan JIT compilers, Liftoff and
Turboshaft Wasm tiers, the garbage collector (Orinoco, MinorMS, Scavenger), object
representation, IC system, builtins, CSA/Torque, and the broader Blink/Node.js
integration context.

Your task: write a concise expert commentary on a git commit, as you would in a
technical discussion with a fellow V8 engineer. Start with a max 2 sentence summary
of the commit contents. Otherwise, do NOT repeat or paraphrase the
commit message — assume the reader has already read it. Focus entirely on what
the commit message doesn't say: the broader context, the subsystem implications,
related past work, subtle risks, or why this matters.

Calibrate length to complexity:
- A narrow, self-explanatory change: 2–4 sentences.
- A change with real architectural or performance implications: a few short paragraphs.
  Use prose, not bullet points or section headers.

Use git tools to look at surrounding code, related commits, blame history, and
relevant files before writing. Ground every claim in what you actually see.
Never pad with generic observations.\
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


# ── stats ─────────────────────────────────────────────────────────────────────


@dataclass
class _RunStats:
    interesting: int = 0
    skipped: int = 0
    failed: int = 0
    filter_tokens: int = 0
    analysis_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        *,
        interesting: bool = False,
        skipped: bool = False,
        failed: bool = False,
        filter_tokens: int = 0,
        analysis_tokens: int = 0,
    ) -> None:
        with self._lock:
            if interesting:
                self.interesting += 1
            if skipped:
                self.skipped += 1
            if failed:
                self.failed += 1
            self.filter_tokens += filter_tokens
            self.analysis_tokens += analysis_tokens

    @property
    def total_tokens(self) -> int:
        return self.filter_tokens + self.analysis_tokens

    def summary(self) -> str:
        total = self.interesting + self.skipped + self.failed
        return (
            f"evaluated={total}  interesting={self.interesting}"
            f"  skip={self.skipped}  failed={self.failed}"
            f"  tokens={self.total_tokens:,}"
            f" (filter={self.filter_tokens:,} analysis={self.analysis_tokens:,})"
        )


# ── filter & analysis ─────────────────────────────────────────────────────────


def is_interesting(meta: dict, filter_model) -> tuple[bool, int]:
    msg = meta["subject"]
    if meta.get("body"):
        msg += "\n\n" + meta["body"]
    prompt = f"author: {meta['author']}\ndate: {meta['date']}\n\n{msg}"
    from tools import _invoke_with_backoff, _token_count

    response = _invoke_with_backoff(
        filter_model.invoke,
        [
            SystemMessage(content=FILTER_SYSTEM),
            HumanMessage(content=prompt),
        ],
    )
    verdict = extract_text(response.content).strip().upper()
    return verdict.startswith("INTERESTING"), _token_count(response)


def analyse_commit(
    commit_hash: str,
    meta: dict,
    analysis_model,
    stop_event: threading.Event | None = None,
) -> tuple[str, int]:
    tools = [git_show, git_blame, git_log, git_grep, read_around]
    user_prompt = (
        f"Analyse commit {commit_hash}.\n\n"
        f"Subject: {meta['subject']}\n"
        f"Author:  {meta['author']}\n"
        f"Date:    {meta['date']}\n\n"
        "Use git tools to gather the full context before writing your analysis."
    )
    return invoke_with_tools(
        analysis_model, tools, ANALYSIS_SYSTEM, user_prompt, stop_event=stop_event
    )


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
    stats: _RunStats,
    stop_event: threading.Event | None = None,
) -> None:
    short = commit_hash[:8]
    meta = git_commit_meta(commit_hash)
    logging.info("Evaluating %s — %s …", short, meta["subject"])
    try:
        interesting, filter_tok = is_interesting(meta, filter_model)
    except Exception as e:
        logging.warning("Filter failed for %s: %s — skipping", short, e)
        stats.record(failed=True)
        return

    if not interesting:
        logging.info("%s → SKIP  [filter: %d tok]", short, filter_tok)
        stats.record(skipped=True, filter_tokens=filter_tok)
        state.mark_processed(commit_hash)
        return

    if stop_event and stop_event.is_set():
        return

    logging.info("%s → INTERESTING — analysing …", short)
    try:
        analysis, analysis_tok = analyse_commit(
            commit_hash, meta, analysis_model, stop_event
        )
        out_path = write_output(output_dir, meta, analysis, repo)
        logging.info(
            "%s → written: %s  [filter: %d tok  analysis: %d tok]",
            short,
            out_path,
            filter_tok,
            analysis_tok,
        )
        stats.record(
            interesting=True, filter_tokens=filter_tok, analysis_tokens=analysis_tok
        )
        state.mark_processed(commit_hash)
    except Exception as e:
        logging.warning("%s → analysis failed: %s — will retry", short, e)
        stats.record(failed=True, filter_tokens=filter_tok)


def process_commits(
    commits: list[str],
    repo: Path,
    output_dir: Path,
    state: State,
    filter_model,
    analysis_model,
    workers: int = 1,
    stop_event: threading.Event | None = None,
) -> None:
    already = state.processed
    pending = [c for c in commits if c not in already]
    if not pending:
        logging.info("All %d commit(s) already processed.", len(commits))
        return

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    logging.info(
        "Processing %d new commit(s) with %d worker(s).", len(pending), workers
    )

    stats = _RunStats()

    if workers <= 1:
        for commit_hash in pending:
            if _stopped():
                break
            _process_one(
                commit_hash,
                repo,
                output_dir,
                state,
                filter_model,
                analysis_model,
                stats,
                stop_event,
            )
    else:
        # Parallel path: daemon threads so Ctrl-C exits immediately.
        q: queue.Queue = queue.Queue()
        for c in pending:
            q.put(c)

        def _worker() -> None:
            while not _stopped():
                try:
                    commit_hash = q.get_nowait()
                except queue.Empty:
                    return
                _process_one(
                    commit_hash,
                    repo,
                    output_dir,
                    state,
                    filter_model,
                    analysis_model,
                    stats,
                    stop_event,
                )

        threads = [
            threading.Thread(target=_worker, daemon=True)
            for _ in range(min(workers, len(pending)))
        ]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                if _stopped():
                    logging.info(
                        "Stop requested — abandoning in-flight analyses (will retry)."
                    )
                    return
                t.join(timeout=1.0)

    logging.info("Summary: %s", stats.summary())


# ── modes ─────────────────────────────────────────────────────────────────────


def _setup_signal_handlers(stop_event: threading.Event) -> None:
    """Install SIGINT/SIGTERM handlers that set stop_event, then restore defaults."""

    def _handle(signum, frame):
        logging.info(
            "Signal %s received — stopping (Ctrl-C again to force quit) …", signum
        )
        stop_event.set()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


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
    stop_event = threading.Event()
    _setup_signal_handlers(stop_event)

    from_ref = git_resolve(from_hash)
    to_ref = git_resolve(to_hash)
    commits = git_commits_since(from_ref, to_ref)
    logging.info(
        "Range %s..%s → %d commit(s)",
        from_hash[:8] if len(from_hash) > 8 else from_hash,
        to_hash[:8] if len(to_hash) > 8 else to_hash,
        len(commits),
    )
    process_commits(
        commits,
        repo,
        output_dir,
        state,
        filter_model,
        analysis_model,
        workers,
        stop_event,
    )


def run_since(
    repo: Path,
    output_dir: Path,
    state: State,
    filter_model,
    analysis_model,
    since: str,
    ref: str = "HEAD",
    workers: int = 1,
) -> None:
    stop_event = threading.Event()
    _setup_signal_handlers(stop_event)

    commits = git_commits_since_date(since, ref)
    logging.info("--since %r on %s → %d commit(s)", since, ref, len(commits))
    process_commits(
        commits,
        repo,
        output_dir,
        state,
        filter_model,
        analysis_model,
        workers,
        stop_event,
    )


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
    start_from: str | None = None,
) -> None:
    stop_event = threading.Event()
    _setup_signal_handlers(stop_event)

    remote_ref = f"{remote}/{branch}"

    # First run: optionally backfill from a known start point, then set daemon_tip
    if state.daemon_tip is None:
        git_fetch(remote)
        tip = git_resolve(remote_ref)
        if tip.startswith("Error:"):
            logging.error("Cannot resolve %s: %s", remote_ref, tip)
            sys.exit(1)
        if start_from:
            from_ref = git_resolve(start_from)
            if from_ref.startswith("Error:"):
                logging.error(
                    "Cannot resolve --start-from %r: %s", start_from, from_ref
                )
                sys.exit(1)
            commits = git_commits_since(from_ref, tip)
            logging.info(
                "First run — backfilling %d commit(s) from %s to %s …",
                len(commits),
                start_from,
                tip[:8],
            )
            process_commits(
                commits,
                repo,
                output_dir,
                state,
                filter_model,
                analysis_model,
                workers,
                stop_event,
            )
        else:
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
                        commits,
                        repo,
                        output_dir,
                        state,
                        filter_model,
                        analysis_model,
                        workers,
                        stop_event,
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
        "--since",
        dest="since_spec",
        metavar="DATE",
        help="One-shot mode: process commits since DATE (passed to git --since, e.g. '2 weeks ago', '2024-01-01')",
    )
    parser.add_argument(
        "--range",
        dest="range_spec",
        metavar="FROM..TO",
        help="One-shot mode: process commits in FROM..TO range",
    )
    parser.add_argument(
        "--start-from",
        metavar="REF",
        help="Daemon mode: on first run, backfill commits from REF (hash, tag, or branch) to current tip",
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

    if args.since_spec:
        remote_ref = f"{args.remote}/{args.branch}"
        run_since(
            repo,
            output_dir,
            state,
            filter_model,
            analysis_model,
            since=args.since_spec,
            ref=remote_ref,
            workers=args.workers,
        )
    elif args.range_spec:
        # One-shot range mode
        parts = args.range_spec.split("..")
        if len(parts) != 2:
            logging.error(
                "--range must be in FROM..TO format, got: %s", args.range_spec
            )
            sys.exit(1)
        from_ref, to_ref = parts
        run_range(
            repo,
            output_dir,
            state,
            filter_model,
            analysis_model,
            from_ref,
            to_ref,
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
            start_from=args.start_from,
        )


if __name__ == "__main__":
    main()
