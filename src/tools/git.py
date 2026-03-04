"""Git read-only tools for use by LangChain/LangGraph agents."""

import os
import subprocess

REPO_ROOT = os.getcwd()


# ── internal helpers ──────────────────────────────────────────────────────────


def _git(args: list[str]) -> str:
    r = subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, errors="replace"
    )
    return r.stdout if r.returncode == 0 else f"Error: {r.stderr.strip()}"


def trim_to_context(full_text: str, line: int | None, context: int = 20) -> str:
    if line is None:
        return full_text
    lines = full_text.splitlines(keepends=True)
    start = max(0, line - 1 - context)
    end = min(len(lines), line - 1 + context + 1)
    return "".join(
        f"{i + 1:>6}  {'>>>' if i + 1 == line else '   '}  {lines[i]}"
        for i in range(start, end)
    )


# ── public helpers ────────────────────────────────────────────────────────────


def in_git_repo() -> bool:
    """Return True if the current working directory is inside a git repository."""
    return (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True
        ).returncode
        == 0
    )


# ── tools ─────────────────────────────────────────────────────────────────────


def git_show(commit_hash: str, line: int | None = None, context: int = 200) -> str:
    """Show the diff and metadata for a git commit in the repository.

    line: if given, centre the output on this 1-based line number and show `context` lines around it
    context: lines to show before and after `line` (default 200); when line is not given, limits total output lines
    """
    out = _git(["git", "show", commit_hash])
    if out.startswith("Error:"):
        return out
    if line is not None:
        return trim_to_context(out, line, context)
    lines = out.splitlines(keepends=True)
    if len(lines) > context:
        return (
            "".join(lines[:context])
            + f"\n[truncated — {len(lines) - context} more lines; use line= to navigate]"
        )
    return out


def git_show_file(
    commit_hash: str, file_path: str, line: int | None = None, context: int = 20
) -> str:
    """Show the content of a file as it existed at a given commit in the repository.

    commit_hash: the git commit hash
    file_path: path relative to the repo root
    line: if given, centre the output on this 1-based line number and show `context` lines around it
    context: lines to show before and after `line` (default 20); ignored when line is not given
    """
    out = _git(["git", "show", f"{commit_hash}:{file_path}"])
    if out.startswith("Error:"):
        return out
    return trim_to_context(out, line, context)


def git_blame(
    file_path: str,
    commit_hash: str | None = None,
    line: int | None = None,
    context: int = 20,
) -> str:
    """Show git blame for a file in the repository.

    file_path: path relative to the repo root
    commit_hash: if given, show blame as of that commit; defaults to HEAD
    line: if given, centre the output on this 1-based line number and show `context` lines around it
    context: lines to show before and after `line` (default 20); ignored when line is not given
    """
    cmd = ["git", "blame", "--date=short"]
    if commit_hash:
        cmd.append(commit_hash)
    cmd.append(file_path)
    out = _git(cmd)
    if out.startswith("Error:"):
        return out
    return trim_to_context(out, line, context)


def git_log(
    limit: int = 10,
    oneline: bool = False,
    grep: str | None = None,
    author: str | None = None,
) -> str:
    """Show the git commit log in the repository.

    Use this to find relevant commits by message content, author, or to see recent history.

    limit: maximum number of commits to show (default 10)
    oneline: if true, show each commit as a single line (hash and subject)
    grep: if provided, only show commits with messages matching this pattern
    author: if provided, only show commits by this author
    """
    cmd = ["git", "log", f"-n{limit}"]
    if oneline:
        cmd.append("--oneline")
    if grep:
        cmd.append(f"--grep={grep}")
    if author:
        cmd.append(f"--author={author}")
    return _git(cmd)


def read_around(file_path: str, line: int, context: int = 20) -> str:
    """Read lines around a given line number in a file inside the repository.

    file_path: path relative to the repo root
    line: 1-based line number to centre on
    context: number of lines to show before and after
    """
    full_path = os.path.join(REPO_ROOT, file_path)
    try:
        with open(full_path, errors="replace") as f:
            return trim_to_context(f.read(), line, context)
    except OSError as e:
        return f"Error reading {full_path}: {e}"
