"""Git read-only tools for use by LangChain/LangGraph agents."""

import os
import subprocess

from tools._text import cap_lines, trim_to_context

REPO_ROOT = os.getcwd()


# ── internal helpers ──────────────────────────────────────────────────────────


def _git(args: list[str]) -> str:
    r = subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, errors="replace"
    )
    return r.stdout if r.returncode == 0 else f"Error: {r.stderr.strip()}"


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


def git_grep(
    pattern: str,
    path: str | None = None,
    git_context: int = 0,
    line: int | None = None,
    context: int = 80,
) -> str:
    """Search for a pattern across all tracked files in the git repository.

    pattern: the search pattern (passed to git grep -E)
    path: optional path to restrict the search (relative to repo root)
    git_context: lines of context around each match (git grep -C)
    line: if given, centre the output on this 1-based line number and show `context` lines around it
    context: lines to show before and after `line` (default 80); when line is not given, limits total output lines
    """
    cmd = ["git", "grep", "-En", "--heading"]
    if git_context:
        cmd += ["-C", str(git_context)]
    cmd.append(pattern)
    if path:
        cmd += ["--", path]
    out = _git(cmd)
    if out.startswith("Error:"):
        return out
    if line is not None:
        return trim_to_context(out, line, context)
    return cap_lines(out, context)


def git_show(
    commit_hash: str,
    file_path: str | None = None,
    line: int | None = None,
    context: int = 80,
) -> str:
    """Show a git commit or a file's content at a given commit.

    commit_hash: the git commit hash
    file_path: if given, show the content of this file at the commit (relative to repo root);
               if omitted, show the full commit diff and metadata
    line: if given, centre the output on this 1-based line number and show `context` lines around it
    context: lines to show before and after `line` (default 80); when line is not given, limits total output lines
    """
    cmd = ["git", "show", f"{commit_hash}:{file_path}" if file_path else commit_hash]
    out = _git(cmd)
    if out.startswith("Error:"):
        return out
    if line is not None:
        return trim_to_context(out, line, context)
    return cap_lines(out, context)


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
    if line is not None:
        return trim_to_context(out, line, context)
    return cap_lines(out, 150)


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


# ── plumbing (non-tool helpers used by repowatcher etc.) ──────────────────────


def git_fetch(remote: str = "origin") -> None:
    """Fetch from a remote in REPO_ROOT (silently ignores errors)."""
    result = _git(["git", "fetch", remote])
    if result.startswith("Error:"):
        import logging

        logging.warning("git fetch %s failed: %s", remote, result)


def git_resolve(ref: str) -> str:
    """Resolve a ref (branch, tag, hash) to a full commit hash."""
    return _git(["git", "rev-parse", ref]).strip()


def git_commits_since(since: str, until: str) -> list[str]:
    """Return commit hashes in (since, until], oldest first."""
    out = _git(["git", "rev-list", "--reverse", f"{since}..{until}"])
    if out.startswith("Error:") or not out:
        return []
    return out.splitlines()


def git_commits_since_date(since: str, ref: str = "HEAD") -> list[str]:
    """Return commit hashes reachable from ref with date >= since, oldest first.

    since is passed directly to git --since and accepts any format git understands:
    "2 weeks ago", "2024-01-01", "yesterday", "2024-03-01T00:00:00", etc.
    """
    out = _git(["git", "log", "--reverse", "--format=%H", f"--since={since}", ref])
    if out.startswith("Error:") or not out:
        return []
    return out.splitlines()


def git_commit_meta(commit_hash: str) -> dict[str, str]:
    """Return a dict with keys: hash, author, date, subject, body."""
    fmt = "%H%n%ae%n%ci%n%s%n%n%b"
    out = _git(["git", "show", "-s", f"--format={fmt}", commit_hash])
    lines = out.splitlines()
    if len(lines) < 4:
        return {
            "hash": commit_hash,
            "author": "",
            "date": "",
            "subject": "",
            "body": "",
        }
    return {
        "hash": lines[0],
        "author": lines[1],
        "date": lines[2],
        "subject": lines[3],
        "body": "\n".join(lines[4:]).strip(),
    }


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
