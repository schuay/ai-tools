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
    """Search tracked files for a pattern (git grep -E).

    pattern: regex search pattern
    path: restrict to this path (relative to repo root)
    git_context: lines of context around matches (git grep -C)
    line: centre output on this line number
    context: lines around `line`, or max output lines (default 80)
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
    """Show a commit diff, or a file at a given commit.

    commit_hash: git commit hash
    file_path: show this file at the commit; if omitted, show full commit
    line: centre output on this line number
    context: lines around `line`, or max output lines (default 80)
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
    """Show git blame for a file.

    file_path: path relative to repo root
    commit_hash: blame as of this commit (default HEAD)
    line: centre output on this line number
    context: lines around `line` (default 20)
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
    """Show git commit log.

    limit: max commits (default 10)
    oneline: one line per commit
    grep: filter by commit message pattern
    author: filter by author
    """
    cmd = ["git", "log", f"-n{limit}"]
    if oneline:
        cmd.append("--oneline")
    if grep:
        cmd.append(f"--grep={grep}")
    if author:
        cmd.append(f"--author={author}")
    return _git(cmd)


def git_diff(
    ref: str | None = None,
    file_path: str | None = None,
    staged: bool = False,
    context: int = 80,
) -> str:
    """Show working tree, staged, or inter-commit diffs.

    ref: compare against this ref (e.g. HEAD~1, main); omit for unstaged changes
    file_path: restrict to this file
    staged: if True and no ref, show staged changes
    context: max output lines (default 80)
    """
    cmd = ["git", "diff"]
    if staged and not ref:
        cmd.append("--cached")
    if ref:
        cmd.append(ref)
    if file_path:
        cmd += ["--", file_path]
    return cap_lines(_git(cmd), context)


def git_status() -> str:
    """Show the working tree status: modified, staged, and untracked files."""
    return _git(["git", "status", "--short"])


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
    """Read lines around a line number in a repo file.

    file_path: path relative to repo root
    line: 1-based line number to centre on
    context: lines to show before and after
    """
    full_path = os.path.join(REPO_ROOT, file_path)
    try:
        with open(full_path, errors="replace") as f:
            return trim_to_context(f.read(), line, context)
    except OSError as e:
        return f"Error reading {full_path}: {e}"
