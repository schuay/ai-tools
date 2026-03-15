"""Filesystem tools: directory listing, file creation, and search-and-replace editing.

edit_file uses fuzzy matching to locate the search block:
  1. Exact substring match.
  2. Strip trailing whitespace per line, then exact match.
  3. Sliding-window SequenceMatcher over line groups (threshold ≥ 0.85).

User approval for write operations is handled by deepagents' interrupt_on
mechanism: the agent is configured with interrupt_on={"edit_file": True,
"write_file": True}, so execution pauses before these functions are called.
"""

import difflib
import os
import subprocess
from pathlib import Path

from tools._text import cap_lines, resolve_path, trim_to_context


# ── fuzzy matching (internal) ─────────────────────────────────────────────────


def _find(content: str, search: str) -> tuple[int, int] | None:
    """Return (start, end) byte offsets of *search* inside *content*, or None."""

    # Pass 1: exact.
    idx = content.find(search)
    if idx != -1:
        return idx, idx + len(search)

    # Pass 2: strip trailing whitespace per line, match on stripped versions.
    def _strip_trailing(text: str) -> list[str]:
        return [line.rstrip() for line in text.splitlines()]

    c_stripped = _strip_trailing(content)
    s_stripped = _strip_trailing(search)
    n = len(s_stripped)
    if n == 0:
        return None
    for i in range(len(c_stripped) - n + 1):
        if c_stripped[i : i + n] == s_stripped:
            lines = content.splitlines(keepends=True)
            start = sum(len(l) for l in lines[:i])
            end = sum(len(l) for l in lines[: i + n])
            return start, end

    # Pass 3: sliding window with SequenceMatcher.
    THRESHOLD = 0.85
    lines = content.splitlines(keepends=True)
    if n > len(lines):
        return None
    search_joined = search.strip()
    best_ratio, best_i = 0.0, -1
    for i in range(len(lines) - n + 1):
        window = "".join(lines[i : i + n]).strip()
        ratio = difflib.SequenceMatcher(
            None, window, search_joined, autojunk=False
        ).ratio()
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
            if ratio >= 0.99:
                break
    if best_ratio >= THRESHOLD:
        start = sum(len(l) for l in lines[:best_i])
        end = sum(len(l) for l in lines[: best_i + n])
        return start, end

    return None


def _unified_diff(original: str, modified: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
            n=3,
        )
    )


# ── diff previews (for HITL display; not agent tools) ────────────────────────


def preview_diff(path: str, search: str, replace: str) -> str:
    """Unified diff for an edit_file operation (shown to user at approval time)."""
    file_path = resolve_path(path)
    if not file_path.exists():
        return f"Error: {path} does not exist"
    try:
        original = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {path}: {e}"

    match = _find(original, search)
    if match is None:
        return (
            f"Error: could not locate the search block in {path}.\n"
            "Tips: ensure indentation is exact, add more surrounding context lines,\n"
            "or verify the text against the actual file with read_around / git_show."
        )

    start, end = match
    return _unified_diff(
        original, original[:start] + replace + original[end:], file_path.name
    )


def preview_write(path: str, content: str) -> str:
    """Unified diff for a write_file operation (shown to user at approval time)."""
    file_path = resolve_path(path)
    original = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    return _unified_diff(original, content, file_path.name)


# ── agent tools ───────────────────────────────────────────────────────────────


def read_file(
    path: str,
    line: int | None = None,
    context: int = 80,
) -> str:
    """Read a file from the filesystem.

    path: file path (absolute or relative to cwd)
    line: centre output on this line number
    context: lines around `line`, or max output lines (default 80)
    """
    file_path = resolve_path(path)
    if not file_path.exists():
        return f"Error: {path} does not exist"
    if not file_path.is_file():
        return f"Error: {path} is not a file"
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {path}: {e}"

    if line is not None:
        return trim_to_context(content, line, context)
    return cap_lines(content, context)


def grep_files(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    grep_context: int = 0,
    line: int | None = None,
    context: int = 80,
) -> str:
    """Search files with grep -E (for untracked files, logs, build outputs).

    pattern: regex pattern
    path: file or directory (default: cwd)
    glob: file pattern filter (e.g. "*.py")
    grep_context: lines of context (grep -C)
    line: centre output on this line number
    context: lines around `line`, or max output lines (default 80)
    """
    cmd = ["grep", "-rEn", "--color=never"]
    if grep_context:
        cmd += ["-C", str(grep_context)]
    if glob:
        cmd += ["--include", glob]
    cmd += [pattern, path]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    out = r.stdout if r.returncode in (0, 1) else f"Error: {r.stderr.strip()}"
    if not out.strip():
        return "No matches found."
    if line is not None:
        return trim_to_context(out, line, context)
    return cap_lines(out, context)


def list_dir(path: str = ".") -> str:
    """List directory contents (dirs first, with trailing '/').

    path: directory path (default: '.')
    """
    dir_path = Path(os.getcwd()) / path
    if not dir_path.exists():
        return f"Error: {path} does not exist"
    if not dir_path.is_dir():
        return f"Error: {path} is not a directory"

    try:
        entries = sorted(
            dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
    except PermissionError as e:
        return f"Error: {e}"

    lines = []
    for entry in entries:
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            size = entry.stat().st_size
            lines.append(f"{entry.name}  ({size:,} B)")
    return "\n".join(lines) if lines else "(empty)"


def edit_file(path: str, search: str, replace: str) -> str:
    """Find and replace a code block in a file (fuzzy matching).

    Include 3-5 lines of unchanged context so the location is unambiguous.
    Reproduce indentation exactly. The block must be unique.

    path: file path
    search: existing code to find (with context lines)
    replace: replacement code
    """
    file_path = resolve_path(path)
    if not file_path.exists():
        return f"Error: {path} does not exist"

    try:
        original = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {path}: {e}"

    match = _find(original, search)
    if match is None:
        return (
            f"Error: could not locate the search block in {path}.\n"
            "Tips: ensure indentation is exact, add more surrounding context lines,\n"
            "or verify the text against the actual file with read_around / git_show."
        )

    start, end = match
    modified = original[:start] + replace + original[end:]

    try:
        file_path.write_text(modified, encoding="utf-8")
        return f"Applied edit to {file_path}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file. Parent directories are created automatically.

    path: file path
    content: full text content to write
    """
    file_path = resolve_path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Written {file_path}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"
