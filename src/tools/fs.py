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
    file_path = Path(path).expanduser()
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
            "or verify the text against the actual file with read_around / git_show_file."
        )

    start, end = match
    return _unified_diff(
        original, original[:start] + replace + original[end:], file_path.name
    )


def preview_write(path: str, content: str) -> str:
    """Unified diff for a write_file operation (shown to user at approval time)."""
    file_path = Path(path).expanduser()
    original = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    return _unified_diff(original, content, file_path.name)


# ── agent tools ───────────────────────────────────────────────────────────────


def grep_files(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    grep_context: int = 0,
    line: int | None = None,
    context: int = 80,
) -> str:
    """Search for a pattern in files on the filesystem using grep.

    Useful for files not tracked by git (build outputs, logs, traces, etc.).

    pattern: the search pattern (extended regex, grep -E)
    path: file or directory to search (default: current working directory)
    glob: if given, restrict to files matching this pattern (e.g. "*.py", "*.log")
    grep_context: lines of context around each match (grep -C)
    line: if given, centre the output on this 1-based line number and show `context` lines around it
    context: lines to show before and after `line` (default 80); when line is not given, limits total output lines
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
        from tools.git import trim_to_context

        return trim_to_context(out, line, context)
    lines = out.splitlines(keepends=True)
    if len(lines) > context:
        return (
            "".join(lines[:context])
            + f"\n[truncated — {len(lines) - context} more lines; use line= to navigate]"
        )
    return out


def list_dir(path: str = ".") -> str:
    """List the contents of a directory.

    Directories are shown with a trailing '/'. Entries are sorted
    alphabetically, directories first.

    path: directory path relative to the working directory (default: '.')
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
    """Edit a file by finding and replacing a specific block of code.

    Locates *search* in the file using fuzzy matching (handles minor whitespace
    differences) and writes the result. The user is asked to approve or reject
    before this tool runs (configured via interrupt_on in the agent).

    Guidance for writing a good search block:
    - Include 3-5 lines of unchanged surrounding context so the location is
      unambiguous, especially in large files.
    - Reproduce the indentation exactly as it appears in the file.
    - The block must be unique — if the same lines appear more than once,
      add more context to distinguish them.

    path:    absolute path or path relative to the working directory
    search:  the existing code to find (verbatim, with context lines)
    replace: the new code that replaces the search block exactly
    """
    file_path = Path(path).expanduser()
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
            "or verify the text against the actual file with read_around / git_show_file."
        )

    start, end = match
    modified = original[:start] + replace + original[end:]

    try:
        file_path.write_text(modified, encoding="utf-8")
        return f"Applied edit to {file_path}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Parent directories are created automatically if they do not exist.
    The user is asked to approve or reject before this tool runs
    (configured via interrupt_on in the agent).

    path:    absolute path or path relative to the working directory
    content: the full text content to write
    """
    file_path = Path(path) if Path(path).is_absolute() else Path(os.getcwd()) / path
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Written {file_path}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"
