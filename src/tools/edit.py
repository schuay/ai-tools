"""File editing tool: fuzzy search-and-replace with mandatory user approval.

Design notes
------------
The model supplies a *search* block (a few lines of context that uniquely
identify the location) and a *replace* block (what those lines should become).
Fuzzy matching handles the minor whitespace/indentation differences that LLMs
commonly introduce when reproducing code.

Matching is attempted in three passes (cheapest first):
  1. Exact substring match.
  2. Strip trailing whitespace from every line, then exact match.
  3. Sliding-window SequenceMatcher over line groups (threshold ≥ 0.85).

User approval is handled by deepagents' interrupt_on mechanism: the agent is
configured with interrupt_on={"file_edit": True}, so execution pauses before
this function is ever called and the user can approve or reject.
"""

import difflib
from pathlib import Path


# ── fuzzy matching ────────────────────────────────────────────────────────────


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
        ratio = difflib.SequenceMatcher(None, window, search_joined, autojunk=False).ratio()
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


def preview_diff(path: str, search: str, replace: str) -> str:
    """Unified diff for a fuzzy search-and-replace (file_edit / edit_file args)."""
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
    return _unified_diff(original, original[:start] + replace + original[end:], file_path.name)


def preview_write(path: str, content: str) -> str:
    """Unified diff for creating or overwriting a file (write_file args)."""
    file_path = Path(path).expanduser()
    original = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    return _unified_diff(original, content, file_path.name)


# ── tool ─────────────────────────────────────────────────────────────────────


def file_edit(path: str, search: str, replace: str) -> str:
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
