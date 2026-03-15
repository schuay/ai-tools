"""Shared text utilities for output capping and context extraction."""

from pathlib import Path

CHARS_PER_LINE = 500  # char cap triggers when average output line exceeds this


def cap_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n[truncated — output exceeded {limit:,} chars; use line= to navigate]"
    )


def trim_to_context(full_text: str, line: int | None, context: int = 20) -> str:
    if line is None:
        return full_text
    lines = full_text.splitlines(keepends=True)
    start = max(0, line - 1 - context)
    end = min(len(lines), line - 1 + context + 1)
    result = "".join(
        f"{i + 1:>6}  {'>>>' if i + 1 == line else '   '}  {lines[i]}"
        for i in range(start, end)
    )
    return cap_chars(result, context * CHARS_PER_LINE)


def cap_lines(text: str, max_lines: int) -> str:
    """Truncate text to max_lines, with a message if truncated."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return cap_chars(text, max_lines * CHARS_PER_LINE)
    return cap_chars(
        "".join(lines[:max_lines])
        + f"\n[truncated — {len(lines) - max_lines} more lines; use line= to navigate]",
        max_lines * CHARS_PER_LINE,
    )


def resolve_path(path: str) -> Path:
    """Resolve a user-supplied path: expand ~ and make absolute."""
    import os

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    return p
