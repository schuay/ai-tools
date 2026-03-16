"""Tests for the streaming markdown renderer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mdrender import MarkdownRenderer


class FakeConsole:
    """Captures print calls for assertions."""

    def __init__(self):
        self.calls: list[tuple] = []

    def print(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    @property
    def texts(self) -> list[str]:
        """Return the first positional arg of each call as a string."""
        return [str(a[0]) if a else "" for a, _kw in self.calls]


def _make():
    c = FakeConsole()
    r = MarkdownRenderer(c.print)
    return r, c


# ── inline formatting ────────────────────────────────────────────────────────


def test_bold():
    r, c = _make()
    r.feed("hello **world**")
    assert len(c.calls) == 1
    text = c.calls[0][0][0]
    assert "[bold]world[/bold]" in text
    assert c.calls[0][1].get("markup") is True


def test_italic():
    r, c = _make()
    r.feed("hello *world*")
    text = c.calls[0][0][0]
    assert "[italic]world[/italic]" in text


def test_inline_code():
    r, c = _make()
    r.feed("use `foo()` here")
    text = c.calls[0][0][0]
    assert "[bold cyan]foo()[/bold cyan]" in text


def test_bold_italic():
    r, c = _make()
    r.feed("***emphasis***")
    text = c.calls[0][0][0]
    assert "[bold italic]emphasis[/bold italic]" in text


# ── Rich markup escaping ─────────────────────────────────────────────────────


def test_escapes_rich_markup():
    r, c = _make()
    r.feed("list[0] and [red] text")
    text = c.calls[0][0][0]
    # Square brackets should be escaped to prevent Rich interpretation
    assert r"\[0]" in text
    assert r"\[red]" in text


# ── styled passthrough ───────────────────────────────────────────────────────


def test_styled_bypasses_markdown():
    r, c = _make()
    r.feed("**bold** text", style="dim italic")
    assert len(c.calls) == 1
    text = c.calls[0][0][0]
    # Should NOT convert markdown when style is set
    assert "**bold** text" == text
    assert c.calls[0][1].get("markup") is False


# ── code blocks ──────────────────────────────────────────────────────────────


def test_code_block():
    r, c = _make()
    r.feed("```python")
    r.feed("x = 42")
    r.feed("```")
    assert len(c.calls) == 1
    # Should be a Markdown object (rendered by Rich)
    from rich.markdown import Markdown

    assert isinstance(c.calls[0][0][0], Markdown)


def test_code_block_flush_unterminated():
    r, c = _make()
    r.feed("```")
    r.feed("some code")
    # No closing fence — flush should still emit
    r.flush()
    assert len(c.calls) == 1
    from rich.markdown import Markdown

    assert isinstance(c.calls[0][0][0], Markdown)


# ── headers ──────────────────────────────────────────────────────────────────


def test_header():
    r, c = _make()
    r.feed("## Hello")
    assert len(c.calls) == 1
    from rich.markdown import Markdown

    assert isinstance(c.calls[0][0][0], Markdown)


# ── lists ────────────────────────────────────────────────────────────────────


def test_list_accumulation():
    r, c = _make()
    r.feed("- item one")
    r.feed("- item two")
    r.feed("- item three")
    # Still accumulating — nothing emitted yet
    assert len(c.calls) == 0
    # Non-list line flushes the list
    r.feed("regular text")
    assert len(c.calls) == 2  # list block + regular text
    from rich.markdown import Markdown

    assert isinstance(c.calls[0][0][0], Markdown)


def test_numbered_list():
    r, c = _make()
    r.feed("1. first")
    r.feed("2. second")
    r.flush()
    assert len(c.calls) == 1
    from rich.markdown import Markdown

    assert isinstance(c.calls[0][0][0], Markdown)


# ── empty lines ──────────────────────────────────────────────────────────────


def test_empty_line_passes_through():
    r, c = _make()
    r.feed("")
    assert len(c.calls) == 1
    assert c.calls[0][0][0] == ""


# ── styled flush clears block state ─────────────────────────────────────────


def test_styled_line_flushes_pending_list():
    r, c = _make()
    r.feed("- item one")
    r.feed("- item two")
    # A styled line should flush the pending list first
    r.feed("tool output", style="dim")
    from rich.markdown import Markdown

    assert isinstance(c.calls[0][0][0], Markdown)  # the flushed list
    assert c.calls[1][0][0] == "tool output"
