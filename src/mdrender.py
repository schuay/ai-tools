"""Streaming markdown renderer for Rich Console output."""

import re
from enum import Enum, auto
from typing import Any, Callable

from rich.markdown import Markdown

# Inline patterns (order matters: bold+italic before bold before italic)
_BOLD_ITALIC = re.compile(r"\*\*\*(.+?)\*\*\*")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_INLINE_CODE = re.compile(r"`([^`]+)`")

_CODE_FENCE = re.compile(r"^(`{3,})(\w*)\s*$")
_LIST_ITEM = re.compile(r"^(\s*[-*]|\s*\d+\.)\s")
_HEADER = re.compile(r"^#{1,6}\s")


class _State(Enum):
    NORMAL = auto()
    CODE_BLOCK = auto()
    LIST = auto()


class MarkdownRenderer:
    """Converts streaming markdown lines to Rich-formatted terminal output.

    Call feed() with each complete line.  Call flush() at end-of-turn to emit
    any accumulated block content (e.g. an unterminated code block).
    """

    def __init__(self, print_fn: Callable[..., Any]) -> None:
        self._print = print_fn
        self._state = _State.NORMAL
        self._block: list[str] = []
        self._fence = ""  # the opening fence string (``` or more)

    def feed(self, line: str, style: str | None = None) -> None:
        """Process one line.  Styled lines bypass markdown rendering."""
        if style is not None:
            self._flush_block()
            self._print(line, style=style, highlight=False, markup=False)
            return

        if self._state is _State.CODE_BLOCK:
            self._handle_code(line)
        elif self._state is _State.LIST:
            self._handle_list(line)
        else:
            self._handle_normal(line)

    def flush(self) -> None:
        self._flush_block()

    # -- state handlers -------------------------------------------------------

    def _handle_normal(self, line: str) -> None:
        m = _CODE_FENCE.match(line)
        if m:
            self._state = _State.CODE_BLOCK
            self._fence = m.group(1)
            self._block = [line]
            return
        if _LIST_ITEM.match(line):
            self._state = _State.LIST
            self._block = [line]
            return
        if _HEADER.match(line):
            self._print(Markdown(line))
            return
        self._emit_inline(line)

    def _handle_code(self, line: str) -> None:
        self._block.append(line)
        # Close when we see a fence of equal or greater length
        if line.strip().startswith("`") and line.strip() == self._fence[0] * len(
            line.strip()
        ):
            fence_len = len(line.strip())
            if fence_len >= len(self._fence) and all(c == "`" for c in line.strip()):
                self._print(Markdown("\n".join(self._block)))
                self._block.clear()
                self._state = _State.NORMAL

    def _handle_list(self, line: str) -> None:
        if _LIST_ITEM.match(line) or line.strip() == "":
            self._block.append(line)
        else:
            self._flush_block()
            self._handle_normal(line)

    def _flush_block(self) -> None:
        if not self._block:
            return
        if self._state is _State.CODE_BLOCK:
            # Unterminated code block — close it so Markdown renders properly
            self._block.append(self._fence)
        self._print(Markdown("\n".join(self._block)))
        self._block.clear()
        self._state = _State.NORMAL

    # -- inline formatting ----------------------------------------------------

    def _emit_inline(self, line: str) -> None:
        escaped = line.replace("[", r"\[")
        escaped = _BOLD_ITALIC.sub(r"[bold italic]\1[/bold italic]", escaped)
        escaped = _BOLD.sub(r"[bold]\1[/bold]", escaped)
        escaped = _ITALIC.sub(r"[italic]\1[/italic]", escaped)
        escaped = _INLINE_CODE.sub(r"[bold cyan]\1[/bold cyan]", escaped)
        self._print(escaped, markup=True, highlight=False)
