"""
Textual TUI for the LangGraph v8 commit explainer.

Pure UI layer — no LangGraph imports. All agent logic lives in session.py.
Implements the SessionIO protocol so the Session can write output and set
the input placeholder from its worker thread.
"""

import sys

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog

from session import Session


class AgentApp(App):
    CSS = """
    RichLog {
        height: 1fr;
        border: none;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    Input { dock: bottom; }
    """

    BINDINGS = [
        ("pageup", "page_up", "Scroll up"),
        ("pagedown", "page_down", "Scroll down"),
    ]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True, wrap=True)
        yield Input(placeholder="Agent is running…")

    def on_mount(self) -> None:
        self._session = Session(io=self, prompt=self._prompt)
        self.query_one(Input).focus()
        self._append(f"> {self._prompt}", style="bold green")
        self.run_worker(self._session.run, thread=True)

    def on_unmount(self) -> None:
        self._session.stop()

    # ── SessionIO ────────────────────────────────────────────────────────────
    # These are called from the session's worker thread.

    def write(self, text: str, style: str | None = None) -> None:
        self.call_from_thread(self._append, text, style)

    def set_status(self, text: str) -> None:
        self.call_from_thread(self._set_placeholder, text)

    # ── internal UI helpers (main thread only) ───────────────────────────────

    def _append(self, text: str, style: str | None = None) -> None:
        log = self.query_one(RichLog)
        log.auto_scroll = log.scroll_y >= log.max_scroll_y
        log.write(Text(text, style=style) if style else text)

    def _set_placeholder(self, text: str) -> None:
        self.query_one(Input).placeholder = text

    # ── input ────────────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.clear()
        if not value:
            return
        self._append(f"> {value}", style="bold green")
        self._session.submit(value)

    # ── key bindings ─────────────────────────────────────────────────────────

    def action_page_up(self) -> None:
        self.query_one(RichLog).scroll_page_up()

    def action_page_down(self) -> None:
        self.query_one(RichLog).scroll_page_down()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tui.py <prompt>", file=sys.stderr)
        sys.exit(1)
    AgentApp(" ".join(sys.argv[1:])).run()
