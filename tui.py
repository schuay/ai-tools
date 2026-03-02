"""
Textual TUI for the LangGraph v8 commit explainer.

Pure UI layer — no LangGraph imports. All agent logic lives in session.py.
Implements the SessionIO protocol so the Session can write output and set
the input placeholder from its worker thread.
"""

import sys

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import RichLog, TextArea

from session import Session


class _InputArea(TextArea):
    """Compact input box: starts 1 line tall and grows with content.

    Textual walks the full MRO when dispatching _on_key, so a bare `return`
    in our override is not enough — TextArea._on_key would still run and
    insert a newline. Instead, for Enter we call prevent_default() (which
    sets _no_default_action, causing _get_dispatch_methods to break before
    reaching TextArea) and stop() (prevents Key from bubbling), then post a
    Submitted message that the App handles cleanly.
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            value = self.text.strip()
            if value:
                self.post_message(self.Submitted(value))
            return
        if event.key == "ctrl+j":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)

    def on_text_area_changed(self, _: TextArea.Changed) -> None:
        # +2 for the top and bottom border rows (border: tall)
        num_lines = self.text.count("\n") + 1
        self.styles.height = min(num_lines + 2, 12)


class AgentApp(App):
    CSS = """
    RichLog {
        height: 1fr;
        border: tall $primary;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    _InputArea {
        dock: bottom;
        height: 3;
        max-height: 12;
        border: tall $primary;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("pageup", "page_up", "Scroll up"),
        ("pagedown", "page_down", "Scroll down"),
    ]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt
        self._history: list[str] = []
        self._history_index: int = -1
        self._draft: str = ""

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True, wrap=True)
        yield _InputArea()

    def on_mount(self) -> None:
        self._session = Session(io=self, prompt=self._prompt)
        self.query_one(_InputArea).focus()
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

    # ── internal helpers (main thread only) ──────────────────────────────────

    def _append(self, text: str, style: str | None = None) -> None:
        log = self.query_one(RichLog)
        log.auto_scroll = log.scroll_y >= log.max_scroll_y
        log.write(Text(text, style=style) if style else text)

    def _set_placeholder(self, text: str) -> None:
        self.query_one(_InputArea).placeholder = text

    # ── history ──────────────────────────────────────────────────────────────

    def _navigate_history(self, direction: int) -> None:
        """direction=+1: older entry (Up), direction=-1: newer entry (Down)."""
        if not self._history:
            return
        ta = self.query_one(_InputArea)
        if self._history_index == -1:
            self._draft = ta.text
        new_index = self._history_index + direction
        if 0 <= new_index < len(self._history):
            self._history_index = new_index
            ta.load_text(self._history[-(new_index + 1)])
        elif new_index == -1:
            self._history_index = -1
            ta.load_text(self._draft)
        else:
            return
        ta.move_cursor((ta.document.line_count, 0))

    # ── submit ────────────────────────────────────────────────────────────────

    @on(_InputArea.Submitted)
    def _on_submitted(self, event: _InputArea.Submitted) -> None:
        ta = self.query_one(_InputArea)
        ta.load_text("")
        self._history.append(event.value)
        self._history_index = -1
        self._draft = ""
        self._append(f"> {event.value}", style="bold green")
        self._session.submit(event.value)

    # ── key handling ─────────────────────────────────────────────────────────

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.prevent_default()
            self._session.interrupt()
            return
        ta = self.query_one(_InputArea)
        if event.key == "up" and ta.cursor_location == (0, 0):
            event.prevent_default()
            self._navigate_history(1)
        elif event.key == "down":
            last_line = ta.document.line_count - 1
            last_col = len(ta.document.get_line(last_line))
            if ta.cursor_location == (last_line, last_col):
                event.prevent_default()
                self._navigate_history(-1)

    def action_page_up(self) -> None:
        self.query_one(RichLog).scroll_page_up()

    def action_page_down(self) -> None:
        self.query_one(RichLog).scroll_page_down()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tui.py <prompt>", file=sys.stderr)
        sys.exit(1)
    AgentApp(" ".join(sys.argv[1:])).run()
