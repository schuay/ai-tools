"""
Textual TUI for the LangGraph v8 commit explainer.

Pure UI layer — no LangGraph imports. All agent logic lives in session.py.
Implements the SessionIO protocol so the Session can write output and set
the input placeholder from its worker thread.
"""

import sys

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import RichLog, TextArea

from session import Session


class AgentApp(App):
    CSS = """
    RichLog {
        height: 1fr;
        border: tall $primary;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    TextArea {
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
        # Using TextArea for multiline support
        yield TextArea(placeholder="Agent is running…")

    def on_mount(self) -> None:
        self._session = Session(io=self, prompt=self._prompt)
        self.query_one(TextArea).focus()
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
        self.query_one(TextArea).placeholder = text

    # ── input and history ───────────────────────────────────────────────────

    def _update_height(self, text: str) -> None:
        """Update textarea height based on number of lines.

        +2 accounts for the top and bottom border rows (border: tall).
        """
        num_lines = text.count("\n") + 1
        self.query_one(TextArea).styles.height = min(num_lines + 2, 12)

    def _handle_history(self, direction: int) -> None:
        """Cycle through command history."""
        if not self._history:
            return

        textarea = self.query_one(TextArea)
        # If moving from current draft, save it
        if self._history_index == -1:
            self._draft = textarea.text

        new_index = self._history_index + direction
        if 0 <= new_index < len(self._history):
            self._history_index = new_index
            textarea.load_text(self._history[-(new_index + 1)])
            textarea.move_cursor((textarea.document.line_count, 0))
            self._update_height(textarea.text)
        elif new_index == -1:
            self._history_index = -1
            textarea.load_text(self._draft)
            textarea.move_cursor((textarea.document.line_count, 0))
            self._update_height(textarea.text)

    def _submit(self) -> None:
        """Submit current text and reset input."""
        textarea = self.query_one(TextArea)
        value = textarea.text.strip()
        if not value:
            return

        # Clear and reset
        textarea.load_text("")
        self._update_height("")
        self._history.append(value)
        self._history_index = -1
        self._draft = ""

        # Output and session logic
        self._append(f"> {value}", style="bold green")
        self._session.submit(value)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_height(event.text_area.text)

    def on_key(self, event: events.Key) -> None:
        textarea = self.query_one(TextArea)

        if event.key == "escape":
            event.prevent_default()
            self._session.interrupt()
        elif event.key == "enter":
            # Plain enter submits
            event.prevent_default()
            self._submit()
        elif event.key == "ctrl+j":
            # Ctrl-J inserts newline
            event.prevent_default()
            textarea.insert("\n")
        elif event.key == "up":
            # History up if at the very first line/position
            if textarea.cursor_location == (0, 0):
                event.prevent_default()
                self._handle_history(1)
        elif event.key == "down":
            # History down if at the very last line/position
            last_line = textarea.document.line_count - 1
            last_col = len(textarea.document.get_line(last_line))
            if textarea.cursor_location == (last_line, last_col):
                event.prevent_default()
                self._handle_history(-1)

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
