"""
Terminal CLI for the multi-agent assistant.

Uses prompt_toolkit for input and rich Console for styled output.
All agent logic lives in session.py.
"""

import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.text import Text

from session import Session

PASTE_COLLAPSE_THRESHOLD = 3  # lines


def _make_key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter → newline
    def _newline(event):
        event.current_buffer.insert_text("\n")

    return kb


class TerminalIO:
    """SessionIO implementation: rich Console for output, prompt_toolkit for input."""

    def __init__(self) -> None:
        self._console = Console(highlight=False)
        self._prompt_session = PromptSession(
            key_bindings=_make_key_bindings(),
            multiline=True,
            history=InMemoryHistory(),
            prompt_continuation="  ",
        )

    def write(self, text: str, style: str | None = None) -> None:
        if style:
            self._console.print(Text(text, style=style), highlight=False)
        else:
            self._console.print(text, highlight=False)

    def set_status(self, text: str) -> None:
        pass  # no status bar in terminal mode

    def get_input(self, prompt: str = "> ") -> str:
        """Block until the user submits non-empty input."""
        while True:
            try:
                text = self._prompt_session.prompt(
                    HTML(f"<b>{prompt}</b>"),
                )
                text = text.strip()
                if not text:
                    continue
                lines = text.splitlines()
                if len(lines) > PASTE_COLLAPSE_THRESHOLD:
                    first = lines[0][:60]
                    self.write(f"> {first}… [{len(lines)} lines]", style="bold green")
                else:
                    self.write(f"> {text}", style="bold green")
                return text
            except KeyboardInterrupt:
                continue  # Ctrl+C clears current input
            except EOFError:
                raise SystemExit(0)  # Ctrl+D exits


def main() -> None:
    prompt = " ".join(sys.argv[1:])
    io = TerminalIO()
    session = Session(io=io, prompt=prompt)
    try:
        session.run()
    except (SystemExit, KeyboardInterrupt):
        session.stop()


if __name__ == "__main__":
    main()
