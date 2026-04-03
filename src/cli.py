"""
Terminal CLI for the multi-agent assistant.

Uses prompt_toolkit for input (with patch_stdout so the prompt stays
visible while the agent streams output above it) and rich Console for
styled output.  All agent logic lives in session.py.
"""

import sys
import threading
from threading import Thread

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console

from mdrender import MarkdownRenderer
from session import Session

PASTE_COLLAPSE_THRESHOLD = 3  # lines

# Disable the default bottom-toolbar reverse-video so our inline colors work.
_TOOLBAR_STYLE = Style.from_dict({"bottom-toolbar": "noreverse"})


def _make_key_bindings(session: Session) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter → newline
    def _newline_alt(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-j")  # Ctrl+J → newline (matches old TUI)
    def _newline_ctrl(event):
        event.current_buffer.insert_text("\n")

    @kb.add("escape")  # Escape → interrupt agent
    def _interrupt(event):
        session.interrupt()

    @kb.add("up")
    def _history_back(event):
        buf = event.current_buffer
        # On first line: navigate history. Otherwise: move cursor up.
        if buf.document.cursor_position_row == 0:
            buf.history_backward()
        else:
            buf.cursor_up()

    @kb.add("down")
    def _history_forward(event):
        buf = event.current_buffer
        # On last line: navigate history. Otherwise: move cursor down.
        if buf.document.cursor_position_row == buf.document.line_count - 1:
            buf.history_forward()
        else:
            buf.cursor_down()

    return kb


class TerminalIO:
    """SessionIO implementation: rich Console for output, status via toolbar."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._md = MarkdownRenderer(console.print)
        self._status = ""

    def write(self, text: str, style: str | None = None) -> None:
        self._md.feed(text, style=style)

    def flush_markdown(self) -> None:
        self._md.flush()

    def set_status(self, text: str) -> None:
        self._status = text

    @property
    def status(self) -> str:
        return self._status


def _make_toolbar(io: TerminalIO, session: Session):
    """Return a callable for prompt_toolkit's bottom_toolbar."""

    def _toolbar():
        status = io.status
        if not status:
            return HTML(
                '<style bg="#1a1a2e" fg="#555555">'
                " <b>Esc</b> interrupt  <b>Alt+Enter</b> newline"
                "</style>"
            )

        # Approval prompt — highlight the shortcuts
        if "approve" in status.lower() or "reject" in status.lower():
            return HTML(
                '<style bg="#1a1a2e">'
                " <style fg='#c3e88d'><b>Y</b>/Enter accept</style>"
                "  <b>N</b> reject"
                "  <b>E</b> edit"
                "</style>"
            )

        # Active status (routing, running, waiting)
        escaped = status.replace("&", "&amp;").replace("<", "&lt;")
        return HTML(f'<style bg="#1a1a2e" fg="#888888"> <i>{escaped}</i></style>')

    return _toolbar


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    import traceback

    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    sys.stderr.flush()


def main() -> None:
    threading.excepthook = _thread_excepthook
    args = sys.argv[1:]
    if "--trace" in args:
        args.remove("--trace")
        from graph import enable_tracing

        enable_tracing()
    prompt_text = " ".join(args)

    with patch_stdout(raw=True):
        console = Console(force_terminal=True, highlight=False)
        io = TerminalIO(console)
        session = Session(io=io, prompt=prompt_text)

        worker = Thread(target=session.run, daemon=True)
        worker.start()

        ps = PromptSession(
            key_bindings=_make_key_bindings(session),
            multiline=True,
            history=InMemoryHistory(),
            prompt_continuation="  ",
            bottom_toolbar=_make_toolbar(io, session),
            style=_TOOLBAR_STYLE,
        )

        while worker.is_alive():
            try:
                text = ps.prompt("> ").strip()
                if not text:
                    continue
                lines = text.splitlines()
                if len(lines) > PASTE_COLLAPSE_THRESHOLD:
                    io.write(f"> [{len(lines)} lines]", style="bold green")
                session.submit(text)
            except KeyboardInterrupt:
                continue  # Ctrl+C clears current input
            except EOFError:
                session.stop()
                break

        worker.join(timeout=2)


if __name__ == "__main__":
    main()
