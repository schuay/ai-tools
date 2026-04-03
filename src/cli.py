"""
Terminal CLI for the multi-agent assistant.

Uses prompt_toolkit for input (with patch_stdout so the prompt stays
visible while the agent streams output above it) and rich Console for
styled output.  All agent logic lives in session.py.
"""

import io
import sys
import threading
from dataclasses import dataclass
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
TOOL_COLLAPSE_LINES = 5  # lines shown when collapsed
TOOL_COLLAPSE_WIDTH = 80  # max chars per line when collapsed


class _TeeFile:
    """Wraps a file object and captures all writes into a StringIO buffer.

    Call snapshot(clear=True) to get the captured text and reset the buffer.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._buf = io.StringIO()

    def write(self, s: str) -> int:
        self._buf.write(s)
        return self._inner.write(s)

    def flush(self) -> None:
        self._inner.flush()

    def snapshot(self, *, clear: bool = False) -> str:
        """Return captured text. If clear, reset the buffer."""
        text = self._buf.getvalue()
        if clear:
            self._buf = io.StringIO()
        return text

    # Delegate everything else to inner file.
    def __getattr__(self, name: str):
        return getattr(self._inner, name)


# Disable the default bottom-toolbar reverse-video so our inline colors work.
_TOOLBAR_STYLE = Style.from_dict({"bottom-toolbar": "noreverse"})


def _make_key_bindings(session: Session, io: "TerminalIO") -> KeyBindings:
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

    @kb.add("c-o")  # Ctrl+O → toggle tool output collapse/expand
    def _toggle_tools(event):
        if session._waiting_for_input:
            io.toggle_tool_output()

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


@dataclass
class _ToolResult:
    """A captured tool result for collapse/expand replay."""

    output: str


class TerminalIO:
    """SessionIO implementation: rich Console for output, status via toolbar.

    Tool results are captured in a segment list so that Ctrl-O can toggle
    between collapsed and expanded views after a turn completes.
    """

    def __init__(self, console: Console, tee: _TeeFile) -> None:
        self._console = console
        self._tee = tee
        self._md = MarkdownRenderer(console.print)
        self._status = ""
        # Segment list for the current turn: interleaved ANSI text and tool results.
        self._segments: list[str | _ToolResult] = []
        self._collapsed = True

    def write(self, text: str, style: str | None = None) -> None:
        self._md.feed(text, style=style)

    def flush_markdown(self) -> None:
        self._md.flush()

    def set_status(self, text: str) -> None:
        self._status = text

    @property
    def status(self) -> str:
        return self._status

    # ── tool output collapse/expand ──────────────────────────────────────────

    def on_turn_start(self) -> None:
        """Reset segment tracking for the new turn."""
        self._md.flush()
        self._tee.snapshot(clear=True)
        self._segments.clear()

    def write_tool_result(self, output: str) -> None:
        """Render a tool result (collapsed by default) and record it for replay."""
        self._md.flush()
        # Snapshot any ANSI text printed since the last segment boundary.
        ansi = self._tee.snapshot(clear=True)
        if ansi:
            self._segments.append(ansi)
        tr = _ToolResult(output)
        self._segments.append(tr)
        # Render (collapsed or expanded) — then discard from the tee buffer
        # so the next snapshot starts fresh.
        self._render_tool(tr, self._collapsed)
        self._tee.snapshot(clear=True)

    def toggle_tool_output(self) -> None:
        """Ctrl-O handler: toggle collapsed state and redraw the current turn."""
        self._collapsed = not self._collapsed
        self._redraw()

    def _render_tool(self, tr: _ToolResult, collapsed: bool) -> None:
        """Print a tool result to the console, optionally truncated."""
        lines = tr.output.splitlines()
        if not lines:
            return
        if collapsed:
            limit = TOOL_COLLAPSE_LINES
            w = TOOL_COLLAPSE_WIDTH
            for line in lines[:limit]:
                truncated = (line[:w] + "…") if len(line) > w else line
                self._console.print(f"  {truncated}", style="dim")
            hidden = len(lines) - limit
            if hidden > 0:
                self._console.print(
                    f"  … ({hidden} more lines, Ctrl-O to expand)",
                    style="dim italic",
                )
        else:
            sep = "┄" * min(48, max(len(ln) for ln in lines[:80]))
            self._console.print(f"  {sep}", style="dim")
            for line in lines:
                self._console.print(f"  {line}", style="dim")
            self._console.print(f"  {sep}", style="dim")

    def _redraw(self) -> None:
        """Clear screen and replay all segments with the current collapse state."""
        self._md.flush()
        # Snapshot any trailing text since the last segment boundary.
        ansi = self._tee.snapshot(clear=True)
        if ansi:
            self._segments.append(ansi)
        # Clear screen.
        out = self._tee._inner
        out.write("\033[2J\033[H")
        out.flush()
        # Replay each segment, writing directly to inner (bypass tee capture).
        for seg in self._segments:
            if isinstance(seg, _ToolResult):
                # Render via Console (goes through tee), then discard from buffer.
                self._render_tool(seg, self._collapsed)
                self._tee.snapshot(clear=True)
            else:
                out.write(seg)
                out.flush()


def _make_toolbar(io: TerminalIO, session: Session):
    """Return a callable for prompt_toolkit's bottom_toolbar."""

    def _toolbar():
        status = io.status
        if not status:
            return HTML(
                '<style bg="#1a1a2e" fg="#555555">'
                " <b>Esc</b> interrupt  <b>Ctrl-O</b> toggle tools"
                "  <b>Alt+Enter</b> newline"
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
        tee = _TeeFile(sys.stdout)
        console = Console(file=tee, force_terminal=True, highlight=False)
        io = TerminalIO(console, tee)
        session = Session(io=io, prompt=prompt_text)

        worker = Thread(target=session.run, daemon=True)
        worker.start()

        ps = PromptSession(
            key_bindings=_make_key_bindings(session, io),
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
