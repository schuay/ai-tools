"""Textual TUI for the LangGraph v8 commit explainer."""

import json
import sys
import uuid
from threading import Event

from rich.text import Text

from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog

from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import make_agent


class AgentApp(App):
    CSS = """
    RichLog {
        height: 1fr;
        border: none;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    Input {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("pageup", "page_up", "Scroll up"),
        ("pagedown", "page_down", "Scroll down"),
    ]

    def action_page_up(self) -> None:
        self.query_one(RichLog).scroll_page_up()

    def action_page_down(self) -> None:
        self.query_one(RichLog).scroll_page_down()

    def __init__(self, commit_hash: str):
        super().__init__()
        self.commit_hash = commit_hash
        self._input_ready = Event()
        self._user_input: str = ""
        self._accepting_input: bool = False
        self._steer_event = Event()
        self._steer_message: str = ""

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True, wrap=True)
        yield Input(placeholder="Agent is running…")

    def on_mount(self) -> None:
        self._agent = make_agent(checkpointer=MemorySaver())
        self._config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.query_one(Input).focus()
        self._do_write(f"> explain {self.commit_hash}", style="bold green")
        self.run_worker(self._run_agent, thread=True)

    # ── thread-safe UI helpers ───────────────────────────────────────────────

    def _do_write(self, text: str, style: str | None = None) -> None:
        log = self.query_one(RichLog)
        # only follow new output if already at the bottom
        log.auto_scroll = log.scroll_y >= log.max_scroll_y
        if style:
            log.write(Text(text, style=style))
        else:
            log.write(text)

    def _ui_write(self, text: str, style: str | None = None) -> None:
        self.call_from_thread(self._do_write, text, style)

    def _do_set_placeholder(self, text: str) -> None:
        self.query_one(Input).placeholder = text

    def _wait_input(self, placeholder: str = "> ") -> str:
        """Block the worker thread until the user presses Enter."""
        self._accepting_input = True
        self._input_ready.clear()
        self.call_from_thread(self._do_set_placeholder, placeholder)
        self._input_ready.wait()
        self._accepting_input = False
        return self._user_input

    # ── input handler ────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.clear()
        if not value:
            return
        self._do_write(f"> {value}", style="bold green")
        if self._accepting_input:
            # agent is waiting for our response to an interrupt / post-completion prompt
            self._user_input = value
            self._input_ready.set()
        else:
            # agent is mid-run — steer it
            self._steer_message = value
            self._steer_event.set()

    # ── interrupt handler ────────────────────────────────────────────────────

    def _handle_interrupt(self, value: object) -> Command:
        if isinstance(value, dict) and "question" in value:
            self._ui_write(f"[agent asks] {value['question']}", style="bold yellow")
            answer = self._wait_input("> ")
            return Command(resume=answer)

        if isinstance(value, dict) and "action_requests" in value:
            decisions = []
            for action_req, review_cfg in zip(
                value["action_requests"], value["review_configs"]
            ):
                name = action_req["name"]
                args = action_req["args"]
                allowed = review_cfg["allowed_decisions"]
                opts = "/".join(allowed)
                self._ui_write(f"[approve?] tool={name}", style="bold yellow")
                self._ui_write(json.dumps(args, indent=2), style="dim")
                while True:
                    choice = self._wait_input(f"[{opts}]> ").lower()
                    if choice in allowed:
                        break
                    self._ui_write(f"  choose one of: {opts}")
                if choice == "approve":
                    decisions.append({"type": "approve"})
                elif choice == "reject":
                    msg = self._wait_input("rejection message (optional): ")
                    decisions.append({"type": "reject", **({"message": msg} if msg else {})})
                elif choice == "edit":
                    self._ui_write(f"  current args: {json.dumps(args)}")
                    raw = self._wait_input("new args (JSON): ")
                    try:
                        new_args = json.loads(raw)
                        decisions.append({"type": "edit", "edited_action": {"name": name, "args": new_args}})
                    except json.JSONDecodeError:
                        self._ui_write("  invalid JSON — approving as-is")
                        decisions.append({"type": "approve"})
            return Command(resume={"decisions": decisions})

        self._ui_write(f"[interrupt] {value!r}", style="yellow")
        self._wait_input("Press Enter to continue… ")
        return Command(resume=None)

    # ── agent worker ─────────────────────────────────────────────────────────

    def _run_agent(self) -> None:
        current_input: dict | Command = {
            "messages": [{
                "role": "user",
                "content": f"Please explain git commit {self.commit_hash} from the v8 repository.",
            }]
        }
        _current_block: str | None = None
        _seen_tool_ids: set[str] = set()
        _text_buf: str = ""
        _text_style: str | None = None

        def flush_buf() -> None:
            nonlocal _text_buf
            if _text_buf:
                self._ui_write(_text_buf, style=_text_style)
                _text_buf = ""

        def write_streaming(text: str, style: str | None = None) -> None:
            nonlocal _text_buf, _text_style
            _text_style = style
            _text_buf += text
            while "\n" in _text_buf:
                line, _text_buf = _text_buf.split("\n", 1)
                self._ui_write(line, style=style)

        while True:
            resume_command: Command | None = None
            steered = False
            self._steer_event.clear()

            for mode, data in self._agent.stream(
                current_input,
                config=self._config,
                stream_mode=["messages", "updates"],
            ):
                # check for user steering after every chunk
                if self._steer_event.is_set():
                    flush_buf()
                    _current_block = None
                    steering = self._steer_message
                    self._ui_write(f"[steered] {steering}", style="bold cyan")
                    current_input = {"messages": [{"role": "user", "content": steering}]}
                    steered = True
                    break

                if mode == "updates" and "__interrupt__" in data:
                    flush_buf()
                    _current_block = None
                    resume_command = self._handle_interrupt(data["__interrupt__"][0].value)
                    break

                if mode != "messages":
                    continue

                chunk, _meta = data

                if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        tool_id = tc.get("id")
                        name = tc.get("name")
                        if name and tool_id and tool_id not in _seen_tool_ids:
                            _seen_tool_ids.add(tool_id)
                            flush_buf()
                            self._ui_write(f"[tool] {name}", style="dim")
                            _current_block = None
                    continue

                if isinstance(chunk, ToolMessage):
                    flush_buf()
                    preview = str(chunk.content)[:120].replace("\n", " ")
                    self._ui_write(f"  → {preview}…", style="dim")
                    continue

                if not isinstance(chunk, AIMessageChunk):
                    continue

                content = chunk.content
                if not content:
                    continue

                blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]

                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")

                    if btype == "thinking":
                        text = block.get("thinking", "")
                        if not text:
                            continue
                        if _current_block != "thinking":
                            flush_buf()
                            self._ui_write("[thinking]", style="dim")
                            _current_block = "thinking"
                        write_streaming(text, style="dim")

                    elif btype == "text":
                        text = block.get("text", "")
                        if not text:
                            continue
                        if _current_block == "thinking":
                            flush_buf()
                        _current_block = "text"
                        write_streaming(text)

            flush_buf()

            if steered:
                continue  # current_input already updated above

            if resume_command is not None:
                current_input = resume_command
                continue

            # normal completion — wait for the next user message
            current_input = {
                "messages": [{"role": "user", "content": self._wait_input("> ")}]
            }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tui.py <git-hash>", file=sys.stderr)
        sys.exit(1)

    AgentApp(sys.argv[1]).run()
