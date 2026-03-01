import json
import sys
import uuid

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import make_agent

# CLI needs its own checkpointer; langgraph dev provides one automatically
agent = make_agent(checkpointer=MemorySaver())

# ANSI colour helpers
_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_YELLOW = "\033[33m"


# ── prompt_toolkit input ────────────────────────────────────────────────────
# Meta+Enter (or Esc Enter) submits; Enter adds a newline inside the input.

_history = InMemoryHistory()
_kb = KeyBindings()


@_kb.add("enter")
def _newline(event):
    event.current_buffer.insert_text("\n")


@_kb.add("escape", "enter")  # Meta+Enter
def _submit(event):
    event.current_buffer.validate_and_handle()


def _prompt(label: str) -> str:
    return prompt(
        label,
        history=_history,
        key_bindings=_kb,
        multiline=True,
        prompt_continuation="  ",
    ).strip()


# ── HITL interrupt handler ──────────────────────────────────────────────────


def _handle_interrupt(value: object) -> Command:
    """Present an interrupt to the CLI user and return the Command to resume."""
    print()

    # ask_user tool: agent is asking a question
    if isinstance(value, dict) and "question" in value:
        print(f"{_YELLOW}{_BOLD}[agent asks]{_RESET} {value['question']}")
        answer = _prompt(HTML("<b>&gt; </b>"))
        return Command(resume=answer)

    # HumanInTheLoopMiddleware: approve / edit / reject a tool call
    if isinstance(value, dict) and "action_requests" in value:
        decisions = []
        for action_req, review_cfg in zip(
            value["action_requests"], value["review_configs"]
        ):
            name = action_req["name"]
            args = action_req["args"]
            allowed = review_cfg["allowed_decisions"]

            print(f"{_YELLOW}{_BOLD}[approve?]{_RESET} tool={_BOLD}{name}{_RESET}")
            print(f"{_DIM}{json.dumps(args, indent=2)}{_RESET}")
            opts = "/".join(allowed)

            while True:
                choice = _prompt(HTML(f"<b>[{opts}]&gt; </b>")).lower()
                if choice in allowed:
                    break
                print(f"  choose one of: {opts}")

            if choice == "approve":
                decisions.append({"type": "approve"})
            elif choice == "reject":
                msg = _prompt("  rejection message (optional): ")
                decisions.append(
                    {"type": "reject", **({"message": msg} if msg else {})}
                )
            elif choice == "edit":
                print(f"  current args: {json.dumps(args)}")
                raw = _prompt("  new args (JSON): ")
                try:
                    new_args = json.loads(raw)
                    decisions.append(
                        {
                            "type": "edit",
                            "edited_action": {"name": name, "args": new_args},
                        }
                    )
                except json.JSONDecodeError:
                    print("  invalid JSON — approving as-is")
                    decisions.append({"type": "approve"})

        return Command(resume={"decisions": decisions})

    # Fallback for unknown interrupt shapes
    print(f"{_YELLOW}[interrupt]{_RESET} {value!r}")
    _prompt("Press Meta+Enter to continue… ")
    return Command(resume=None)


# ── entry point ─────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python main.py <git-hash>", file=sys.stderr)
    sys.exit(1)

commit_hash = sys.argv[1]

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

current_input: dict | Command = {
    "messages": [
        {
            "role": "user",
            "content": f"Please explain git commit {commit_hash} from the v8 repository.",
        }
    ]
}

# ── streaming loop (resumes automatically after each interrupt) ─────────────

_current_block: str | None = None
_seen_tool_ids: set[str] = set()

while True:
    resume_command: Command | None = None

    try:
        for mode, data in agent.stream(
            current_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            # ── interrupt detected in the updates channel ──
            if mode == "updates" and "__interrupt__" in data:
                intr = data["__interrupt__"][0]
                _current_block = None
                resume_command = _handle_interrupt(intr.value)
                break

            if mode != "messages":
                continue

            chunk, _meta = data

            # ── tool call announced ──
            if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    tool_id = tc.get("id")
                    name = tc.get("name")
                    if name and tool_id and tool_id not in _seen_tool_ids:
                        _seen_tool_ids.add(tool_id)
                        if _current_block:
                            print()
                        print(f"{_DIM}[tool] {name}{_RESET}", flush=True)
                        _current_block = None
                continue

            # ── tool result ──
            if isinstance(chunk, ToolMessage):
                preview = str(chunk.content)[:120].replace("\n", " ")
                print(f"{_DIM}  → {preview}…{_RESET}", flush=True)
                continue

            if not isinstance(chunk, AIMessageChunk):
                continue

            content = chunk.content
            if not content:
                continue

            blocks = (
                content
                if isinstance(content, list)
                else [{"type": "text", "text": content}]
            )

            for block in blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")

                if btype == "thinking":
                    text = block.get("thinking", "")
                    if not text:
                        continue
                    if _current_block != "thinking":
                        print(f"\n{_DIM}[thinking]", flush=True)
                        _current_block = "thinking"
                    print(f"{_DIM}{text}{_RESET}", end="", flush=True)

                elif btype == "text":
                    text = block.get("text", "")
                    if not text:
                        continue
                    if _current_block == "thinking":
                        print(f"\n{_RESET}", end="", flush=True)
                    elif _current_block is None:
                        print()
                    _current_block = "text"
                    print(text, end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n{_YELLOW}[interrupted]{_RESET}")
        try:
            steering = _prompt(
                HTML("<b>steering (Meta+Enter to send, empty to quit)&gt; </b>")
            )
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
        if not steering:
            sys.exit(0)
        _current_block = None
        current_input = {"messages": [{"role": "user", "content": steering}]}
        continue

    if resume_command is None:
        break  # normal completion — no more interrupts
    current_input = resume_command

print()
