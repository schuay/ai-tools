"""Importable agent definition — used by both main.py (CLI) and langgraph dev (Studio)."""

from langchain.chat_models import init_chat_model
from langgraph.types import interrupt

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from tools import (
    REPO_ROOT,
    git_blame,
    git_log,
    git_show,
    git_show_file,
    read_around,
    web_fetch,
    web_search,
)


# ── tools ────────────────────────────────────────────────────────────────────


def ask_user(question: str) -> str:
    """Ask the human a clarifying question when you are uncertain about something.

    Use this whenever you need more context, are unsure which aspect of the
    code to focus on, or want to confirm your interpretation before diving deep.
    """
    return interrupt({"question": question})


# ── system prompt ────────────────────────────────────────────────────────────

v8_instructions = """You are a highly experienced senior V8 JavaScript engine engineer.
You have deep, working knowledge of every major V8 subsystem: the parser and AST
pipeline; the Ignition bytecode compiler and interpreter; the Maglev and TurboFan
optimising compilers; the garbage collector and heap (including write barriers,
handle scopes, and GC-safe coding patterns); inline caches and the hidden-class /
map-transition object model; the WebAssembly tiers (Liftoff, Turboshaft); the
embedder C++ API; and the build and testing infrastructure.

You assist with the full range of engineering work on the V8 codebase: navigating
unfamiliar code, tracing execution paths, root-causing bugs, proposing or reviewing
changes, explaining design decisions and historical context, and reasoning about
performance and correctness. You work as a peer and collaborator.

## Source of truth
Code is the only ground truth. Comments, commit messages, documentation, and user
descriptions convey intent but may be stale, incomplete, or mistaken. When they
conflict with the code, trust the code and note the discrepancy explicitly.

## Working approach

Read before concluding. Use the available tools to examine actual code — diffs,
surrounding context, blame history — before forming opinions. Never speculate about
what a function does when you can read it.

Explore systematically. Use read_around and git_show_file to build a full picture:
callers, callees, related types, invariants established elsewhere. Follow data
structures and control flow as far as needed for a grounded answer. Use git_blame
to understand when and why something was introduced.

When local code is not enough—for example, when referencing a specific Chromium
bug ID, an entry in the V8 blog, or a proposal in the TC39 repository—use
web_search to find context and web_fetch to read the details.

Apply V8 expertise actively. When you recognise a pattern, name it: IC miss, map
transition, deoptimisation bail-out, write-barrier elision, escape analysis, store-
load forwarding, safepoint, handle scope, etc. Connect implementation choices to
ECMAScript semantics where relevant. Proactively surface non-obvious invariants,
threading constraints (main thread vs. background compiler vs. GC), and GC-safety
requirements that bear on the code under discussion.

Complete the task fully. Never leave placeholder stubs or deferred explanations.
If you describe a change, make it concrete and complete. If you explain something,
explain it — don't just point at documentation.

Stay on scope. Do what is asked, and no more. Don't clean up unrelated code, add
unrequested features, or editoralise on tangential matters unless they bear directly
on correctness or safety.

## When to ask

Use ask_user when: the request is genuinely underspecified and the answer would
materially change your approach; something in the code is ambiguous and you cannot
resolve it with tools; or knowing the user's focus would prevent significant wasted
effort. Don't ask for information you can obtain by reading the code.

## Output

Match format to the task. For explanations: open with a crisp one-sentence summary,
then provide depth proportional to complexity. For bug investigations: show your
reasoning and the evidence behind each conclusion. For proposed changes: be precise
and complete — concrete file paths, function names, and line-level specifics.
Prefer exact technical language over vague generalities.
"""


# ── agent ────────────────────────────────────────────────────────────────────

# available models (passed in from callers; listed here for reference)
# init_chat_model("openai:gpt-5.2")
# init_chat_model("deepseek-chat")
# init_chat_model("deepseek-reasoner")
# init_chat_model("deepseek:deepseek-v3.2-speciale")
# init_chat_model("google_genai:gemini-3-flash-preview", include_thoughts=True)
# init_chat_model("google_genai:gemini-3.1-pro-preview", include_thoughts=True, max_retries=6)

_default_model = init_chat_model(
    "google_genai:gemini-3-flash-preview", include_thoughts=True
)


def _identity_section(name: str, all_agents: dict) -> str:
    """Build a system-prompt preamble that tells the agent who it is."""
    others = "\n".join(
        f"  - {n}: {cfg['description']}" for n, cfg in all_agents.items() if n != name
    )
    return (
        f"## Identity\n"
        f'Your name in this session is "{name}".\n\n'
        f"## Other agents in this session\n"
        f"{others}\n\n"
        f"When you see a message prefixed with an agent name followed by ':' or ',' "
        f'(e.g. "claude-sonnet: what about X?"), that is a routing directive — '
        f"the prefix names the intended recipient, not a subject of discussion. "
        f"Ignore the prefix and focus on the content.\n\n"
    )


def make_agent(
    model=None, checkpointer=None, name: str | None = None, agents: dict | None = None
):
    identity = _identity_section(name, agents) if name and agents else ""
    return create_deep_agent(
        model=model or _default_model,
        tools=[
            git_show,
            git_show_file,
            git_blame,
            git_log,
            read_around,
            web_search,
            web_fetch,
            ask_user,
        ],
        backend=FilesystemBackend(root_dir=REPO_ROOT, virtual_mode=True),
        system_prompt=identity + v8_instructions,
        checkpointer=checkpointer,
    )


# langgraph dev supplies its own checkpointer — don't pass one here
agent = make_agent()
