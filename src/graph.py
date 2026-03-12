"""Agent definition — used by both the CLI session and langgraph dev (Studio)."""

import copy
import os

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
from langchain.chat_models import init_chat_model
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langgraph.types import interrupt

from deepagents.backends import FilesystemBackend
from deepagents.graph import BASE_AGENT_PROMPT
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT, SubAgentMiddleware
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    compute_summarization_defaults,
)

from tools import REPO_ROOT, standard_tools


# ── schema helpers ───────────────────────────────────────────────────────────


def _add_items_to_arrays(schema: dict) -> None:
    """Recursively add items:{} to array types missing it (required by Gemini)."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "array" and "items" not in schema:
        schema["items"] = {}
    for v in schema.values():
        if isinstance(v, dict):
            _add_items_to_arrays(v)
        elif isinstance(v, list):
            for item in v:
                _add_items_to_arrays(item)


def _fix_tool_schema(tool: StructuredTool) -> StructuredTool:
    """Patch a tool's args_schema to add missing array items fields (Gemini)."""
    schema_cls = getattr(tool, "args_schema", None)
    if schema_cls is None:
        return tool
    orig_fn = schema_cls.model_json_schema.__func__

    def patched(cls, **kwargs):
        s = copy.deepcopy(orig_fn(cls, **kwargs))
        _add_items_to_arrays(s)
        return s

    schema_cls.model_json_schema = classmethod(patched)
    return tool


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

Explore systematically. Use read_around and git_show to build a full picture:
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

## Making changes

You can propose edits to files using the edit_file tool. Always read the
target region first (read_around or git_show) and copy the search block
verbatim — include 3-5 lines of unchanged context on each side to uniquely
anchor the location. The user will review a diff and approve or reject before
any change is written.

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
        f'(e.g. "sonnet: what about X?"), that is a routing directive — '
        f"the prefix names the intended recipient, not a subject of discussion. "
        f"Ignore the prefix and focus on the content.\n\n"
    )


def make_agent(
    model=None,
    checkpointer=None,
    name: str | None = None,
    agents: dict | None = None,
    interrupt_on: dict | None = None,
    extra_tools: list | None = None,
    system_prompt: str | None = None,
):
    model = model or _default_model
    identity = _identity_section(name, agents) if name and agents else ""
    tools = standard_tools(fs=True)
    # ask_user uses LangGraph interrupt() — only meaningful in interactive sessions
    if interrupt_on:
        tools = tools + [ask_user]
    if extra_tools:
        tools = tools + extra_tools

    tools = [
        _fix_tool_schema(
            StructuredTool.from_function(t, handle_tool_error=True)
            if callable(t) and not isinstance(t, StructuredTool)
            else t
        )
        for t in tools
    ]

    # FilesystemBackend is used by SummarizationMiddleware for history storage only —
    # no filesystem tools are exposed to agents (FilesystemMiddleware is intentionally absent).
    backend = FilesystemBackend(root_dir=REPO_ROOT, virtual_mode=True)
    summarization_defaults = compute_summarization_defaults(model)

    def _summarization() -> SummarizationMiddleware:
        return SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=summarization_defaults["trigger"],
            keep=summarization_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summarization_defaults["truncate_args_settings"],
        )

    # ── GP subagent middleware (no FilesystemMiddleware) ──────────────────────
    gp_middleware = [
        TodoListMiddleware(),
        _summarization(),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if interrupt_on:
        gp_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    gp_subagent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools,
        "middleware": gp_middleware,
    }

    # ── Main agent middleware (no FilesystemMiddleware) ───────────────────────
    middleware = [
        TodoListMiddleware(),
        SubAgentMiddleware(backend=backend, subagents=[gp_subagent]),
        _summarization(),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if interrupt_on:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    system_prompt = system_prompt or (
        identity + v8_instructions + "\n\n" + BASE_AGENT_PROMPT
    )

    return create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
        name=name,
    ).with_config({"recursion_limit": 1000})


# langgraph dev supplies its own checkpointer — don't pass one here
agent = make_agent()
