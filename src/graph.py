"""Agent definition — used by both the CLI session and langgraph dev (Studio)."""

import copy

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
from langchain.chat_models import init_chat_model
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langgraph.types import interrupt

from deepagents.backends import FilesystemBackend

from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT, SubAgentMiddleware
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    compute_summarization_defaults,
)

from tools import REPO_ROOT, standard_tools

# ── tracing ──────────────────────────────────────────────────────────────────
#
# Enable with: agent --trace ...
# Logs to stderr: agent setup (system prompt + tools), model calls (messages).

import sys

TRACE = False  # set by cli.py from --trace flag


def enable_tracing() -> None:
    """Turn on tracing output to stderr."""
    global TRACE
    TRACE = True


def _trace(msg: str) -> None:
    """Write a trace line to stderr."""
    print(msg, file=sys.stderr)


def _trace_agent_setup(system_prompt: str, tools: list) -> None:
    """Log system prompt size and per-tool description/schema sizes."""
    _trace("\n=== AGENT SETUP ===")
    _trace(f"System prompt: {len(system_prompt)} chars")
    _trace(f"Tools: {len(tools)}")
    for t in tools:
        name = getattr(t, "name", "?")
        desc = getattr(t, "description", "") or ""
        schema = getattr(t, "args_schema", None)
        if schema is not None:
            schema_str = str(
                schema.model_json_schema()
                if hasattr(schema, "model_json_schema")
                else schema
            )
        else:
            schema_str = ""
        _trace(f"  {name}: desc={len(desc)} schema={len(schema_str)}")
    _trace("=== END AGENT SETUP ===\n")


def _trace_messages(messages) -> None:
    """Log each message role and size before a model call."""
    _trace("\n=== MODEL CALL ===")
    total_chars = 0
    for msg in messages:
        role = getattr(msg, "type", getattr(msg, "role", "?"))
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            chars = sum(len(str(p)) for p in content)
        else:
            chars = len(str(content))
        total_chars += chars
        preview = str(content)[:120].replace("\n", "\\n")
        _trace(f"  [{role}] {chars} chars: {preview}")
    _trace(f"Total: {total_chars} chars")
    _trace("=== END MODEL CALL ===\n")


_patched_classes: set[type] = set()


def _patch_model_class_for_tracing(model) -> None:
    """Monkey-patch the model class to log messages on _generate/_astream."""
    cls = type(model)
    if cls in _patched_classes:
        return
    _patched_classes.add(cls)

    orig_generate = cls._generate

    def traced_generate(self, messages, stop=None, run_manager=None, **kwargs):
        _trace_messages(messages)
        return orig_generate(
            self, messages, stop=stop, run_manager=run_manager, **kwargs
        )

    cls._generate = traced_generate

    if hasattr(cls, "_astream"):
        orig_astream = cls._astream

        async def traced_astream(self, messages, stop=None, run_manager=None, **kwargs):
            _trace_messages(messages)
            async for chunk in orig_astream(
                self, messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk

        cls._astream = traced_astream


# ── schema helpers ───────────────────────────────────────────────────────────


# Keys stripped from tool schemas to reduce token count and fix Gemini compat.
# "title" and "description" duplicate info already in the tool definition.
_STRIP_KEYS = {"additionalProperties", "$schema", "title", "description"}


def _clean_schema(schema: dict) -> None:
    """Recursively clean a tool schema: strip redundant keys, fix arrays."""
    if not isinstance(schema, dict):
        return
    for key in _STRIP_KEYS & schema.keys():
        del schema[key]
    if schema.get("type") == "array" and "items" not in schema:
        schema["items"] = {}
    for v in schema.values():
        if isinstance(v, dict):
            _clean_schema(v)
        elif isinstance(v, list):
            for item in v:
                _clean_schema(item)


def _fix_tool_schema(tool: StructuredTool) -> StructuredTool:
    """Patch a tool's args_schema for Gemini compatibility."""
    schema_cls = getattr(tool, "args_schema", None)
    if schema_cls is None:
        return tool
    if isinstance(schema_cls, dict):
        _clean_schema(schema_cls)
        return tool
    orig_fn = schema_cls.model_json_schema.__func__

    def patched(cls, **kwargs):
        s = copy.deepcopy(orig_fn(cls, **kwargs))
        _clean_schema(s)
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

v8_instructions = """You are a senior V8 JavaScript engine engineer with deep knowledge of
all major subsystems: parser/AST, Ignition bytecode, Maglev/TurboFan compilers, GC/heap,
inline caches/maps, WebAssembly tiers, embedder API, and build/test infrastructure.

## Working approach

Read before concluding. Use tools to examine actual code — diffs, context, blame — before
forming opinions. Explore systematically: follow callers, callees, types, and invariants.
Use git_blame to understand when/why something was introduced. Use web_search/web_fetch
for external references (bug IDs, blog posts, TC39 proposals).

Apply V8 expertise actively. Name patterns: IC miss, map transition, deopt bail-out,
write-barrier elision, escape analysis, safepoint, handle scope, etc. Surface non-obvious
invariants, threading constraints, and GC-safety requirements.

Use parallel tool calls when you need multiple independent pieces of information.

## Making changes

Read the target region first (read_around or git_show) and copy the search block
verbatim — include 3-5 lines of unchanged context to uniquely anchor the location.
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
    if TRACE:
        _patch_model_class_for_tracing(model)
    identity = _identity_section(name, agents) if name and agents else ""
    tools = standard_tools(fs=True)
    # ask_user uses LangGraph interrupt() — only meaningful in interactive sessions
    if interrupt_on:
        tools = tools + [ask_user]
    if extra_tools:
        tools = tools + extra_tools

    fixed_tools = []
    for t in tools:
        if callable(t) and not isinstance(t, StructuredTool):
            t = StructuredTool.from_function(t, handle_tool_error=True)
        elif hasattr(t, "handle_tool_error"):
            t.handle_tool_error = True
        fixed_tools.append(_fix_tool_schema(t))
    tools = fixed_tools

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

    def _middleware(extra: list | None = None) -> list:
        mw = [TodoListMiddleware()]
        if extra:
            mw.extend(extra)
        mw += [
            _summarization(),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
        if interrupt_on:
            mw.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
        return mw

    gp_subagent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools,
        "middleware": _middleware(),
    }

    middleware = _middleware(
        [SubAgentMiddleware(backend=backend, subagents=[gp_subagent])]
    )

    system_prompt = system_prompt or (
        identity + v8_instructions + "\n\n"
        "Be concise. Don't add preamble. Read first, then act, then verify. "
        "Keep working until done.\n"
    )

    if TRACE:
        _trace_agent_setup(system_prompt, tools)

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
