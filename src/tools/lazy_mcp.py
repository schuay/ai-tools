"""Lazy MCP tool loading via search-based discovery.

Instead of exposing all MCP tool schemas to the LLM upfront, this module
provides a `search_mcp_tools` tool that lets the LLM discover tools on
demand.  A companion `LazyMCPMiddleware` filters `request.tools` in each
model call so that only previously-searched ("unlocked") MCP tools have
their schemas sent to the LLM.

Usage (inside _mcp_runner):

    search_tool, lazy_mw = make_lazy_mcp(mcp_tools, verbose=True)
    agent = make_agent(
        ...,
        extra_tools=[search_tool] + mcp_tools,   # all in ToolNode
        extra_middleware=[lazy_mw],               # filters request.tools
    )
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.tools import BaseTool

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_SEARCH_TOOL_NAME = "search_mcp_tools"

_TOOLS_HEADER = "TOOLS:"


def _format_tool(tool: BaseTool) -> str:
    """Format a single tool for the search result."""
    schema = tool.args_schema
    if isinstance(schema, dict):
        schema_str = json.dumps(schema, indent=2)
    elif schema is not None:
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
    else:
        schema_str = "{}"
    return f"### {tool.name}\n{tool.description}\n\nInput schema:\n```json\n{schema_str}\n```"


def _make_search_tool(
    mcp_tools: list[BaseTool],
    unlocked: set[str],
    verbose: bool,
) -> StructuredTool:
    """Create the search_mcp_tools tool as a closure over *mcp_tools* and *unlocked*."""

    def search_mcp_tools(query: str) -> str:
        """Search available MCP tools by name or keyword.

        Returns tool names, descriptions, and input schemas for matching tools.
        You must call this before using any MCP tool — tools only become
        available after being returned by this search.
        """
        q = query.lower()
        matched = [
            t
            for t in mcp_tools
            if q in t.name.lower() or q in (t.description or "").lower()
        ]
        if not matched:
            names = ", ".join(t.name for t in mcp_tools[:10])
            hint = f"No tools matched {query!r}. Available tools: {names}"
            if len(mcp_tools) > 10:
                hint += f", ... ({len(mcp_tools)} total)"
            if verbose:
                print(
                    f"[lazy_mcp] search query={query!r} → no matches", file=sys.stderr
                )
            return hint

        for t in matched:
            unlocked.add(t.name)

        if verbose:
            names = [t.name for t in matched]
            print(f"[lazy_mcp] search query={query!r} → {names}", file=sys.stderr)

        header = f"{_TOOLS_HEADER} {', '.join(t.name for t in matched)}"
        details = "\n\n".join(_format_tool(t) for t in matched)
        return f"{header}\n\n{details}"

    return StructuredTool.from_function(
        search_mcp_tools,
        name=_SEARCH_TOOL_NAME,
        handle_tool_error=True,
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@dataclass
class LazyMCPMiddleware(AgentMiddleware):
    """Filters MCP tool schemas out of model calls until they are unlocked.

    Tools are unlocked by calling ``search_mcp_tools``, which writes tool
    names into the shared *unlocked* set.  On each model call the middleware
    also scans message history for prior search results so that tools stay
    unlocked across turns.
    """

    mcp_tool_names: set[str] = field(default_factory=set)
    unlocked: set[str] = field(default_factory=set)
    verbose: bool = False

    # -- AgentMiddleware interface ------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._filter_tools(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._filter_tools(request))

    # -- internals ----------------------------------------------------------

    def _seed_from_history(self, messages: list) -> None:
        """Scan message history for prior search results and seed *unlocked*."""
        before = len(self.unlocked)
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            if msg.name != _SEARCH_TOOL_NAME:
                continue
            content = msg.content if isinstance(msg.content, str) else ""
            if not content.startswith(_TOOLS_HEADER):
                continue
            header = content.split("\n", 1)[0]
            names = [n.strip() for n in header[len(_TOOLS_HEADER) :].split(",")]
            self.unlocked.update(n for n in names if n in self.mcp_tool_names)
        if self.verbose:
            added = len(self.unlocked) - before
            if added:
                print(
                    f"[lazy_mcp] seeded {added} tool(s) from history: {self.unlocked}",
                    file=sys.stderr,
                )

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        """Return a new request with only unlocked MCP tools."""
        self._seed_from_history(request.messages)

        filtered = []
        for t in request.tools:
            name = getattr(t, "name", None)
            if name in self.mcp_tool_names:
                if name in self.unlocked:
                    filtered.append(t)
                # else: hide from LLM
            else:
                filtered.append(t)  # standard tool or search_tool: always include

        if self.verbose:
            mcp_count = sum(
                1 for t in filtered if getattr(t, "name", None) in self.mcp_tool_names
            )
            total = len(filtered)
            print(
                f"[lazy_mcp] presenting {total} tools ({mcp_count} MCP unlocked)",
                file=sys.stderr,
            )

        return request.override(tools=filtered)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_lazy_mcp(
    mcp_tools: list[BaseTool],
    *,
    verbose: bool = False,
) -> tuple[StructuredTool, LazyMCPMiddleware]:
    """Create a (search_tool, middleware) pair for lazy MCP tool loading.

    The caller should pass both to make_agent::

        search_tool, lazy_mw = make_lazy_mcp(mcp_tools)
        agent = make_agent(
            ...,
            extra_tools=[search_tool] + mcp_tools,
            extra_middleware=[lazy_mw],
        )

    All *mcp_tools* remain registered in the ToolNode for execution, but
    the middleware hides their schemas from the LLM until the search tool
    has returned them.
    """
    unlocked: set[str] = set()
    search_tool = _make_search_tool(mcp_tools, unlocked, verbose)
    middleware = LazyMCPMiddleware(
        mcp_tool_names={t.name for t in mcp_tools},
        unlocked=unlocked,
        verbose=verbose,
    )

    if verbose:
        print(
            f"[lazy_mcp] initialized with {len(mcp_tools)} MCP tools", file=sys.stderr
        )

    return search_tool, middleware
