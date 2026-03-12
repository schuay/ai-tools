"""Validate that tool schemas are well-formed for Gemini.

Catches two failure modes from the LangChain+Pydantic v1 ValidatedFunction bug:
  1. Spurious "v__args" field injected when a param is named "args" or "kwargs"
  2. Array fields with empty items ({}) which Gemini rejects as INVALID_ARGUMENT
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.tools import StructuredTool

from tools import standard_tools
from tools.shell import run_d8


def _get_all_schemas() -> list[tuple[str, dict]]:
    """Return (name, json_schema) for every standard tool + run_d8."""
    fns = standard_tools(web=True, git=True, fs=True, shell=True) + [run_d8]
    schemas = []
    for fn in fns:
        if callable(fn) and not isinstance(fn, StructuredTool):
            tool = StructuredTool.from_function(fn, handle_tool_error=True)
        else:
            tool = fn
        schema_cls = getattr(tool, "args_schema", None)
        if schema_cls is None:
            continue
        schema = (
            schema_cls
            if isinstance(schema_cls, dict)
            else schema_cls.model_json_schema()
        )
        schemas.append((tool.name, schema))
    return schemas


def _find_array_fields_without_items(schema: dict, path: str = "") -> list[str]:
    """Recursively find array-typed fields that are missing or have empty items."""
    bad: list[str] = []
    if not isinstance(schema, dict):
        return bad
    if schema.get("type") == "array" and not schema.get("items"):
        bad.append(path or "<root>")
    for key, value in schema.items():
        child_path = f"{path}.{key}" if path else key
        if isinstance(value, dict):
            bad.extend(_find_array_fields_without_items(value, child_path))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                bad.extend(_find_array_fields_without_items(item, f"{child_path}[{i}]"))
    return bad


@pytest.mark.parametrize("name,schema", _get_all_schemas())
def test_no_v__args_field(name: str, schema: dict) -> None:
    """No tool schema should contain the 'v__args' ghost field.

    This field is injected by Pydantic v1's ValidatedFunction when a function
    parameter is named 'args', shadowing the real parameter and corrupting the
    schema sent to LLMs.
    """
    props = schema.get("properties", {})
    assert "v__args" not in props, (
        f"Tool '{name}' has a spurious 'v__args' field — likely a parameter "
        f"named 'args' or 'kwargs'. Rename it to avoid the Pydantic v1 "
        f"ValidatedFunction sentinel collision. Full schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


@pytest.mark.parametrize("name,schema", _get_all_schemas())
def test_no_empty_array_items(name: str, schema: dict) -> None:
    """No array field in any tool schema should have missing or empty items.

    Gemini rejects such schemas with INVALID_ARGUMENT: missing field items.
    """
    bad = _find_array_fields_without_items(schema)
    assert not bad, (
        f"Tool '{name}' has array field(s) with missing/empty 'items': {bad}. "
        f"Gemini requires a non-empty items schema on all array types. "
        f"Full schema:\n{json.dumps(schema, indent=2)}"
    )
