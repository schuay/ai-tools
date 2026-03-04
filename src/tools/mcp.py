"""MCP config loader."""

import json
from pathlib import Path

_CONFIG_PATH = Path("~/.config/ai-tools/mcp.json").expanduser()


def load_config() -> dict:
    """Load MCP server config from ~/.config/ai-tools/mcp.json.

    Returns a dict suitable for passing to MultiServerMCPClient, or an empty
    dict if the file does not exist.
    """
    if not _CONFIG_PATH.exists():
        return {}
    return json.loads(_CONFIG_PATH.read_text())
