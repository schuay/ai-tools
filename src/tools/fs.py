"""Filesystem tools: directory listing and file creation."""

import os
from pathlib import Path


def list_dir(path: str = ".") -> str:
    """List the contents of a directory.

    Directories are shown with a trailing '/'. Entries are sorted
    alphabetically, directories first.

    path: directory path relative to the working directory (default: '.')
    """
    dir_path = Path(os.getcwd()) / path
    if not dir_path.exists():
        return f"Error: {path} does not exist"
    if not dir_path.is_dir():
        return f"Error: {path} is not a directory"

    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError as e:
        return f"Error: {e}"

    lines = []
    for entry in entries:
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            size = entry.stat().st_size
            lines.append(f"{entry.name}  ({size:,} B)")
    return "\n".join(lines) if lines else "(empty)"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Parent directories are created automatically if they do not exist.
    The user is asked to approve or reject before this tool runs
    (configured via interrupt_on in the agent).

    path:    absolute path or path relative to the working directory
    content: the full text content to write
    """
    file_path = Path(path) if Path(path).is_absolute() else Path(os.getcwd()) / path
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Written {file_path}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"
