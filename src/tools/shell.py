"""Shell execution tool for use by agents."""

import subprocess
from pathlib import Path

MAX_OUTPUT = 50_000  # chars; truncate beyond this to avoid flooding context


def run_shell(command: str, timeout: int = 60) -> str:
    """Run a shell command (/bin/sh -c) and return stdout+stderr.

    command: shell command to run
    timeout: max seconds (default 60)
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"

    out = result.stdout
    if result.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + result.stderr
    if result.returncode not in (0, 1):
        out += f"\n[exit {result.returncode}]"

    out = out.strip()
    if not out:
        return "(no output)"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n[truncated — {len(out) - MAX_OUTPUT:,} more chars]"
    return out


def run_d8(
    d8_dir: str,
    d8_args: list[str],
    timeout: int = 60,
    stdout_file: str | None = None,
    stderr_file: str | None = None,
) -> str:
    """Run the d8 JavaScript shell.

    d8_dir: directory containing the d8 binary
    d8_args: arguments to pass to d8
    timeout: max seconds (default 60)
    stdout_file: redirect stdout to this file instead of capturing
    stderr_file: redirect stderr to this file instead of capturing
    """
    cmd = [str(Path(d8_dir) / "d8"), *d8_args]
    stdout = open(stdout_file, "w") if stdout_file else subprocess.PIPE
    stderr = open(stderr_file, "w") if stderr_file else subprocess.PIPE
    try:
        result = subprocess.run(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"Error: d8 timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        if stdout_file:
            stdout.close()
        if stderr_file:
            stderr.close()

    parts: list[str] = []
    if stdout_file:
        parts.append(f"[stdout → {stdout_file}]")
    elif result.stdout:
        parts.append(result.stdout)
    if stderr_file:
        parts.append(f"[stderr → {stderr_file}]")
    elif result.stderr:
        parts.append("[stderr]\n" + result.stderr)
    if result.returncode not in (0, 1):
        parts.append(f"[exit {result.returncode}]")

    out = "\n".join(parts).strip()
    if not out:
        return "(no output)"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n[truncated — {len(out) - MAX_OUTPUT:,} more chars]"
    return out
