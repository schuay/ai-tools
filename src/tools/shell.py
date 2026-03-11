"""Shell execution tool for use by agents."""

import subprocess

MAX_OUTPUT = 50_000  # chars; truncate beyond this to avoid flooding context


def run_shell(command: str, timeout: int = 60) -> str:
    """Run a shell command and return its combined stdout/stderr output.

    Use this to invoke profiling tools (perf, flamegraph, addr2line, objdump),
    compilers, test runners, or any system command needed during analysis.
    Commands are executed via /bin/sh -c in the current working directory.

    command: the shell command to run
    timeout: maximum seconds to wait before killing the process (default 60)
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
