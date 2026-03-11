import os

from tools.fs import edit_file, grep_files, list_dir, read_file, write_file
from tools.git import (
    REPO_ROOT,
    git_blame,
    git_commit_meta,
    git_commits_since,
    git_commits_since_date,
    git_fetch,
    git_grep,
    git_log,
    git_resolve,
    git_show,
    in_git_repo,
    read_around,
    trim_to_context,
)
from tools.shell import run_shell
from tools.web import web_fetch, web_search


def standard_tools(
    *,
    web: bool = True,
    git: bool = True,
    fs: bool = False,
    shell: bool = False,
) -> list:
    """Return the standard tool set for an agent, based on context and flags.

    web:   include web_fetch (always) and web_search (if TAVILY_API_KEY is set)
    git:   include git read tools (only if the cwd is inside a git repo)
    fs:    include filesystem tools: read_file, grep_files, list_dir,
           edit_file, write_file
    shell: include run_shell for arbitrary command execution
    """
    tools: list = []

    if web and os.environ.get("TAVILY_API_KEY"):
        tools.append(web_search)
    if web:
        tools.append(web_fetch)

    if git and in_git_repo():
        tools += [git_grep, git_show, git_blame, git_log, read_around]

    if fs:
        tools += [read_file, grep_files, list_dir, edit_file, write_file]

    if shell:
        tools.append(run_shell)

    return tools


__all__ = [
    "REPO_ROOT",
    "edit_file",
    "grep_files",
    "git_blame",
    "git_commit_meta",
    "git_commits_since",
    "git_commits_since_date",
    "git_fetch",
    "git_grep",
    "git_log",
    "git_resolve",
    "git_show",
    "in_git_repo",
    "list_dir",
    "read_file",
    "read_around",
    "run_shell",
    "standard_tools",
    "trim_to_context",
    "web_fetch",
    "web_search",
    "write_file",
]
