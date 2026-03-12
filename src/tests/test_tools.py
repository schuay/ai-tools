"""Unit tests for tool implementations.

Tests focus on pure logic and I/O with tmp files — no network calls,
no live git operations, no model API calls.
"""

import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── fs helpers ────────────────────────────────────────────────────────────────

from tools.fs import _find, _unified_diff, edit_file, list_dir, read_file, write_file
from tools.git import _cap_chars, git_commit_meta, trim_to_context
from tools.shell import MAX_OUTPUT, run_shell


# ══════════════════════════════════════════════════════════════════════════════
# _find
# ══════════════════════════════════════════════════════════════════════════════


class TestFind:
    def test_exact_match(self):
        content = "hello\nworld\n"
        start, end = _find(content, "hello\n")
        assert content[start:end] == "hello\n"

    def test_exact_match_middle(self):
        content = "aaa\nbbb\nccc\n"
        start, end = _find(content, "bbb\n")
        assert content[start:end] == "bbb\n"

    def test_no_match(self):
        assert _find("hello world", "goodbye") is None

    def test_empty_search(self):
        # empty string matches at position 0 (Python str.find behaviour)
        assert _find("hello world", "") == (0, 0)

    def test_trailing_whitespace_pass2(self):
        # content has trailing spaces, search does not
        content = "def foo():  \n    pass\n"
        search = "def foo():\n    pass\n"
        result = _find(content, search)
        assert result is not None

    def test_multiline_exact(self):
        content = "line1\nline2\nline3\n"
        start, end = _find(content, "line1\nline2\n")
        assert content[start:end] == "line1\nline2\n"

    def test_similar_but_below_threshold(self):
        # Completely different content should not match
        assert _find("aaaaaaaaaa", "bbbbbbbbbb") is None


# ══════════════════════════════════════════════════════════════════════════════
# _unified_diff
# ══════════════════════════════════════════════════════════════════════════════


class TestUnifiedDiff:
    def test_identical_inputs(self):
        assert _unified_diff("hello\n", "hello\n", "f.py") == ""

    def test_diff_header_format(self):
        diff = _unified_diff("old\n", "new\n", "myfile.py")
        assert "a/myfile.py" in diff
        assert "b/myfile.py" in diff

    def test_diff_shows_change(self):
        diff = _unified_diff("old\n", "new\n", "f.py")
        assert "-old" in diff
        assert "+new" in diff


# ══════════════════════════════════════════════════════════════════════════════
# read_file
# ══════════════════════════════════════════════════════════════════════════════


class TestReadFile:
    def test_reads_content(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world\n")
        assert read_file(str(f)) == "hello world\n"

    def test_missing_file(self, tmp_path):
        result = read_file(str(tmp_path / "nope.txt"))
        assert result.startswith("Error:")
        assert "does not exist" in result

    def test_directory_error(self, tmp_path):
        result = read_file(str(tmp_path))
        assert result.startswith("Error:")
        assert "not a file" in result

    def test_truncation(self, tmp_path):
        f = tmp_path / "big.txt"
        # Each line ~10 chars; context default 80 * 500 chars/line = 40k chars cap
        # Use a small context to force truncation
        f.write_text("x" * 1000 + "\n")
        result = read_file(str(f), context=1)  # 1 * 500 = 500 chars max
        assert "truncated" in result

    def test_line_context(self, tmp_path):
        lines = [f"line{i}\n" for i in range(1, 101)]
        f = tmp_path / "many.txt"
        f.write_text("".join(lines))
        result = read_file(str(f), line=50, context=5)
        assert ">>>" in result
        assert "line50" in result
        assert "line1\n" not in result  # too far from line 50


# ══════════════════════════════════════════════════════════════════════════════
# write_file / edit_file
# ══════════════════════════════════════════════════════════════════════════════


class TestWriteFile:
    def test_creates_file(self, tmp_path):
        f = tmp_path / "new.txt"
        result = write_file(str(f), "content\n")
        assert "Written" in result
        assert f.read_text() == "content\n"

    def test_creates_parents(self, tmp_path):
        f = tmp_path / "a" / "b" / "c.txt"
        write_file(str(f), "hi\n")
        assert f.exists()

    def test_overwrites(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("old\n")
        write_file(str(f), "new\n")
        assert f.read_text() == "new\n"


class TestEditFile:
    def test_simple_replace(self, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("def foo():\n    return 1\n")
        result = edit_file(str(f), "return 1", "return 2")
        assert "Applied" in result
        assert f.read_text() == "def foo():\n    return 2\n"

    def test_missing_file(self, tmp_path):
        result = edit_file(str(tmp_path / "nope.py"), "x", "y")
        assert result.startswith("Error:")

    def test_search_not_found(self, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("hello\n")
        result = edit_file(str(f), "goodbye", "hi")
        assert "Error" in result or "not found" in result.lower()

    def test_no_change_outside_match(self, tmp_path):
        f = tmp_path / "f.py"
        original = "aaa\nbbb\nccc\n"
        f.write_text(original)
        edit_file(str(f), "bbb", "BBB")
        assert f.read_text() == "aaa\nBBB\nccc\n"


# ══════════════════════════════════════════════════════════════════════════════
# list_dir
# ══════════════════════════════════════════════════════════════════════════════


class TestListDir:
    def test_lists_files_and_dirs(self, tmp_path):
        (tmp_path / "a_dir").mkdir()
        (tmp_path / "z_file.txt").write_text("x")
        result = list_dir(str(tmp_path))
        assert "a_dir/" in result
        assert "z_file.txt" in result

    def test_dirs_before_files(self, tmp_path):
        (tmp_path / "zzz_dir").mkdir()
        (tmp_path / "aaa_file.txt").write_text("x")
        result = list_dir(str(tmp_path))
        assert result.index("zzz_dir/") < result.index("aaa_file.txt")

    def test_empty_dir(self, tmp_path):
        assert list_dir(str(tmp_path)) == "(empty)"

    def test_missing_path(self, tmp_path):
        result = list_dir(str(tmp_path / "nope"))
        assert result.startswith("Error:")

    def test_file_not_dir(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x")
        result = list_dir(str(f))
        assert result.startswith("Error:")
        assert "not a directory" in result


# ══════════════════════════════════════════════════════════════════════════════
# _cap_chars
# ══════════════════════════════════════════════════════════════════════════════


class TestCapChars:
    def test_within_limit(self):
        assert _cap_chars("hello", 100) == "hello"

    def test_at_limit(self):
        s = "x" * 100
        assert _cap_chars(s, 100) == s

    def test_over_limit(self):
        s = "x" * 200
        result = _cap_chars(s, 100)
        assert result.startswith("x" * 100)
        assert "truncated" in result
        assert "100" in result  # excess chars mentioned

    def test_empty(self):
        assert _cap_chars("", 100) == ""


# ══════════════════════════════════════════════════════════════════════════════
# trim_to_context
# ══════════════════════════════════════════════════════════════════════════════


class TestTrimToContext:
    def _make_text(self, n: int) -> str:
        return "".join(f"line{i}\n" for i in range(1, n + 1))

    def test_none_returns_full(self):
        text = self._make_text(10)
        assert trim_to_context(text, None) == text

    def test_marker_on_target_line(self):
        text = self._make_text(20)
        result = trim_to_context(text, 10, context=3)
        assert ">>>" in result
        assert "line10" in result

    def test_context_limits_output(self):
        text = self._make_text(100)
        result = trim_to_context(text, 50, context=5)
        # Should show ~10 lines around line 50, not all 100
        assert "line1\n" not in result
        assert "line100\n" not in result

    def test_line_1_at_start(self):
        text = self._make_text(50)
        result = trim_to_context(text, 1, context=5)
        assert ">>>" in result
        assert "line1" in result

    def test_last_line(self):
        text = self._make_text(50)
        result = trim_to_context(text, 50, context=5)
        assert ">>>" in result
        assert "line50" in result


# ══════════════════════════════════════════════════════════════════════════════
# git_commit_meta
# ══════════════════════════════════════════════════════════════════════════════


class TestGitCommitMeta:
    def test_parses_well_formed_output(self, monkeypatch):
        fake_output = "abc123\nauthor@example.com\n2024-01-15 10:00:00 +0000\nFix the bug\n\nLonger body here.\n"

        import tools.git as git_mod

        monkeypatch.setattr(git_mod, "_git", lambda args: fake_output)
        meta = git_commit_meta("abc123")
        assert meta["hash"] == "abc123"
        assert meta["author"] == "author@example.com"
        assert meta["date"] == "2024-01-15 10:00:00 +0000"
        assert meta["subject"] == "Fix the bug"
        assert "Longer body" in meta["body"]

    def test_handles_short_output(self, monkeypatch):
        import tools.git as git_mod

        monkeypatch.setattr(git_mod, "_git", lambda args: "Error: bad hash")
        meta = git_commit_meta("badhash")
        # hash key echoes the input commit_hash; other fields are empty
        assert meta["hash"] == "badhash"
        assert meta["author"] == ""
        assert meta["subject"] == ""


# ══════════════════════════════════════════════════════════════════════════════
# run_shell
# ══════════════════════════════════════════════════════════════════════════════


class TestRunShell:
    def test_basic_command(self):
        result = run_shell("echo hello")
        assert result == "hello"

    def test_empty_output(self):
        result = run_shell("true")
        assert result == "(no output)"

    def test_stderr_included(self):
        result = run_shell("echo err >&2")
        assert "err" in result

    def test_nonzero_exit_code(self):
        result = run_shell("exit 2")
        assert "[exit 2]" in result

    def test_exit_1_not_flagged(self):
        # exit code 1 is treated as success (grep convention)
        result = run_shell("exit 1")
        assert "[exit 1]" not in result

    def test_truncation(self):
        # Generate output larger than MAX_OUTPUT
        result = run_shell(f"python3 -c \"print('x' * {MAX_OUTPUT + 100})\"")
        assert "truncated" in result

    def test_timeout(self):
        result = run_shell("sleep 10", timeout=1)
        assert "timed out" in result


# ══════════════════════════════════════════════════════════════════════════════
# standard_tools composition
# ══════════════════════════════════════════════════════════════════════════════


class TestStandardTools:
    def test_web_fetch_always_included(self):
        from tools import standard_tools
        from tools.web import web_fetch

        tools = standard_tools(web=True, git=False, fs=False)
        assert web_fetch in tools

    def test_web_search_requires_tavily_key(self, monkeypatch):
        from tools import standard_tools
        from tools.web import web_search

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tools = standard_tools(web=True, git=False, fs=False)
        assert web_search not in tools

    def test_fs_tools_included_when_flag_set(self):
        from tools import standard_tools
        from tools.fs import edit_file, grep_files, list_dir, read_file, write_file

        tools = standard_tools(web=False, git=False, fs=True)
        assert read_file in tools
        assert grep_files in tools
        assert list_dir in tools
        assert edit_file in tools
        assert write_file in tools

    def test_shell_tool_included_when_flag_set(self):
        from tools import standard_tools
        from tools.shell import run_shell

        tools = standard_tools(web=False, git=False, fs=False, shell=True)
        assert run_shell in tools

    def test_no_shell_by_default(self):
        from tools import standard_tools
        from tools.shell import run_shell

        tools = standard_tools()
        assert run_shell not in tools


# ══════════════════════════════════════════════════════════════════════════════
# _add_items_to_arrays (schema helper)
# ══════════════════════════════════════════════════════════════════════════════


class TestAddItemsToArrays:
    def setup_method(self):
        from graph import _add_items_to_arrays

        self.fix = _add_items_to_arrays

    def test_adds_items_to_bare_array(self):
        schema = {"type": "array"}
        self.fix(schema)
        assert "items" in schema

    def test_leaves_existing_items_alone(self):
        schema = {"type": "array", "items": {"type": "string"}}
        self.fix(schema)
        assert schema["items"] == {"type": "string"}

    def test_nested_array_fixed(self):
        schema = {"properties": {"x": {"type": "array"}}}
        self.fix(schema)
        assert "items" in schema["properties"]["x"]

    def test_array_in_list(self):
        schema = {"anyOf": [{"type": "array"}, {"type": "null"}]}
        self.fix(schema)
        assert "items" in schema["anyOf"][0]

    def test_non_dict_noop(self):
        self.fix("not a dict")  # should not raise

    def test_object_untouched(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        self.fix(schema)
        assert "items" not in schema
