"""Integration: register and drive the real write tools through ToolRegistry."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from coding_agent.edit import register_edit_tools
from coding_agent.files import PathEscapesRootError, register_file_tools
from coding_agent.protocol import ToolCall, ToolResult
from coding_agent.tools import ToolRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> ToolRegistry:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8"
    )

    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "token.txt").write_text("SECRET_TOKEN=abc\n", encoding="utf-8")

    reg = ToolRegistry()
    register_file_tools(reg, workspace)
    register_edit_tools(reg, workspace)
    return reg


def _execute(
    registry: ToolRegistry, name: str, arguments: dict[str, object]
) -> ToolResult:
    return registry.execute(
        ToolCall(id=f"call-{name}", name=name, arguments=arguments)
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_registry_contains_read_and_write_tools(registry: ToolRegistry) -> None:
    for name in ("list", "read", "search", "write", "edit"):
        assert name in registry


def test_write_then_edit_form_a_coherent_chain(registry: ToolRegistry) -> None:
    write = _execute(registry, "write", {"path": "notes.txt", "content": "hello\n"})
    assert write.is_error is False
    assert write.content == "wrote 6 characters to notes.txt"

    edit = _execute(
        registry,
        "edit",
        {"path": "src/app.py", "old_string": "return 1", "new_string": "return 2"},
    )
    assert edit.is_error is False
    assert edit.content == "edited src/app.py (1 replacement)"

    read_back = _execute(registry, "read", {"path": "src/app.py"})
    assert read_back.is_error is False
    assert read_back.content == "def main():\n    return 2"


def test_out_of_bounds_write_is_a_protocol_error_not_a_result(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(PathEscapesRootError):
        _execute(registry, "write", {"path": "../secrets/token.txt", "content": "x\n"})


def test_out_of_bounds_edit_is_a_protocol_error_not_a_result(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(PathEscapesRootError):
        _execute(
            registry,
            "edit",
            {"path": "../secrets/token.txt", "old_string": "abc", "new_string": "x"},
        )


def test_missing_parent_write_is_an_error_result(registry: ToolRegistry) -> None:
    result = _execute(registry, "write", {"path": "no_dir/f.txt", "content": "x\n"})

    assert result.is_error is True
    assert "FileNotFoundError" in result.content


def test_changes_are_visible_via_git_and_rollbackable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "app.py")
    _git(
        repo,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-q",
        "-m",
        "baseline",
    )

    reg = ToolRegistry()
    register_edit_tools(reg, repo)

    assert _execute(reg, "write", {"path": "notes.txt", "content": "hello\n"}).is_error is False
    assert (
        _execute(
            reg,
            "edit",
            {"path": "app.py", "old_string": "return 1", "new_string": "return 2"},
        ).is_error
        is False
    )

    status = _git(repo, "status", "--porcelain")
    assert "notes.txt" in status
    assert "app.py" in status
    assert ".tmp-" not in status

    # Simulate the grader/eval framework's rollback (the Agent has no such tool).
    _git(repo, "reset", "--hard", "-q", "HEAD")
    _git(repo, "clean", "-fdq")

    assert _git(repo, "status", "--porcelain") == ""
    assert (repo / "app.py").read_text(encoding="utf-8") == "def main():\n    return 1\n"
    assert not (repo / "notes.txt").exists()
