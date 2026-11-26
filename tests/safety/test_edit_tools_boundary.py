"""Safety: write tools cannot modify anything outside the workspace root."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.edit import register_edit_tools
from coding_agent.files import PathEscapesRootError, register_file_tools
from coding_agent.protocol import ToolCall, ToolResult
from coding_agent.tools import ToolRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> ToolRegistry:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "public.txt").write_text("public data\n", encoding="utf-8")

    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "facts.json").write_text('{"secret": "answer"}\n', encoding="utf-8")

    # A symlink planted inside the workspace pointing at the hidden file.
    (workspace / "link").symlink_to(hidden / "facts.json")

    reg = ToolRegistry()
    register_file_tools(reg, workspace)
    register_edit_tools(reg, workspace)
    return reg


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=arguments)


def _execute(
    registry: ToolRegistry, name: str, arguments: dict[str, object]
) -> ToolResult:
    return registry.execute(_call(name, arguments))


def test_write_outside_root_is_rejected_and_untouched(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    hidden = tmp_path / "hidden" / "facts.json"
    original = hidden.read_text(encoding="utf-8")

    with pytest.raises(PathEscapesRootError):
        _execute(registry, "write", {"path": "../hidden/facts.json", "content": "pwned\n"})

    assert hidden.read_text(encoding="utf-8") == original


def test_edit_outside_root_is_rejected_and_untouched(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    hidden = tmp_path / "hidden" / "facts.json"
    original = hidden.read_text(encoding="utf-8")

    with pytest.raises(PathEscapesRootError):
        _execute(
            registry,
            "edit",
            {"path": "../hidden/facts.json", "old_string": "answer", "new_string": "pwned"},
        )

    assert hidden.read_text(encoding="utf-8") == original


def test_absolute_write_outside_root_is_rejected(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    hidden_facts = tmp_path / "hidden" / "facts.json"
    with pytest.raises(PathEscapesRootError):
        _execute(registry, "write", {"path": str(hidden_facts), "content": "pwned\n"})


def test_symlink_escape_cannot_overwrite_hidden_file(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    hidden = tmp_path / "hidden" / "facts.json"
    original = hidden.read_text(encoding="utf-8")

    with pytest.raises(PathEscapesRootError):
        _execute(registry, "write", {"path": "link", "content": "pwned\n"})

    assert hidden.read_text(encoding="utf-8") == original


def test_no_rollback_or_shell_tool_is_registered(registry: ToolRegistry) -> None:
    forbidden = {
        "git",
        "shell",
        "bash",
        "sh",
        "reset",
        "rollback",
        "revert",
        "checkout",
        "rm",
        "mv",
        "delete",
    }

    for name in forbidden:
        assert name not in registry
