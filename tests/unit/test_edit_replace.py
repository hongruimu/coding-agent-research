"""Tests for the ``edit`` workspace tool (M2.2-T2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.edit import register_edit_tools
from coding_agent.files import PathEscapesRootError
from coding_agent.protocol import ToolCall, ToolResult
from coding_agent.tools import ToolRegistry


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "app.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8"
    )
    (workspace / "dup.txt").write_text("x = 1\nx = 1\n", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")

    (workspace / "link_out").symlink_to(outside / "secret.txt")
    return workspace


@pytest.fixture()
def registry(root: Path) -> ToolRegistry:
    reg = ToolRegistry()
    register_edit_tools(reg, root)
    return reg


def _edit(
    registry: ToolRegistry, path: str, old_string: str, new_string: str
) -> ToolResult:
    return registry.execute(
        ToolCall(
            id="call-1",
            name="edit",
            arguments={
                "path": path,
                "old_string": old_string,
                "new_string": new_string,
            },
        )
    )


def test_edit_replaces_the_unique_occurrence(
    root: Path, registry: ToolRegistry
) -> None:
    result = _edit(registry, "app.py", "return 1", "return 2")

    assert result.is_error is False
    assert (
        root / "app.py"
    ).read_text(encoding="utf-8") == "def main():\n    return 2\n"


def test_edit_returns_a_confirmation(registry: ToolRegistry) -> None:
    result = _edit(registry, "app.py", "return 1", "return 2")

    assert result.is_error is False
    assert result.content == "edited app.py (1 replacement)"


def test_edit_non_unique_old_string_is_an_error_result(
    registry: ToolRegistry,
) -> None:
    result = _edit(registry, "dup.txt", "x = 1", "x = 2")

    assert result.is_error is True
    assert "not unique" in result.content


def test_edit_missing_old_string_is_an_error_result(
    registry: ToolRegistry,
) -> None:
    result = _edit(registry, "app.py", "does not exist", "whatever")

    assert result.is_error is True
    assert "not found" in result.content


def test_edit_empty_old_string_is_an_error_result(registry: ToolRegistry) -> None:
    result = _edit(registry, "app.py", "", "whatever")

    assert result.is_error is True


def test_edit_multiline_old_string_replaces(
    root: Path, registry: ToolRegistry
) -> None:
    result = _edit(
        registry,
        "app.py",
        "def main():\n    return 1",
        "def main():\n    return 2",
    )

    assert result.is_error is False
    assert (
        root / "app.py"
    ).read_text(encoding="utf-8") == "def main():\n    return 2\n"


def test_edit_missing_file_is_an_error_result(registry: ToolRegistry) -> None:
    result = _edit(registry, "missing.txt", "a", "b")

    assert result.is_error is True
    assert "FileNotFoundError" in result.content


def test_edit_directory_is_an_error_result(registry: ToolRegistry) -> None:
    result = _edit(registry, "sub", "a", "b")

    assert result.is_error is True
    assert "IsADirectoryError" in result.content


def test_edit_relative_escape_is_a_protocol_error(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _edit(registry, "../outside/secret.txt", "secret", "pwned")


def test_edit_symlink_escape_is_a_protocol_error(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _edit(registry, "link_out", "secret", "pwned")


def test_edit_schema_exposes_path_old_string_new_string(
    registry: ToolRegistry,
) -> None:
    tool = registry.get("edit")

    assert [parameter.name for parameter in tool.schema.parameters] == [
        "path",
        "old_string",
        "new_string",
    ]
    assert all(parameter.required for parameter in tool.schema.parameters)
