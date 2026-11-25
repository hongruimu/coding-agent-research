"""Tests for the read-only ``list`` workspace tool."""

from __future__ import annotations

from pathlib import Path

import pytest

import coding_agent.files as files
from coding_agent.files import PathEscapesRootError, register_file_tools
from coding_agent.protocol import ToolCall, ToolResult
from coding_agent.tools import ToolRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> ToolRegistry:
    root = tmp_path / "workspace"
    (root / "sub").mkdir(parents=True)
    (root / "empty").mkdir(parents=True)
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    (root / "b.py").write_text("b\n", encoding="utf-8")
    (root / "sub" / "c.txt").write_text("c\n", encoding="utf-8")

    reg = ToolRegistry()
    register_file_tools(reg, root)
    return reg


def _list(registry: ToolRegistry, path: str = ".") -> ToolResult:
    return registry.execute(
        ToolCall(id="call-1", name="list", arguments={"path": path})
    )


def test_list_returns_sorted_entries_with_directory_slash(
    registry: ToolRegistry,
) -> None:
    result = _list(registry)

    assert result.is_error is False
    assert result.content == "a.txt\nb.py\nempty/\nsub/"


def test_list_subdirectory_paths_are_relative_to_root(registry: ToolRegistry) -> None:
    result = _list(registry, "sub")

    assert result.is_error is False
    assert result.content == "sub/c.txt"


def test_list_empty_directory_returns_empty_string(registry: ToolRegistry) -> None:
    result = _list(registry, "empty")

    assert result.is_error is False
    assert result.content == ""


def test_list_out_of_bounds_is_a_protocol_error(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _list(registry, "../outside")


def test_list_file_path_is_an_error_result(registry: ToolRegistry) -> None:
    result = _list(registry, "a.txt")

    assert result.is_error is True
    assert "NotADirectoryError" in result.content


def test_list_nonexistent_path_is_an_error_result(registry: ToolRegistry) -> None:
    result = _list(registry, "missing")

    assert result.is_error is True
    assert "FileNotFoundError" in result.content


def test_list_truncates_with_a_marker(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files, "MAX_OUTPUT_LINES", 2)

    result = _list(registry)

    assert result.is_error is False
    assert result.content.splitlines() == [
        "a.txt",
        "b.py",
        "[truncated: showing 2 of 4 entries]",
    ]


def test_list_schema_exposes_only_the_path_parameter(registry: ToolRegistry) -> None:
    tool = registry.get("list")

    assert [parameter.name for parameter in tool.schema.parameters] == ["path"]
    assert tool.schema.parameters[0].required is False
