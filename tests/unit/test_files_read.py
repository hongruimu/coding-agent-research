"""Tests for the read-only ``read`` workspace tool."""

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
    root.mkdir()
    (root / "f.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    (root / "empty.txt").write_text("", encoding="utf-8")
    (root / "sub").mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")

    reg = ToolRegistry()
    register_file_tools(reg, root)
    return reg


def _read(
    registry: ToolRegistry,
    path: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> ToolResult:
    arguments: dict[str, object] = {"path": path}
    if start_line is not None:
        arguments["start_line"] = start_line
    if end_line is not None:
        arguments["end_line"] = end_line
    return registry.execute(ToolCall(id="call-1", name="read", arguments=arguments))


def test_read_whole_file(registry: ToolRegistry) -> None:
    result = _read(registry, "f.txt")

    assert result.is_error is False
    assert result.content == "one\ntwo\nthree\nfour"


def test_read_empty_file_returns_empty_string(registry: ToolRegistry) -> None:
    result = _read(registry, "empty.txt")

    assert result.is_error is False
    assert result.content == ""


def test_read_line_range_is_one_based_inclusive(registry: ToolRegistry) -> None:
    result = _read(registry, "f.txt", start_line=2, end_line=3)

    assert result.is_error is False
    assert result.content == "two\nthree"


def test_read_start_line_only_reads_to_end(registry: ToolRegistry) -> None:
    result = _read(registry, "f.txt", start_line=3)

    assert result.is_error is False
    assert result.content == "three\nfour"


def test_read_end_line_only_reads_from_start(registry: ToolRegistry) -> None:
    result = _read(registry, "f.txt", end_line=2)

    assert result.is_error is False
    assert result.content == "one\ntwo"


def test_read_end_line_beyond_eof_clamps_to_end(registry: ToolRegistry) -> None:
    result = _read(registry, "f.txt", end_line=10)

    assert result.is_error is False
    assert result.content == "one\ntwo\nthree\nfour"


def test_read_out_of_bounds_is_a_protocol_error(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _read(registry, "../outside/secret.txt")


def test_read_directory_is_an_error_result(registry: ToolRegistry) -> None:
    result = _read(registry, "sub")

    assert result.is_error is True
    assert "IsADirectoryError" in result.content


def test_read_nonexistent_file_is_an_error_result(registry: ToolRegistry) -> None:
    result = _read(registry, "missing.txt")

    assert result.is_error is True
    assert "FileNotFoundError" in result.content


def test_read_start_line_beyond_eof_returns_empty(registry: ToolRegistry) -> None:
    result = _read(registry, "f.txt", start_line=10)

    assert result.is_error is False
    assert result.content == ""


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [(0, None), (2, 1), (None, 0), (5, 3)],
)
def test_read_invalid_line_range_is_an_error_result(
    registry: ToolRegistry,
    start_line: int | None,
    end_line: int | None,
) -> None:
    result = _read(registry, "f.txt", start_line=start_line, end_line=end_line)

    assert result.is_error is True
    assert "invalid line range" in result.content


def test_read_truncates_with_a_marker(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files, "MAX_OUTPUT_LINES", 2)

    result = _read(registry, "f.txt")

    assert result.is_error is False
    assert result.content.splitlines() == [
        "one",
        "two",
        "[truncated: showing 2 of 4 lines]",
    ]


def test_read_schema_parameters(registry: ToolRegistry) -> None:
    tool = registry.get("read")

    assert [parameter.name for parameter in tool.schema.parameters] == [
        "path",
        "start_line",
        "end_line",
    ]
    assert tool.schema.parameters[0].required is True
    assert tool.schema.parameters[1].required is False
    assert tool.schema.parameters[2].required is False
