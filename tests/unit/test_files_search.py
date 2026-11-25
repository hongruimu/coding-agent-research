"""Tests for the read-only ``search`` workspace tool."""

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
    (root / "a.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (root / "b.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
    (root / "sub" / "c.txt").write_text("gamma again\ndelta\n", encoding="utf-8")
    (root / "many.txt").write_text("hit\n" * 5, encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET\n", encoding="utf-8")

    reg = ToolRegistry()
    register_file_tools(reg, root)
    return reg


def _search(registry: ToolRegistry, keyword: str, path: str = ".") -> ToolResult:
    return registry.execute(
        ToolCall(id="call-1", name="search", arguments={"keyword": keyword, "path": path})
    )


def test_search_returns_matches_with_file_line_content(registry: ToolRegistry) -> None:
    result = _search(registry, "gamma")

    assert result.is_error is False
    assert result.content == "a.txt:3: gamma\nsub/c.txt:1: gamma again"


def test_search_no_matches_returns_marker(registry: ToolRegistry) -> None:
    result = _search(registry, "zzz")

    assert result.is_error is False
    assert result.content == "0 matches"


def test_search_out_of_bounds_is_a_protocol_error(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _search(registry, "secret", path="../outside")


def test_search_file_path_is_an_error_result(registry: ToolRegistry) -> None:
    result = _search(registry, "alpha", path="a.txt")

    assert result.is_error is True
    assert "NotADirectoryError" in result.content


def test_search_nonexistent_path_is_an_error_result(registry: ToolRegistry) -> None:
    result = _search(registry, "alpha", path="missing")

    assert result.is_error is True
    assert "FileNotFoundError" in result.content


def test_search_truncates_with_a_marker(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files, "MAX_OUTPUT_LINES", 2)

    result = _search(registry, "hit")

    assert result.is_error is False
    assert result.content.splitlines() == [
        "many.txt:1: hit",
        "many.txt:2: hit",
        "[truncated: showing 2 of 5 matches]",
    ]


def test_search_does_not_follow_symlink_escaping_root(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    root = tmp_path / "workspace"
    (root / "link_out").symlink_to(tmp_path / "outside" / "secret.txt")

    result = _search(registry, "TOPSECRET")

    assert result.is_error is False
    assert result.content == "0 matches"


def test_search_schema_parameters(registry: ToolRegistry) -> None:
    tool = registry.get("search")

    assert [parameter.name for parameter in tool.schema.parameters] == [
        "keyword",
        "path",
    ]
    assert tool.schema.parameters[0].required is True
    assert tool.schema.parameters[1].required is False
