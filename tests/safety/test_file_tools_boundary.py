"""Safety: file tools cannot read or list anything outside the workspace root."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.files import PathEscapesRootError, register_file_tools
from coding_agent.protocol import ToolCall, ToolResult
from coding_agent.tools import ToolRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> ToolRegistry:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "public.txt").write_text("public data\n", encoding="utf-8")

    # Hidden sibling directory that must never be reachable from the workspace.
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "facts.json").write_text('{"secret": "answer"}\n', encoding="utf-8")

    # A symlink planted inside the workspace pointing at the hidden file.
    (workspace / "link").symlink_to(hidden / "facts.json")

    reg = ToolRegistry()
    register_file_tools(reg, workspace)
    return reg


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=arguments)


def _execute(registry: ToolRegistry, name: str, arguments: dict[str, object]) -> ToolResult:
    return registry.execute(_call(name, arguments))


def test_read_outside_root_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _execute(registry, "read", {"path": "../hidden/facts.json"})


def test_absolute_read_outside_root_is_rejected(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    hidden_facts = tmp_path / "hidden" / "facts.json"
    with pytest.raises(PathEscapesRootError):
        _execute(registry, "read", {"path": str(hidden_facts)})


def test_search_outside_root_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _execute(registry, "search", {"keyword": "secret", "path": "../hidden"})


def test_list_does_not_expose_outside_entries(registry: ToolRegistry) -> None:
    result = _execute(registry, "list", {"path": "."})

    assert result.is_error is False
    assert result.content == "link\npublic.txt"

    with pytest.raises(PathEscapesRootError):
        _execute(registry, "list", {"path": "../hidden"})


def test_symlink_escape_cannot_leak_hidden_content(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _execute(registry, "read", {"path": "link"})

    result = _execute(registry, "search", {"keyword": "secret"})

    assert result.is_error is False
    assert result.content == "0 matches"
