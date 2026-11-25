"""Integration: register and drive the real file tools through ToolRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.files import PathEscapesRootError, register_file_tools
from coding_agent.protocol import ToolCall, ToolResult
from coding_agent.tools import ToolRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> ToolRegistry:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Project\n\nA demo.\n", encoding="utf-8")

    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "token.txt").write_text("SECRET_TOKEN=abc123\n", encoding="utf-8")

    reg = ToolRegistry()
    register_file_tools(reg, root)
    return reg


def _execute(
    registry: ToolRegistry, name: str, arguments: dict[str, object]
) -> ToolResult:
    return registry.execute(
        ToolCall(id=f"call-{name}", name=name, arguments=arguments)
    )


def test_registry_contains_all_three_file_tools(registry: ToolRegistry) -> None:
    assert "list" in registry
    assert "read" in registry
    assert "search" in registry


def test_list_read_search_form_a_coherent_chain(registry: ToolRegistry) -> None:
    listing = _execute(registry, "list", {"path": "."})
    assert listing.is_error is False
    assert listing.content == "README.md\nsrc/"

    read = _execute(registry, "read", {"path": "src/app.py"})
    assert read.is_error is False
    assert read.content == "def main():\n    return 1"

    search = _execute(registry, "search", {"keyword": "return"})
    assert search.is_error is False
    assert search.content == "src/app.py:2:     return 1"


def test_out_of_bounds_read_is_a_protocol_error_not_a_result(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(PathEscapesRootError):
        _execute(registry, "read", {"path": "../secrets/token.txt"})


def test_nonexistent_read_is_an_error_result(registry: ToolRegistry) -> None:
    result = _execute(registry, "read", {"path": "missing.txt"})

    assert result.is_error is True
    assert "FileNotFoundError" in result.content
