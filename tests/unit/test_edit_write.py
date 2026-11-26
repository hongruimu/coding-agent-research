"""Tests for the ``write`` workspace tool (M2.2-T1)."""

from __future__ import annotations

import os
import stat
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
    (workspace / "a.txt").write_text("old content\n", encoding="utf-8")

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


def _write(registry: ToolRegistry, path: str, content: str) -> ToolResult:
    return registry.execute(
        ToolCall(id="call-1", name="write", arguments={"path": path, "content": content})
    )


def test_write_creates_a_new_file(root: Path, registry: ToolRegistry) -> None:
    result = _write(registry, "new.txt", "hello\n")

    assert result.is_error is False
    assert (root / "new.txt").read_text(encoding="utf-8") == "hello\n"


def test_write_overwrites_an_existing_file(root: Path, registry: ToolRegistry) -> None:
    result = _write(registry, "a.txt", "replacement\n")

    assert result.is_error is False
    assert (root / "a.txt").read_text(encoding="utf-8") == "replacement\n"


def test_write_returns_a_confirmation_with_relative_path(
    registry: ToolRegistry,
) -> None:
    result = _write(registry, "a.txt", "hi\n")

    assert result.is_error is False
    assert result.content == "wrote 3 characters to a.txt"


def test_write_missing_parent_directory_is_an_error_result(
    registry: ToolRegistry,
) -> None:
    result = _write(registry, "no_such_dir/file.txt", "x\n")

    assert result.is_error is True
    assert "FileNotFoundError" in result.content


def test_write_to_an_existing_directory_is_an_error_result(
    registry: ToolRegistry,
) -> None:
    result = _write(registry, "sub", "x\n")

    assert result.is_error is True
    assert "IsADirectoryError" in result.content


def test_write_relative_escape_is_a_protocol_error(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _write(registry, "../outside/secret.txt", "pwned\n")


def test_write_absolute_escape_is_a_protocol_error(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    outside_secret = tmp_path / "outside" / "secret.txt"
    with pytest.raises(PathEscapesRootError):
        _write(registry, str(outside_secret), "pwned\n")


def test_write_symlink_escape_is_a_protocol_error(registry: ToolRegistry) -> None:
    with pytest.raises(PathEscapesRootError):
        _write(registry, "link_out", "pwned\n")


def test_write_atomic_failure_keeps_original_and_leaves_no_temp(
    root: Path, registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = root / "a.txt"
    original = target.read_text(encoding="utf-8")

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)

    result = _write(registry, "a.txt", "should not land\n")

    assert result.is_error is True
    assert target.read_text(encoding="utf-8") == original
    leftovers = [entry for entry in root.iterdir() if entry.name.startswith(".tmp-")]
    assert leftovers == []


def test_write_preserves_existing_file_mode(
    root: Path, registry: ToolRegistry
) -> None:
    script = root / "script.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(script, 0o755)

    result = _write(registry, "script.sh", "#!/bin/sh\necho hi\n")

    assert result.is_error is False
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    assert script.read_text(encoding="utf-8") == "#!/bin/sh\necho hi\n"


def test_write_schema_exposes_only_path_and_content(registry: ToolRegistry) -> None:
    tool = registry.get("write")

    assert [parameter.name for parameter in tool.schema.parameters] == [
        "path",
        "content",
    ]
    assert all(parameter.required for parameter in tool.schema.parameters)
