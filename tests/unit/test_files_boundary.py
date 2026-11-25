"""Boundary tests for the workspace path-resolution primitive."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.files import PathEscapesRootError, resolve_within_root
from coding_agent.protocol import ProtocolValidationError


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "f.txt").write_text("hello", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")

    (root / "link_out").symlink_to(outside / "secret.txt")
    (root / "link_in").symlink_to(root / "sub", target_is_directory=True)

    return root.resolve()


def test_error_is_a_protocol_validation_error() -> None:
    assert issubclass(PathEscapesRootError, ProtocolValidationError)


def test_in_bounds_path_resolves(workspace: Path) -> None:
    assert resolve_within_root(workspace, "sub/f.txt") == (workspace / "sub" / "f.txt").resolve()


def test_dot_and_empty_resolve_to_root(workspace: Path) -> None:
    assert resolve_within_root(workspace, ".") == workspace
    assert resolve_within_root(workspace, "") == workspace


def test_nonexistent_path_within_root_resolves(workspace: Path) -> None:
    resolved = resolve_within_root(workspace, "does/not/exist.txt")
    assert resolved == (workspace / "does" / "not" / "exist.txt").resolve()
    assert resolved.is_relative_to(workspace)


def test_relative_escape_is_rejected(workspace: Path) -> None:
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(workspace, "../outside/secret.txt")


def test_absolute_path_is_rejected(workspace: Path, tmp_path: Path) -> None:
    outside_secret = tmp_path / "outside" / "secret.txt"
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(workspace, str(outside_secret))


def test_symlink_escape_is_rejected(workspace: Path) -> None:
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(workspace, "link_out")


def test_symlink_inside_root_resolves(workspace: Path) -> None:
    resolved = resolve_within_root(workspace, "link_in")
    assert resolved == (workspace / "sub").resolve()
    assert resolved.is_relative_to(workspace)


def test_error_carries_the_offending_path(workspace: Path) -> None:
    with pytest.raises(PathEscapesRootError) as caught:
        resolve_within_root(workspace, "../outside")
    assert caught.value.path == "../outside"
