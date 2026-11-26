"""Controlled code-modification tools with a root boundary.

This module owns the first write capability: ``write`` overwrites or creates a
file inside the workspace root bound at registration time. Every path is
resolved against the same root boundary used by ``coding_agent.files``, and any
path that resolves outside that root is rejected as a protocol error before
anything is written. Files are written atomically so a failed write never
leaves a half-written file behind.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from coding_agent.files import resolve_within_root
from coding_agent.tools import ToolRegistry


def _atomic_write(resolved: Path, content: str) -> None:
    """Atomically replace *resolved* with *content*.

    ``resolved`` must already be an absolute, symlink-resolved path inside the
    workspace root (the result of ``resolve_within_root``). The content is first
    written to a temp file in the same directory, then ``os.replace`` swaps it
    into place, so the target is either fully old or fully new — never
    half-written. An existing target's mode is preserved; the temp file is
    removed on any failure.
    """

    fd, tmp_path = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=".tmp-", suffix=".edit"
    )
    try:
        if resolved.exists():
            mode = stat.S_IMODE(resolved.stat().st_mode)
            os.chmod(tmp_path, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(tmp_path, resolved)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def register_edit_tools(registry: ToolRegistry, root: str | Path) -> None:
    """Register the controlled write tools against *root*.

    ``root`` is resolved once here and closed over by each tool, so the model
    can only supply paths relative to it and can never change the boundary.
    """

    resolved_root = Path(root).resolve()

    def write_file(path: str, content: str) -> str:
        target = resolve_within_root(resolved_root, path)
        if not target.parent.is_dir():
            raise FileNotFoundError(f"parent directory does not exist: {path!r}")
        if target.is_dir():
            raise IsADirectoryError(f"not a file: {path!r}")

        _atomic_write(target, content)
        rel = target.relative_to(resolved_root).as_posix()
        return f"wrote {len(content)} characters to {rel}"

    def edit_file(path: str, old_string: str, new_string: str) -> str:
        target = resolve_within_root(resolved_root, path)
        if not target.exists():
            raise FileNotFoundError(f"no such file: {path!r}")
        if not target.is_file():
            raise IsADirectoryError(f"not a file: {path!r}")
        if not old_string:
            raise ValueError("old_string must be non-empty")

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            raise ValueError(f"old_string not found in {path!r}")
        if count > 1:
            raise ValueError(f"old_string matches {count} locations; not unique")

        new_content = content.replace(old_string, new_string, 1)
        _atomic_write(target, new_content)
        rel = target.relative_to(resolved_root).as_posix()
        return f"edited {rel} (1 replacement)"

    registry.register("write", write_file)
    registry.register("edit", edit_file)


__all__ = ["register_edit_tools"]
