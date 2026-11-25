"""Read-only workspace file tools with a root boundary.

This module owns the first filesystem security boundary: every path a tool
receives is resolved against a workspace root bound at registration time, and
any path that resolves outside that root is rejected as a protocol error before
any file is read. The tools themselves are read-only.
"""

from __future__ import annotations

import os
from pathlib import Path

from coding_agent.protocol import ProtocolValidationError
from coding_agent.tools import ToolRegistry


class PathEscapesRootError(ProtocolValidationError):
    """Raised when a tool path resolves outside the bound workspace root."""

    def __init__(self, path: str) -> None:
        super().__init__(f"path {path!r} escapes the workspace root")
        self.path = path


def resolve_within_root(root: Path, rel_path: str) -> Path:
    """Resolve *rel_path* against *root*, rejecting paths that escape *root*.

    ``root`` must already be an absolute, symlink-resolved path (callers pass
    ``Path(root).resolve()``). The candidate is resolved with ``Path.resolve()``
    so relative components, ``..``, absolute inputs, and symbolic links are all
    normalized before the boundary check. A candidate that does not live under
    ``root`` raises ``PathEscapesRootError``; nothing is read or created here.
    """

    candidate = (root / rel_path).resolve()
    if not candidate.is_relative_to(root):
        raise PathEscapesRootError(rel_path)
    return candidate


MAX_OUTPUT_LINES = 2000


def _truncate(items: list[str], *, limit: int, unit: str) -> str:
    total = len(items)
    if total <= limit:
        return "\n".join(items)
    shown = "\n".join(items[:limit])
    return f"{shown}\n[truncated: showing {limit} of {total} {unit}]"


def register_file_tools(registry: ToolRegistry, root: str | Path) -> None:
    """Register the read-only workspace file tools against *root*.

    ``root`` is resolved once here and closed over by each tool, so the model
    can only supply paths relative to it and can never change the boundary.
    """

    resolved_root = Path(root).resolve()

    def list_directory(path: str = ".") -> str:
        target = resolve_within_root(resolved_root, path)
        if not target.exists():
            raise FileNotFoundError(f"no such directory: {path!r}")
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path!r}")

        entries = sorted(target.iterdir(), key=lambda entry: entry.name)
        lines = []
        for entry in entries:
            rel = entry.relative_to(resolved_root).as_posix()
            if entry.is_dir():
                rel += "/"
            lines.append(rel)
        return _truncate(lines, limit=MAX_OUTPUT_LINES, unit="entries")

    def read_file(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        target = resolve_within_root(resolved_root, path)
        if not target.exists():
            raise FileNotFoundError(f"no such file: {path!r}")
        if not target.is_file():
            raise IsADirectoryError(f"not a file: {path!r}")

        lines = target.read_text(encoding="utf-8").splitlines()
        start = 1 if start_line is None else start_line
        end = len(lines) if end_line is None else end_line

        if start < 1:
            raise ValueError(
                f"invalid line range: start_line must be >= 1 (got {start_line!r})"
            )
        if end_line is not None and end_line < 1:
            raise ValueError(
                f"invalid line range: end_line must be >= 1 (got {end_line!r})"
            )
        if start_line is not None and end_line is not None and start_line > end_line:
            raise ValueError(
                f"invalid line range: start_line {start_line} > end_line {end_line}"
            )

        return _truncate(lines[start - 1 : end], limit=MAX_OUTPUT_LINES, unit="lines")

    def search_files(keyword: str, path: str = ".") -> str:
        target = resolve_within_root(resolved_root, path)
        if not target.exists():
            raise FileNotFoundError(f"no such directory: {path!r}")
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path!r}")

        matches: list[tuple[str, int, str]] = []
        for dirpath, _dirnames, filenames in os.walk(target, followlinks=False):
            for filename in filenames:
                file_path = Path(dirpath) / filename
                try:
                    resolved = resolve_within_root(
                        resolved_root, file_path.relative_to(resolved_root).as_posix()
                    )
                except PathEscapesRootError:
                    continue
                try:
                    content = resolved.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                rel = resolved.relative_to(resolved_root).as_posix()
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if keyword in line:
                        matches.append((rel, line_number, line))

        matches.sort()
        lines = [f"{rel}:{ln}: {text}" for rel, ln, text in matches]
        if not lines:
            return "0 matches"
        return _truncate(lines, limit=MAX_OUTPUT_LINES, unit="matches")

    registry.register("list", list_directory)
    registry.register("read", read_file)
    registry.register("search", search_files)


__all__ = [
    "PathEscapesRootError",
    "register_file_tools",
    "resolve_within_root",
]
