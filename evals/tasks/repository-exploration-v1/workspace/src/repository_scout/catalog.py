"""Small deterministic behavior for repository exploration."""


def build_catalog(paths: list[str]) -> dict[str, int]:
    return {path: len(path) for path in sorted(paths)}
