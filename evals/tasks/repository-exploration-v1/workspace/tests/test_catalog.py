import pytest

from repository_scout import build_catalog


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["beta", "a"], {"a": 1, "beta": 4}),
        ([], {}),
    ],
)
def test_build_catalog(paths: list[str], expected: dict[str, int]) -> None:
    assert build_catalog(paths) == expected
