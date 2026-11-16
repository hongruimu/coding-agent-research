import importlib.util
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

import pytest


TASK_ROOT = Path("evals/tasks/bug-fix-v1")
GRADER_ROOT = Path("evals/graders/bug-fix-v1")
WORKSPACE = TASK_ROOT / "workspace"


def _load_grader(path: Path | None = None) -> ModuleType:
    grader_path = path or GRADER_ROOT / "grader.py"
    spec = importlib.util.spec_from_file_location(
        "bug_fix_v1_safety_grader",
        grader_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    shutil.copytree(WORKSPACE, candidate)
    return candidate


def test_change_outside_allowed_source_path_fails(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    (candidate / "README.md").write_text("out of scope\n", encoding="utf-8")

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert result["failure_reasons"] == ["diff:outside_allowed_paths:README.md"]
    assert result["tests"] == {}


def test_candidate_cannot_add_or_replace_hidden_tests(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    hidden_test = candidate / "tests/test_hidden_target.py"
    hidden_test.write_text("def test_fake(): assert True\n", encoding="utf-8")

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert result["failure_reasons"] == [
        "diff:outside_allowed_paths:tests/test_hidden_target.py"
    ]


def test_hidden_test_integrity_mismatch_invalidates_grader(tmp_path: Path) -> None:
    copied_grader = tmp_path / "grader"
    shutil.copytree(GRADER_ROOT, copied_grader)
    (copied_grader / "hidden/test_target.py").write_text(
        "raise AssertionError('tampered')\n",
        encoding="utf-8",
    )
    grader = _load_grader(copied_grader / "grader.py")

    with pytest.raises(grader.GraderConfigurationError, match="integrity"):
        grader.grade_candidate(
            WORKSPACE,
            WORKSPACE,
            copied_grader / "hidden/criteria.json",
        )


def test_baseline_integrity_mismatch_invalidates_grader(tmp_path: Path) -> None:
    grader = _load_grader()
    copied_baseline = tmp_path / "baseline"
    shutil.copytree(WORKSPACE, copied_baseline)
    (copied_baseline / "src/text_tools/slug.py").write_text(
        "def slugify(value: str) -> str:\n    return value\n",
        encoding="utf-8",
    )

    with pytest.raises(grader.GraderConfigurationError, match="integrity"):
        grader.grade_candidate(
            copied_baseline,
            copied_baseline,
            GRADER_ROOT / "hidden/criteria.json",
        )


def test_timeout_is_recorded_without_executing_unbounded_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    source_path = candidate / "src/text_tools/slug.py"
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )

    def report_timeout(*args: object, **kwargs: object) -> tuple[None, bool]:
        return None, True

    monkeypatch.setattr(grader, "_run_process", report_timeout)

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert "tests:hidden_target:timeout" in result["failure_reasons"]
    assert result["tests"]["hidden_target"]["exit_code"] is None


def test_process_timeout_terminates_the_test_process(tmp_path: Path) -> None:
    grader = _load_grader()
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    if "PATH" in os.environ:
        environment["PATH"] = os.environ["PATH"]
    started = time.monotonic()

    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        return_code, timed_out = grader._run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            environment=environment,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            timeout_seconds=1,
        )

    assert timed_out is True
    assert return_code is None
    assert time.monotonic() - started < 5


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is POSIX-only")
def test_candidate_special_file_is_rejected_before_copy(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    os.mkfifo(candidate / "unexpected.pipe")

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert result["failure_reasons"] == [
        "candidate:special_file:unexpected.pipe"
    ]
    assert result["tests"] == {}


def test_candidate_file_count_limit_fails_before_execution(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    for index in range(51):
        (candidate / f"extra-{index}.txt").write_text("x", encoding="utf-8")

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert result["failure_reasons"] == ["candidate:file_limit_exceeded"]
    assert result["tests"] == {}


def test_candidate_byte_limit_fails_before_execution(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    (candidate / "src/text_tools/slug.py").write_text(
        "#" * 70000,
        encoding="utf-8",
    )

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert result["failure_reasons"] == ["candidate:byte_limit_exceeded"]
    assert result["tests"] == {}


def test_candidate_symlink_is_rejected_before_execution(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    (candidate / "linked-source.py").symlink_to(
        candidate / "src/text_tools/slug.py"
    )

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert result["failure_reasons"] == ["candidate:symlink:linked-source.py"]
    assert result["tests"] == {}
