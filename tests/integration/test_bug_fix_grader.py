import hashlib
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

from coding_agent.evals import load_contract_pair


TASK_ROOT = Path("evals/tasks/bug-fix-v1")
GRADER_ROOT = Path("evals/graders/bug-fix-v1")
WORKSPACE = TASK_ROOT / "workspace"


def _load_grader() -> ModuleType:
    grader_path = GRADER_ROOT / "grader.py"
    spec = importlib.util.spec_from_file_location("bug_fix_v1_grader", grader_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    shutil.copytree(WORKSPACE, candidate)
    return candidate


def _apply_reference_fix(candidate: Path) -> None:
    source_path = candidate / "src/text_tools/slug.py"
    source = source_path.read_text(encoding="utf-8")
    fixed = source.replace(
        'return value.strip().lower().replace(" ", "-")',
        'return "-".join(value.strip().lower().split())',
    )
    assert fixed != source
    source_path.write_text(fixed, encoding="utf-8")


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_bug_fix_contract_pair_loads() -> None:
    pair = load_contract_pair(
        TASK_ROOT / "task.json",
        GRADER_ROOT / "grader.json",
        tasks_root=Path("evals/tasks"),
        graders_root=Path("evals/graders"),
    )

    assert pair.task.identity.id == "bug-fix-v1"
    assert pair.task.kind == "bug_fix"
    assert pair.task.workspace_path == WORKSPACE.resolve()
    assert len(pair.grader.hidden_inputs) == 4


def test_committed_baseline_reproduces_target_failure() -> None:
    grader = _load_grader()

    result = grader.grade_candidate(
        WORKSPACE,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert "diff:empty" in result["failure_reasons"]
    assert "tests:hidden_target:failed" in result["failure_reasons"]
    assert result["tests"]["public_regression"]["status"] == "passed"
    assert result["tests"]["hidden_regression"]["status"] == "passed"
    assert result["tests"]["hidden_target"]["tests_run"] == 2


def test_reference_fix_passes_hidden_target_and_regressions(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    _apply_reference_fix(candidate)
    baseline_digest = _tree_digest(WORKSPACE)
    candidate_digest = _tree_digest(candidate)

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is True
    assert result["failure_reasons"] == []
    assert result["changed_files"] == ["src/text_tools/slug.py"]
    assert {
        group: details["status"] for group, details in result["tests"].items()
    } == {
        "hidden_regression": "passed",
        "hidden_target": "passed",
        "public_regression": "passed",
    }
    assert all(details["exit_code"] == 0 for details in result["tests"].values())
    assert all(details["tests_run"] > 0 for details in result["tests"].values())
    assert _tree_digest(WORKSPACE) == baseline_digest
    assert _tree_digest(candidate) == candidate_digest


def test_skipped_tests_cannot_produce_a_pass(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    (candidate / "src/text_tools/slug.py").write_text(
        "from unittest import SkipTest\n\n"
        "def slugify(value: str) -> str:\n"
        "    raise SkipTest('candidate skipped behavior')\n",
        encoding="utf-8",
    )

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert "tests:hidden_target:skipped" in result["failure_reasons"]
    assert result["tests"]["hidden_target"]["skipped"] == 2


def test_excessive_test_output_fails_with_truncation_metadata(tmp_path: Path) -> None:
    grader = _load_grader()
    candidate = _copy_candidate(tmp_path)
    (candidate / "src/text_tools/slug.py").write_text(
        "print('x' * 20000)\n\n"
        "def slugify(value: str) -> str:\n"
        "    return '-'.join(value.strip().lower().split())\n",
        encoding="utf-8",
    )

    result = grader.grade_candidate(
        candidate,
        WORKSPACE,
        GRADER_ROOT / "hidden/criteria.json",
    )

    assert result["passed"] is False
    assert "tests:hidden_target:output_limit_exceeded" in result["failure_reasons"]
    assert result["tests"]["hidden_target"]["stdout"]["truncated"] is True
    assert result["tests"]["hidden_target"]["stdout"]["stored_bytes"] <= 8192
