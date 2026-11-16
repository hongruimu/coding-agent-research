import importlib.util
import json
from pathlib import Path
from types import ModuleType


TASK_ROOT = Path("evals/tasks/repository-exploration-v1")
GRADER_ROOT = Path("evals/graders/repository-exploration-v1")


def _load_grader() -> ModuleType:
    grader_path = GRADER_ROOT / "grader.py"
    spec = importlib.util.spec_from_file_location(
        "repository_exploration_v1_safety_grader",
        grader_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _correct_answer() -> dict[str, object]:
    return {
        "schema": "coding-agent/repository-exploration-answer-v1",
        "task": {"id": "repository-exploration-v1", "version": "1.0.0"},
        "facts": [
            {
                "id": "python_requirement",
                "value": ">=3.13",
                "evidence": ["pyproject.toml"],
            },
            {
                "id": "package_directory",
                "value": "src/repository_scout",
                "evidence": ["src/repository_scout/__init__.py"],
            },
            {
                "id": "tested_callable",
                "value": "build_catalog",
                "evidence": [
                    "src/repository_scout/catalog.py",
                    "tests/test_catalog.py",
                ],
            },
            {
                "id": "test_framework",
                "value": "pytest",
                "evidence": ["pyproject.toml", "tests/test_catalog.py"],
            },
        ],
    }


def test_task_metadata_and_instruction_do_not_disclose_hidden_answers() -> None:
    public_text = "\n".join(
        [
            (TASK_ROOT / "task.json").read_text(encoding="utf-8"),
            (TASK_ROOT / "instruction.md").read_text(encoding="utf-8"),
        ]
    )
    hidden = json.loads(
        (GRADER_ROOT / "hidden/facts.json").read_text(encoding="utf-8")
    )

    for fact in hidden["facts"]:
        assert fact["value"] not in public_text
        for evidence_path in fact["evidence"]:
            assert evidence_path not in public_text


def test_agent_visible_task_tree_contains_no_hidden_grader_links() -> None:
    task_root = TASK_ROOT.resolve()
    grader_root = GRADER_ROOT.resolve()
    visible_paths = list(TASK_ROOT.rglob("*"))

    assert TASK_ROOT.is_dir()
    assert visible_paths
    for path in visible_paths:
        assert not path.is_symlink()
        assert path.resolve().is_relative_to(task_root)
        assert not path.resolve().is_relative_to(grader_root)


def test_hidden_grader_path_cannot_be_claimed_as_workspace_evidence(
    tmp_path: Path,
) -> None:
    grader = _load_grader()
    answer = _correct_answer()
    answer["facts"][2]["evidence"] = [
        "../../graders/repository-exploration-v1/hidden/facts.json"
    ]
    artifact_path = tmp_path / "final_answer.json"
    artifact_path.write_text(json.dumps(answer), encoding="utf-8")

    result = grader.grade_submission(
        artifact_path,
        GRADER_ROOT / "hidden/facts.json",
    )

    assert result["passed"] is False
    assert result["failure_reasons"] == ["artifact:invalid_evidence_path"]
