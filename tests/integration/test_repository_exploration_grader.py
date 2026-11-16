import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from coding_agent.evals import load_contract_pair


TASK_ROOT = Path("evals/tasks/repository-exploration-v1")
GRADER_ROOT = Path("evals/graders/repository-exploration-v1")


def _load_grader() -> ModuleType:
    grader_path = GRADER_ROOT / "grader.py"
    spec = importlib.util.spec_from_file_location(
        "repository_exploration_v1_grader",
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


def _write_answer(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "final_answer.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_repository_exploration_contract_pair_loads() -> None:
    pair = load_contract_pair(
        TASK_ROOT / "task.json",
        GRADER_ROOT / "grader.json",
        tasks_root=Path("evals/tasks"),
        graders_root=Path("evals/graders"),
    )

    assert pair.task.identity.id == "repository-exploration-v1"
    assert pair.task.kind == "repository_exploration"
    assert pair.task.workspace_path == (TASK_ROOT / "workspace").resolve()
    assert pair.grader.hidden_inputs == ((GRADER_ROOT / "hidden/facts.json").resolve(),)


def test_correct_structured_answer_passes_independent_grader(tmp_path: Path) -> None:
    grader = _load_grader()
    artifact_path = _write_answer(tmp_path, _correct_answer())

    result = grader.grade_submission(
        artifact_path,
        GRADER_ROOT / "hidden/facts.json",
    )

    assert result == {
        "passed": True,
        "failure_reasons": [],
        "checked_facts": [
            "package_directory",
            "python_requirement",
            "test_framework",
            "tested_callable",
        ],
    }


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda answer: answer["facts"][0].update({"value": ">=3.12"}),
            "fact:python_requirement:value_mismatch",
        ),
        (
            lambda answer: answer["facts"].pop(2),
            "fact:tested_callable:missing",
        ),
    ],
)
def test_wrong_or_missing_fact_fails(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    expected_reason: str,
) -> None:
    grader = _load_grader()
    answer = _correct_answer()
    mutate(answer)
    artifact_path = _write_answer(tmp_path, answer)

    result = grader.grade_submission(
        artifact_path,
        GRADER_ROOT / "hidden/facts.json",
    )

    assert result["passed"] is False
    assert expected_reason in result["failure_reasons"]


def test_grader_reads_only_submission_and_hidden_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grader = _load_grader()
    artifact_path = _write_answer(tmp_path, _correct_answer()).resolve()
    facts_path = (GRADER_ROOT / "hidden/facts.json").resolve()
    original_read_text = Path.read_text
    observed_reads: list[Path] = []

    def observe_read(path: Path, *args: object, **kwargs: object) -> str:
        observed_reads.append(path.resolve())
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", observe_read)

    result = grader.grade_submission(artifact_path, facts_path)

    assert result["passed"] is True
    assert observed_reads == [facts_path, artifact_path]
