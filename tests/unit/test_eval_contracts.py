import json
from pathlib import Path

import pytest

from coding_agent.evals.contracts import ContractValidationError, load_contract_pair


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _valid_documents() -> tuple[dict[str, object], dict[str, object]]:
    task = {
        "schema": "coding-agent/eval-task-v1",
        "task": {
            "id": "minimal-contract",
            "version": "1.0.0",
            "kind": "repository_exploration",
        },
        "environment": {
            "fixture_revision": "fixture-v1",
            "network": "disabled",
        },
        "agent_input": {
            "instruction_path": "instruction.md",
            "workspace_path": "workspace",
        },
        "grader": {
            "id": "minimal-contract-grader",
            "version": "1.0.0",
        },
        "constraints": {
            "allowed_paths": ["."],
            "forbidden_paths": [".git"],
        },
        "budgets": {
            "max_steps": 10,
            "max_time_seconds": 60,
            "max_tool_calls": 20,
            "max_output_bytes": 4096,
            "max_cost_usd": None,
        },
        "artifacts": {"required": ["final_answer.json"]},
    }
    grader = {
        "schema": "coding-agent/eval-grader-v1",
        "grader": {
            "id": "minimal-contract-grader",
            "version": "1.0.0",
        },
        "task": {"id": "minimal-contract", "version": "1.0.0"},
        "hidden_inputs": ["hidden/criteria.json"],
        "artifacts": {"required": ["final_answer.json"]},
    }
    return task, grader


def _contract_tree(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, object], dict[str, object]]:
    tasks_root = tmp_path / "tasks"
    graders_root = tmp_path / "graders"
    task_dir = tasks_root / "minimal-contract"
    grader_dir = graders_root / "minimal-contract"
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Inspect the repository.", encoding="utf-8")
    _write_json(grader_dir / "hidden" / "criteria.json", {"facts": []})
    task, grader = _valid_documents()
    task_path = task_dir / "task.json"
    grader_path = grader_dir / "grader.json"
    _write_json(task_path, task)
    _write_json(grader_path, grader)
    return tasks_root, graders_root, task_path, grader_path, task, grader


def test_loads_matching_task_and_grader_contracts(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, _, _ = _contract_tree(tmp_path)

    pair = load_contract_pair(
        task_path,
        grader_path,
        tasks_root=tasks_root,
        graders_root=graders_root,
    )

    assert pair.task.identity.id == "minimal-contract"
    assert pair.task.identity.version == "1.0.0"
    assert pair.grader.identity.id == "minimal-contract-grader"
    assert pair.task.workspace_path == task_path.parent / "workspace"
    assert pair.grader.hidden_inputs == (
        grader_path.parent / "hidden" / "criteria.json",
    )


def test_committed_minimal_example_is_valid() -> None:
    example_root = Path("evals/contracts/examples/minimal")

    pair = load_contract_pair(
        example_root / "tasks/minimal-contract/task.json",
        example_root / "graders/minimal-contract/grader.json",
        tasks_root=example_root / "tasks",
        graders_root=example_root / "graders",
    )

    assert pair.task.identity.id == "minimal-contract"
    assert pair.grader.task == pair.task.identity


def test_rejects_incomplete_task_contract(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, task, _ = _contract_tree(tmp_path)
    del task["budgets"]
    _write_json(task_path, task)

    with pytest.raises(ContractValidationError, match=r"task\.budgets"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=graders_root,
        )


def test_rejects_unknown_task_schema(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, task, _ = _contract_tree(tmp_path)
    task["schema"] = "coding-agent/eval-task-v2"
    _write_json(task_path, task)

    with pytest.raises(ContractValidationError, match=r"task\.schema"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=graders_root,
        )


def test_rejects_task_version_mismatch(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, _, grader = _contract_tree(tmp_path)
    grader["task"] = {"id": "minimal-contract", "version": "2.0.0"}
    _write_json(grader_path, grader)

    with pytest.raises(ContractValidationError, match="task identity"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=graders_root,
        )


def test_rejects_overlapping_task_and_grader_roots(tmp_path: Path) -> None:
    tasks_root, _, task_path, grader_path, _, _ = _contract_tree(tmp_path)

    with pytest.raises(ContractValidationError, match="must not overlap"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=tmp_path,
        )


def test_rejects_absolute_agent_visible_path(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, task, _ = _contract_tree(tmp_path)
    task["agent_input"] = {
        "instruction_path": str((tmp_path / "outside.md").resolve()),
        "workspace_path": "workspace",
    }
    _write_json(task_path, task)

    with pytest.raises(ContractValidationError, match="relative path"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=graders_root,
        )


def test_rejects_agent_visible_path_that_escapes_task_directory(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, task, _ = _contract_tree(tmp_path)
    task["agent_input"] = {
        "instruction_path": "../../graders/minimal-contract/hidden/criteria.json",
        "workspace_path": "workspace",
    }
    _write_json(task_path, task)

    with pytest.raises(ContractValidationError, match=r"must not contain '\.\.'"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=graders_root,
        )


def test_rejects_hidden_input_that_escapes_grader_directory(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, _, grader = _contract_tree(tmp_path)
    grader["hidden_inputs"] = ["../../tasks/minimal-contract/instruction.md"]
    _write_json(grader_path, grader)

    with pytest.raises(ContractValidationError, match=r"must not contain '\.\.'"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=graders_root,
        )


def test_rejects_agent_visible_symlink_to_hidden_grader(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, task, _ = _contract_tree(tmp_path)
    hidden_link = task_path.parent / "hidden-grader-link"
    hidden_link.symlink_to(grader_path.parent / "hidden", target_is_directory=True)
    task["agent_input"] = {
        "instruction_path": "hidden-grader-link/criteria.json",
        "workspace_path": "workspace",
    }
    _write_json(task_path, task)

    with pytest.raises(ContractValidationError, match="escapes task directory"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=graders_root,
        )


def test_requires_explicit_cost_budget_availability(tmp_path: Path) -> None:
    tasks_root, graders_root, task_path, grader_path, task, _ = _contract_tree(tmp_path)
    budgets = task["budgets"]
    assert isinstance(budgets, dict)
    del budgets["max_cost_usd"]
    _write_json(task_path, task)

    with pytest.raises(ContractValidationError, match=r"max_cost_usd is required"):
        load_contract_pair(
            task_path,
            grader_path,
            tasks_root=tasks_root,
            graders_root=graders_root,
        )
