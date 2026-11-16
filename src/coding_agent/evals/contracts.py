"""Load and validate versioned evaluation task and grader contracts."""

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


TASK_SCHEMA = "coding-agent/eval-task-v1"
GRADER_SCHEMA = "coding-agent/eval-grader-v1"


class ContractValidationError(ValueError):
    """Raised when an evaluation contract violates the v1 contract."""


@dataclass(frozen=True, slots=True)
class ContractIdentity:
    id: str
    version: str


@dataclass(frozen=True, slots=True)
class TaskContract:
    identity: ContractIdentity
    kind: str
    grader: ContractIdentity
    instruction_path: Path
    workspace_path: Path


@dataclass(frozen=True, slots=True)
class GraderContract:
    identity: ContractIdentity
    task: ContractIdentity
    hidden_inputs: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ContractPair:
    task: TaskContract
    grader: GraderContract


def load_contract_pair(
    task_contract_path: str | Path,
    grader_contract_path: str | Path,
    *,
    tasks_root: str | Path,
    graders_root: str | Path,
) -> ContractPair:
    """Load a task/grader pair and enforce identity and visibility boundaries."""

    resolved_tasks_root = _require_directory(tasks_root, "tasks_root")
    resolved_graders_root = _require_directory(graders_root, "graders_root")
    _validate_separate_roots(resolved_tasks_root, resolved_graders_root)

    resolved_task_path = _require_contract_file(
        task_contract_path,
        root=resolved_tasks_root,
        field="task_contract_path",
    )
    resolved_grader_path = _require_contract_file(
        grader_contract_path,
        root=resolved_graders_root,
        field="grader_contract_path",
    )

    task = _load_task_contract(resolved_task_path)
    grader = _load_grader_contract(resolved_grader_path)

    if grader.task != task.identity:
        raise ContractValidationError(
            "task identity mismatch: grader targets "
            f"{grader.task.id}@{grader.task.version}, but task is "
            f"{task.identity.id}@{task.identity.version}"
        )
    if task.grader != grader.identity:
        raise ContractValidationError(
            "grader identity mismatch: task references "
            f"{task.grader.id}@{task.grader.version}, but grader is "
            f"{grader.identity.id}@{grader.identity.version}"
        )

    return ContractPair(task=task, grader=grader)


def _load_task_contract(path: Path) -> TaskContract:
    document = _load_json_object(path, "task")
    _require_schema(document, TASK_SCHEMA, "task.schema")

    task_section = _require_mapping(document, "task", "task.task")
    identity = _identity(task_section, "task.task")
    kind = _require_string(task_section, "kind", "task.task.kind")

    environment = _require_mapping(document, "environment", "task.environment")
    _require_string(
        environment,
        "fixture_revision",
        "task.environment.fixture_revision",
    )
    network = _require_string(environment, "network", "task.environment.network")
    if network not in {"disabled", "allowed"}:
        raise ContractValidationError(
            "task.environment.network must be 'disabled' or 'allowed'"
        )

    agent_input = _require_mapping(document, "agent_input", "task.agent_input")
    instruction_path = _resolve_member(
        path.parent,
        _require_string(
            agent_input,
            "instruction_path",
            "task.agent_input.instruction_path",
        ),
        field="task.agent_input.instruction_path",
        boundary="task directory",
        expected="file",
    )
    workspace_path = _resolve_member(
        path.parent,
        _require_string(
            agent_input,
            "workspace_path",
            "task.agent_input.workspace_path",
        ),
        field="task.agent_input.workspace_path",
        boundary="task directory",
        expected="directory",
    )

    grader_section = _require_mapping(document, "grader", "task.grader")
    grader = _identity(grader_section, "task.grader")

    constraints = _require_mapping(document, "constraints", "task.constraints")
    _require_relative_path_list(
        constraints,
        "allowed_paths",
        "task.constraints.allowed_paths",
    )
    _require_relative_path_list(
        constraints,
        "forbidden_paths",
        "task.constraints.forbidden_paths",
    )

    budgets = _require_mapping(document, "budgets", "task.budgets")
    for field in (
        "max_steps",
        "max_time_seconds",
        "max_tool_calls",
        "max_output_bytes",
    ):
        _require_positive_number(budgets, field, f"task.budgets.{field}")
    if "max_cost_usd" not in budgets:
        raise ContractValidationError("task.budgets.max_cost_usd is required")
    max_cost = budgets["max_cost_usd"]
    if max_cost is not None and not _is_non_negative_number(max_cost):
        raise ContractValidationError(
            "task.budgets.max_cost_usd must be null or a non-negative number"
        )

    artifacts = _require_mapping(document, "artifacts", "task.artifacts")
    _require_string_list(artifacts, "required", "task.artifacts.required")

    return TaskContract(
        identity=identity,
        kind=kind,
        grader=grader,
        instruction_path=instruction_path,
        workspace_path=workspace_path,
    )


def _load_grader_contract(path: Path) -> GraderContract:
    document = _load_json_object(path, "grader")
    _require_schema(document, GRADER_SCHEMA, "grader.schema")

    grader_section = _require_mapping(document, "grader", "grader.grader")
    identity = _identity(grader_section, "grader.grader")
    task_section = _require_mapping(document, "task", "grader.task")
    task = _identity(task_section, "grader.task")

    hidden_input_values = _require_string_list(
        document,
        "hidden_inputs",
        "grader.hidden_inputs",
    )
    if not hidden_input_values:
        raise ContractValidationError("grader.hidden_inputs must not be empty")
    hidden_inputs = tuple(
        _resolve_member(
            path.parent,
            value,
            field=f"grader.hidden_inputs[{index}]",
            boundary="grader directory",
            expected="file",
        )
        for index, value in enumerate(hidden_input_values)
    )

    artifacts = _require_mapping(document, "artifacts", "grader.artifacts")
    _require_string_list(artifacts, "required", "grader.artifacts.required")

    return GraderContract(
        identity=identity,
        task=task,
        hidden_inputs=hidden_inputs,
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractValidationError(f"cannot read {label} contract {path}: {error}") from error
    if not isinstance(document, dict):
        raise ContractValidationError(f"{label} contract root must be an object")
    return document


def _require_schema(
    document: dict[str, Any], expected: str, field: str
) -> None:
    value = _require_string(document, "schema", field)
    if value != expected:
        raise ContractValidationError(f"{field} must be {expected!r}, got {value!r}")


def _identity(section: dict[str, Any], field: str) -> ContractIdentity:
    return ContractIdentity(
        id=_require_string(section, "id", f"{field}.id"),
        version=_require_string(section, "version", f"{field}.version"),
    )


def _require_mapping(
    document: dict[str, Any], key: str, field: str
) -> dict[str, Any]:
    if key not in document:
        raise ContractValidationError(f"{field} is required")
    value = document[key]
    if not isinstance(value, dict):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _require_string(document: dict[str, Any], key: str, field: str) -> str:
    if key not in document:
        raise ContractValidationError(f"{field} is required")
    value = document[key]
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value


def _require_string_list(
    document: dict[str, Any], key: str, field: str
) -> list[str]:
    if key not in document:
        raise ContractValidationError(f"{field} is required")
    value = document[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ContractValidationError(f"{field} must be a list of non-empty strings")
    return value


def _require_relative_path_list(
    document: dict[str, Any], key: str, field: str
) -> tuple[PurePosixPath, ...]:
    values = _require_string_list(document, key, field)
    return tuple(
        _relative_posix_path(value, f"{field}[{index}]")
        for index, value in enumerate(values)
    )


def _require_positive_number(
    document: dict[str, Any], key: str, field: str
) -> int | float:
    if key not in document:
        raise ContractValidationError(f"{field} is required")
    value = document[key]
    if not _is_number(value) or value <= 0:
        raise ContractValidationError(f"{field} must be a positive number")
    return value


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_non_negative_number(value: object) -> bool:
    return _is_number(value) and value >= 0


def _relative_posix_path(value: str, field: str) -> PurePosixPath:
    if "\\" in value:
        raise ContractValidationError(f"{field} must use POSIX path separators")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ContractValidationError(f"{field} must be a relative path")
    if ".." in path.parts:
        raise ContractValidationError(f"{field} must not contain '..'")
    return path


def _resolve_member(
    root: Path,
    value: str,
    *,
    field: str,
    boundary: str,
    expected: str,
) -> Path:
    relative = _relative_posix_path(value, field)
    candidate = (root / Path(*relative.parts)).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ContractValidationError(f"{field} escapes {boundary}")
    if expected == "file" and not candidate.is_file():
        raise ContractValidationError(f"{field} must reference an existing file")
    if expected == "directory" and not candidate.is_dir():
        raise ContractValidationError(f"{field} must reference an existing directory")
    return candidate


def _require_directory(path: str | Path, field: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ContractValidationError(f"{field} must reference an existing directory")
    return resolved


def _require_contract_file(path: str | Path, *, root: Path, field: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ContractValidationError(f"{field} must be inside {root}")
    if not resolved.is_file():
        raise ContractValidationError(f"{field} must reference an existing file")
    return resolved


def _validate_separate_roots(tasks_root: Path, graders_root: Path) -> None:
    if tasks_root == graders_root:
        raise ContractValidationError("tasks_root and graders_root must be separate")
    if tasks_root.is_relative_to(graders_root) or graders_root.is_relative_to(tasks_root):
        raise ContractValidationError("tasks_root and graders_root must not overlap")
