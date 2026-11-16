"""Deterministic grader for the repository exploration v1 fixture."""

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


ANSWER_SCHEMA = "coding-agent/repository-exploration-answer-v1"
CRITERIA_SCHEMA = "coding-agent/repository-exploration-criteria-v1"


class GraderConfigurationError(ValueError):
    """Raised when committed hidden criteria are malformed."""


class ArtifactValidationError(ValueError):
    """Raised when a candidate artifact violates the answer contract."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def grade_submission(
    artifact_path: str | Path,
    criteria_path: str | Path,
) -> dict[str, object]:
    """Compare one submitted answer with one hidden fact set."""

    task_identity, expected_facts = _load_criteria(Path(criteria_path))
    checked_facts = sorted(expected_facts)
    try:
        artifact = _load_json_object(Path(artifact_path), artifact=True)
        actual_identity, actual_facts = _parse_answer(artifact)
    except ArtifactValidationError as error:
        return _result(False, [error.reason], checked_facts)

    failure_reasons: list[str] = []
    if actual_identity != task_identity:
        failure_reasons.append("artifact:task_identity_mismatch")

    expected_ids = set(expected_facts)
    actual_ids = set(actual_facts)
    for fact_id in sorted(expected_ids - actual_ids):
        failure_reasons.append(f"fact:{fact_id}:missing")
    for fact_id in sorted(actual_ids - expected_ids):
        failure_reasons.append(f"fact:{fact_id}:unexpected")
    for fact_id in sorted(expected_ids & actual_ids):
        expected = expected_facts[fact_id]
        actual = actual_facts[fact_id]
        if actual["value"] != expected["value"]:
            failure_reasons.append(f"fact:{fact_id}:value_mismatch")
        if actual["evidence"] != expected["evidence"]:
            failure_reasons.append(f"fact:{fact_id}:evidence_mismatch")

    return _result(not failure_reasons, failure_reasons, checked_facts)


def _load_criteria(
    path: Path,
) -> tuple[tuple[str, str], dict[str, dict[str, object]]]:
    try:
        document = _load_json_object(path, artifact=False)
        _require_exact_fields(document, {"schema", "task", "facts"})
        if document["schema"] != CRITERIA_SCHEMA:
            raise GraderConfigurationError("hidden criteria schema is invalid")
        identity = _parse_identity(document["task"], artifact=False)
        facts = _parse_facts(document["facts"], artifact=False)
    except GraderConfigurationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise GraderConfigurationError("hidden criteria are invalid") from error
    if not facts:
        raise GraderConfigurationError("hidden criteria facts must not be empty")
    return identity, facts


def _parse_answer(
    document: dict[str, Any],
) -> tuple[tuple[str, str], dict[str, dict[str, object]]]:
    try:
        _require_exact_fields(document, {"schema", "task", "facts"})
        if document["schema"] != ANSWER_SCHEMA:
            raise ArtifactValidationError("artifact:schema_mismatch")
        identity = _parse_identity(document["task"], artifact=True)
        facts = _parse_facts(document["facts"], artifact=True)
    except ArtifactValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError("artifact:invalid_format") from error
    return identity, facts


def _parse_identity(value: object, *, artifact: bool) -> tuple[str, str]:
    if not isinstance(value, dict):
        _raise_validation(artifact, "identity must be an object")
    _require_exact_fields(value, {"id", "version"})
    identity = (value["id"], value["version"])
    if any(not isinstance(item, str) or not item.strip() for item in identity):
        _raise_validation(artifact, "identity values must be non-empty strings")
    return identity


def _parse_facts(
    value: object,
    *,
    artifact: bool,
) -> dict[str, dict[str, object]]:
    if not isinstance(value, list):
        _raise_validation(artifact, "facts must be an array")
    facts: dict[str, dict[str, object]] = {}
    for fact in value:
        if not isinstance(fact, dict):
            _raise_validation(artifact, "each fact must be an object")
        _require_exact_fields(fact, {"id", "value", "evidence"})
        fact_id = fact["id"]
        fact_value = fact["value"]
        if not isinstance(fact_id, str) or not fact_id.strip():
            _raise_validation(artifact, "fact id must be a non-empty string")
        if fact_id in facts:
            _raise_validation(artifact, "fact ids must be unique")
        if not isinstance(fact_value, str) or not fact_value.strip():
            _raise_validation(artifact, "fact value must be a non-empty string")
        evidence = _parse_evidence(fact["evidence"], artifact=artifact)
        facts[fact_id] = {"value": fact_value, "evidence": evidence}
    return facts


def _parse_evidence(value: object, *, artifact: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _raise_validation(artifact, "evidence must be a non-empty array")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or "\\" in item:
            _raise_evidence_path(artifact)
        path = PurePosixPath(item)
        if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
            _raise_evidence_path(artifact)
        normalized.append(path.as_posix())
    if len(normalized) != len(set(normalized)):
        _raise_validation(artifact, "evidence paths must be unique")
    return tuple(sorted(normalized))


def _load_json_object(path: Path, *, artifact: bool) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        if artifact:
            raise ArtifactValidationError("artifact:unreadable") from error
        raise GraderConfigurationError("cannot read hidden criteria") from error
    if not isinstance(document, dict):
        if artifact:
            raise ArtifactValidationError("artifact:invalid_format")
        raise GraderConfigurationError("hidden criteria root must be an object")
    return document


def _require_exact_fields(document: dict[str, Any], expected: set[str]) -> None:
    if set(document) != expected:
        raise ValueError("object fields do not match contract")


def _raise_validation(artifact: bool, message: str) -> None:
    if artifact:
        raise ArtifactValidationError("artifact:invalid_format")
    raise GraderConfigurationError(message)


def _raise_evidence_path(artifact: bool) -> None:
    if artifact:
        raise ArtifactValidationError("artifact:invalid_evidence_path")
    raise GraderConfigurationError("hidden evidence path is invalid")


def _result(
    passed: bool,
    failure_reasons: list[str],
    checked_facts: list[str],
) -> dict[str, object]:
    return {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "checked_facts": checked_facts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("criteria", type=Path)
    args = parser.parse_args(argv)
    result = grade_submission(args.artifact, args.criteria)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
