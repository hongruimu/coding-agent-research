"""Deterministic evaluation outcomes and versioned run-record validation."""

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


RUN_RECORD_SCHEMA = "coding-agent/eval-run-record-v1"
REQUIRED_BUDGETS = (
    "steps",
    "time_seconds",
    "tool_calls",
    "output_bytes",
    "cost_usd",
)
SENSITIVE_KEY_SUFFIXES = (
    "api_key",
    "access_token",
    "refresh_token",
)
SENSITIVE_KEY_PARTS = {
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


class FunctionalStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class SafetyStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"


class RecordStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class BudgetStatus(StrEnum):
    WITHIN = "within"
    EXCEEDED = "exceeded"
    UNVERIFIABLE = "unverifiable"
    NOT_APPLICABLE = "not_applicable"


class BudgetAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class OverallStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class RunRecordValidationError(ValueError):
    """Raised when a run record contradicts the v1 record contract."""


@dataclass(frozen=True, slots=True)
class BudgetResult:
    name: str
    limit: int | float | None
    observed: int | float | None
    source: str | None
    availability: BudgetAvailability
    status: BudgetStatus


@dataclass(frozen=True, slots=True)
class RunOutcome:
    functional: FunctionalStatus
    safety: SafetyStatus
    record: RecordStatus
    budgets: tuple[BudgetResult, ...]
    overall: OverallStatus
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Redaction:
    label: str
    count: int

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "count": self.count}


@dataclass(frozen=True, slots=True)
class RetainedText:
    text: str
    input_bytes: int
    redacted_bytes: int
    stored_bytes: int
    truncated: bool
    redactions: tuple[Redaction, ...]
    redacted_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "input_bytes": self.input_bytes,
            "redacted_bytes": self.redacted_bytes,
            "stored_bytes": self.stored_bytes,
            "truncated": self.truncated,
            "redactions": [redaction.to_dict() for redaction in self.redactions],
            "redacted_sha256": self.redacted_sha256,
        }


@dataclass(frozen=True, slots=True)
class RunRecord:
    document: dict[str, Any]
    budgets: tuple[BudgetResult, ...]
    outcome: RunOutcome


def evaluate_budget(
    name: str,
    *,
    limit: int | float | None,
    observed: int | float | None,
    source: str | None,
) -> BudgetResult:
    """Evaluate one budget without treating missing observations as zero."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("budget name must be a non-empty string")
    _validate_optional_number(limit, "budget limit")
    _validate_optional_number(observed, "budget observed value")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("budget source must be null or a non-empty string")

    if limit is None:
        availability = BudgetAvailability.NOT_APPLICABLE
        status = BudgetStatus.NOT_APPLICABLE
    elif observed is None or source is None:
        availability = BudgetAvailability.UNAVAILABLE
        status = BudgetStatus.UNVERIFIABLE
    elif observed > limit:
        availability = BudgetAvailability.AVAILABLE
        status = BudgetStatus.EXCEEDED
    else:
        availability = BudgetAvailability.AVAILABLE
        status = BudgetStatus.WITHIN

    return BudgetResult(
        name=name,
        limit=limit,
        observed=observed,
        source=source,
        availability=availability,
        status=status,
    )


def determine_outcome(
    *,
    functional: FunctionalStatus,
    safety: SafetyStatus,
    record: RecordStatus,
    budgets: Iterable[BudgetResult],
) -> RunOutcome:
    """Combine independent axes with fail-closed safety and budget semantics."""

    budget_results = tuple(sorted(budgets, key=lambda budget: budget.name))
    names = [budget.name for budget in budget_results]
    if len(names) != len(set(names)):
        raise ValueError("budget names must be unique")

    failure_reasons: list[str] = []
    if functional is not FunctionalStatus.PASS:
        failure_reasons.append(f"functional:{functional.value}")
    if safety is not SafetyStatus.PASS:
        failure_reasons.append(f"safety:{safety.value}")
    if record is not RecordStatus.COMPLETE:
        failure_reasons.append(f"record:{record.value}")
    for budget in budget_results:
        if budget.status in {BudgetStatus.EXCEEDED, BudgetStatus.UNVERIFIABLE}:
            failure_reasons.append(f"budget:{budget.name}:{budget.status.value}")

    overall = OverallStatus.FAIL if failure_reasons else OverallStatus.PASS
    return RunOutcome(
        functional=functional,
        safety=safety,
        record=record,
        budgets=budget_results,
        overall=overall,
        failure_reasons=tuple(failure_reasons),
    )


def redact_and_truncate(
    text: str,
    *,
    secrets: Mapping[str, str],
    max_bytes: int,
) -> RetainedText:
    """Redact explicit fake or real secret values before UTF-8 truncation."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    normalized_secrets: list[tuple[str, str]] = []
    for label, value in secrets.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("secret label must be a non-empty string")
        if not isinstance(value, str) or not value:
            raise ValueError(f"secret {label!r} must not be empty")
        normalized_secrets.append((label, value))
    normalized_secrets.sort(key=lambda item: (-len(item[1]), item[0]))

    input_bytes = len(text.encode("utf-8"))
    redacted_text = text
    redactions: list[Redaction] = []
    for label, value in normalized_secrets:
        count = redacted_text.count(value)
        if count:
            redacted_text = redacted_text.replace(value, f"[REDACTED:{label}]")
            redactions.append(Redaction(label=label, count=count))

    redacted_payload = redacted_text.encode("utf-8")
    redacted_bytes = len(redacted_payload)
    redacted_sha256 = hashlib.sha256(redacted_payload).hexdigest()
    truncated = redacted_bytes > max_bytes
    if truncated:
        retained_text = redacted_payload[:max_bytes].decode("utf-8", errors="ignore")
    else:
        retained_text = redacted_text
    stored_bytes = len(retained_text.encode("utf-8"))

    return RetainedText(
        text=retained_text,
        input_bytes=input_bytes,
        redacted_bytes=redacted_bytes,
        stored_bytes=stored_bytes,
        truncated=truncated,
        redactions=tuple(redactions),
        redacted_sha256=redacted_sha256,
    )


def validate_retained_text(
    document: Mapping[str, Any],
    *,
    max_bytes: int,
    field: str,
) -> RetainedText:
    """Validate persisted redaction and truncation metadata."""

    if not isinstance(document, Mapping):
        raise RunRecordValidationError(f"{field} must be an object")
    text = _required_string_value(document, "text", f"{field}.text", allow_empty=True)
    input_bytes = _required_non_negative_int(document, "input_bytes", field)
    redacted_bytes = _required_non_negative_int(document, "redacted_bytes", field)
    stored_bytes = _required_non_negative_int(document, "stored_bytes", field)
    truncated = _required_bool(document, "truncated", field)
    redacted_sha256 = _required_string_value(
        document,
        "redacted_sha256",
        f"{field}.redacted_sha256",
    )
    if len(redacted_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in redacted_sha256
    ):
        raise RunRecordValidationError(
            f"{field}.redacted_sha256 must be a lowercase SHA-256 hex digest"
        )

    actual_stored_bytes = len(text.encode("utf-8"))
    if stored_bytes != actual_stored_bytes:
        raise RunRecordValidationError(
            f"{field}.stored_bytes does not match persisted text"
        )
    if stored_bytes > max_bytes:
        raise RunRecordValidationError(f"{field}.stored_bytes exceeds retention limit")
    if truncated and redacted_bytes <= stored_bytes:
        raise RunRecordValidationError(
            f"{field}.truncated requires redacted_bytes greater than stored_bytes"
        )
    if not truncated and redacted_bytes != stored_bytes:
        raise RunRecordValidationError(
            f"{field}.redacted_bytes must equal stored_bytes when not truncated"
        )
    if not truncated:
        actual_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if redacted_sha256 != actual_digest:
            raise RunRecordValidationError(
                f"{field}.redacted_sha256 does not match persisted text"
            )

    redaction_documents = _required_list(document, "redactions", f"{field}.redactions")
    redactions: list[Redaction] = []
    labels: set[str] = set()
    for index, item in enumerate(redaction_documents):
        item_field = f"{field}.redactions[{index}]"
        if not isinstance(item, Mapping):
            raise RunRecordValidationError(f"{item_field} must be an object")
        label = _required_string_value(item, "label", f"{item_field}.label")
        count = _required_positive_int(item, "count", item_field)
        if label in labels:
            raise RunRecordValidationError(f"{field}.redactions labels must be unique")
        labels.add(label)
        redactions.append(Redaction(label=label, count=count))

    return RetainedText(
        text=text,
        input_bytes=input_bytes,
        redacted_bytes=redacted_bytes,
        stored_bytes=stored_bytes,
        truncated=truncated,
        redactions=tuple(redactions),
        redacted_sha256=redacted_sha256,
    )


def load_run_record(path: str | Path) -> RunRecord:
    """Load one UTF-8 JSON run record and validate its internal evidence."""

    record_path = Path(path)
    try:
        document = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunRecordValidationError(
            f"cannot read run record {record_path}: {error}"
        ) from error
    return validate_run_record(document)


def validate_run_record(document: object) -> RunRecord:
    """Validate run-record v1 fields and recompute its declared result."""

    if not isinstance(document, dict):
        raise RunRecordValidationError("run record root must be an object")
    schema = _required_string(document, "schema", "schema")
    if schema != RUN_RECORD_SCHEMA:
        raise RunRecordValidationError(
            f"schema must be {RUN_RECORD_SCHEMA!r}, got {schema!r}"
        )

    run = _required_mapping(document, "run", "run")
    _required_string(run, "id", "run.id")
    _validate_identity(_required_mapping(run, "task", "run.task"), "run.task")
    _validate_identity(_required_mapping(run, "grader", "run.grader"), "run.grader")
    code = _required_mapping(run, "code", "run.code")
    _required_string(code, "revision", "run.code.revision")
    working_tree = _required_string(code, "working_tree", "run.code.working_tree")
    if working_tree not in {"clean", "dirty"}:
        raise RunRecordValidationError("run.code.working_tree must be clean or dirty")
    model = _required_mapping(run, "model", "run.model")
    if "provider" not in model:
        raise RunRecordValidationError("run.model.provider is required")
    provider = model["provider"]
    if provider is not None and (not isinstance(provider, str) or not provider.strip()):
        raise RunRecordValidationError(
            "run.model.provider must be null or a non-empty string"
        )
    _required_string(model, "name", "run.model.name")
    model_config = _required_mapping(model, "config", "run.model.config")
    _validate_no_sensitive_keys(model_config, "run.model.config")
    environment = _required_mapping(run, "environment", "run.environment")
    _validate_no_sensitive_keys(environment, "run.environment")
    started_at = _required_datetime(run, "started_at", "run.started_at")
    finished_at = _required_datetime(run, "finished_at", "run.finished_at")
    if finished_at < started_at:
        raise RunRecordValidationError("run.finished_at must not precede run.started_at")

    retention = _required_mapping(document, "retention_policy", "retention_policy")
    max_field_bytes = _required_positive_int(
        retention,
        "max_output_bytes_per_field",
        "retention_policy",
    )
    max_record_bytes = _required_positive_int(
        retention,
        "max_record_bytes",
        "retention_policy",
    )
    redaction = _required_string(
        retention,
        "redaction",
        "retention_policy.redaction",
    )
    if redaction != "explicit_secrets":
        raise RunRecordValidationError(
            "retention_policy.redaction must be 'explicit_secrets'"
        )

    budget_documents = _required_mapping(document, "budgets", "budgets")
    if set(budget_documents) != set(REQUIRED_BUDGETS):
        raise RunRecordValidationError(
            "budgets must contain exactly: " + ", ".join(REQUIRED_BUDGETS)
        )
    budgets = tuple(
        _validate_budget_document(name, budget_documents[name])
        for name in REQUIRED_BUDGETS
    )

    messages = _required_list(document, "messages", "messages")
    for index, message in enumerate(messages):
        field = f"messages[{index}]"
        if not isinstance(message, Mapping):
            raise RunRecordValidationError(f"{field} must be an object")
        _required_string_value(message, "role", f"{field}.role")
        validate_retained_text(
            _required_mapping_value(message, "content", f"{field}.content"),
            max_bytes=max_field_bytes,
            field=f"{field}.content",
        )

    tool_calls = _required_list(document, "tool_calls", "tool_calls")
    for index, call in enumerate(tool_calls):
        field = f"tool_calls[{index}]"
        if not isinstance(call, Mapping):
            raise RunRecordValidationError(f"{field} must be an object")
        _required_string_value(call, "id", f"{field}.id")
        _required_string_value(call, "tool", f"{field}.tool")
        validate_retained_text(
            _required_mapping_value(call, "arguments", f"{field}.arguments"),
            max_bytes=max_field_bytes,
            field=f"{field}.arguments",
        )
        _required_int(call, "exit_code", field)
        validate_retained_text(
            _required_mapping_value(call, "stdout", f"{field}.stdout"),
            max_bytes=max_field_bytes,
            field=f"{field}.stdout",
        )
        validate_retained_text(
            _required_mapping_value(call, "stderr", f"{field}.stderr"),
            max_bytes=max_field_bytes,
            field=f"{field}.stderr",
        )

    tests = _required_list(document, "tests", "tests")
    for index, test in enumerate(tests):
        field = f"tests[{index}]"
        if not isinstance(test, Mapping):
            raise RunRecordValidationError(f"{field} must be an object")
        _required_string_value(test, "name", f"{field}.name")
        status = _required_string_value(test, "status", f"{field}.status")
        if status not in {"passed", "failed", "error", "skipped"}:
            raise RunRecordValidationError(f"{field}.status is invalid")
        _required_int(test, "exit_code", field)

    diff = _required_mapping(document, "diff", "diff")
    _required_string_list(diff, "files_changed", "diff.files_changed")
    _required_bool(diff, "within_allowed_paths", "diff")

    termination = _required_mapping(document, "termination", "termination")
    _required_string(termination, "reason", "termination.reason")
    _required_non_negative_number(termination, "duration_ms", "termination")

    usage = _required_mapping(document, "usage", "usage")
    _validate_optional_non_negative_int(usage, "input_tokens", "usage.input_tokens")
    _validate_optional_non_negative_int(usage, "output_tokens", "usage.output_tokens")
    _validate_optional_non_negative_number(usage, "cost_usd", "usage.cost_usd")
    if "source" not in usage:
        raise RunRecordValidationError("usage.source is required")
    usage_source = usage["source"]
    if usage_source is not None and (
        not isinstance(usage_source, str) or not usage_source.strip()
    ):
        raise RunRecordValidationError(
            "usage.source must be null or a non-empty string"
        )

    results = _required_mapping(document, "results", "results")
    functional = _parse_enum(
        FunctionalStatus,
        _required_string(results, "functional", "results.functional"),
        "results.functional",
    )
    safety = _parse_enum(
        SafetyStatus,
        _required_string(results, "safety", "results.safety"),
        "results.safety",
    )
    record_status = _parse_enum(
        RecordStatus,
        _required_string(results, "record", "results.record"),
        "results.record",
    )
    declared_overall = _parse_enum(
        OverallStatus,
        _required_string(results, "overall", "results.overall"),
        "results.overall",
    )
    declared_reasons = tuple(
        _required_string_list(
            results,
            "failure_reasons",
            "results.failure_reasons",
        )
    )
    outcome = determine_outcome(
        functional=functional,
        safety=safety,
        record=record_status,
        budgets=budgets,
    )
    if (
        declared_overall is not outcome.overall
        or declared_reasons != outcome.failure_reasons
    ):
        raise RunRecordValidationError(
            "results do not match functional, safety, record, and budget evidence"
        )

    canonical_size = len(
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if canonical_size > max_record_bytes:
        raise RunRecordValidationError(
            "run record exceeds retention_policy.max_record_bytes"
        )

    return RunRecord(document=dict(document), budgets=budgets, outcome=outcome)


def _validate_budget_document(name: str, document: object) -> BudgetResult:
    if not isinstance(document, Mapping):
        raise RunRecordValidationError(f"budgets.{name} must be an object")
    for key in ("limit", "observed", "source"):
        if key not in document:
            raise RunRecordValidationError(f"budgets.{name}.{key} is required")
    limit = document["limit"]
    observed = document["observed"]
    source = document["source"]
    if name != "cost_usd" and (
        limit is None or not _is_number(limit) or limit <= 0
    ):
        raise RunRecordValidationError(
            f"budgets.{name}.limit must be a finite positive number"
        )
    try:
        result = evaluate_budget(
            name,
            limit=limit,
            observed=observed,
            source=source,
        )
    except (TypeError, ValueError) as error:
        raise RunRecordValidationError(f"budgets.{name}: {error}") from error
    declared_availability = _parse_enum(
        BudgetAvailability,
        _required_string_value(
            document,
            "availability",
            f"budgets.{name}.availability",
        ),
        f"budgets.{name}.availability",
    )
    declared_status = _parse_enum(
        BudgetStatus,
        _required_string_value(document, "status", f"budgets.{name}.status"),
        f"budgets.{name}.status",
    )
    if declared_availability is not result.availability:
        raise RunRecordValidationError(
            f"budgets.{name}.availability does not match limit and observation"
        )
    if declared_status is not result.status:
        raise RunRecordValidationError(
            f"budgets.{name}.status does not match limit and observation"
        )
    return result


def _validate_no_sensitive_keys(value: object, field: str) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise RunRecordValidationError(f"{field} keys must be strings")
            nested_field = f"{field}.{key}"
            normalized_key = _normalize_configuration_key(key)
            key_parts = set(normalized_key.split("_"))
            if any(
                normalized_key == suffix
                or normalized_key.endswith(f"_{suffix}")
                for suffix in SENSITIVE_KEY_SUFFIXES
            ) or key_parts.intersection(SENSITIVE_KEY_PARTS):
                raise RunRecordValidationError(
                    f"{nested_field} uses an obvious sensitive key"
                )
            _validate_no_sensitive_keys(nested_value, nested_field)
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _validate_no_sensitive_keys(nested_value, f"{field}[{index}]")


def _normalize_configuration_key(key: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^a-zA-Z0-9]+", "_", separated).strip("_").lower()


def _validate_identity(document: Mapping[str, Any], field: str) -> None:
    _required_string_value(document, "id", f"{field}.id")
    _required_string_value(document, "version", f"{field}.version")


def _validate_optional_number(value: object, field: str) -> None:
    if value is None:
        return
    if not _is_non_negative_number(value):
        raise ValueError(f"{field} must be null or a finite non-negative number")


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_non_negative_number(value: object) -> bool:
    return _is_number(value) and value >= 0


def _required_mapping(
    document: dict[str, Any], key: str, field: str
) -> dict[str, Any]:
    if key not in document:
        raise RunRecordValidationError(f"{field} is required")
    value = document[key]
    if not isinstance(value, dict):
        raise RunRecordValidationError(f"{field} must be an object")
    return value


def _required_mapping_value(
    document: Mapping[str, Any], key: str, field: str
) -> Mapping[str, Any]:
    if key not in document:
        raise RunRecordValidationError(f"{field} is required")
    value = document[key]
    if not isinstance(value, Mapping):
        raise RunRecordValidationError(f"{field} must be an object")
    return value


def _required_list(document: Mapping[str, Any], key: str, field: str) -> list[Any]:
    if key not in document:
        raise RunRecordValidationError(f"{field} is required")
    value = document[key]
    if not isinstance(value, list):
        raise RunRecordValidationError(f"{field} must be an array")
    return value


def _required_string(document: dict[str, Any], key: str, field: str) -> str:
    return _required_string_value(document, key, field)


def _required_string_value(
    document: Mapping[str, Any],
    key: str,
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    if key not in document:
        raise RunRecordValidationError(f"{field} is required")
    value = document[key]
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise RunRecordValidationError(f"{field} must be {qualifier}")
    return value


def _required_string_list(
    document: Mapping[str, Any], key: str, field: str
) -> list[str]:
    values = _required_list(document, key, field)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise RunRecordValidationError(f"{field} must contain non-empty strings")
    return values


def _required_bool(document: Mapping[str, Any], key: str, field: str) -> bool:
    if key not in document:
        raise RunRecordValidationError(f"{field}.{key} is required")
    value = document[key]
    if not isinstance(value, bool):
        raise RunRecordValidationError(f"{field}.{key} must be a boolean")
    return value


def _required_int(document: Mapping[str, Any], key: str, field: str) -> int:
    if key not in document:
        raise RunRecordValidationError(f"{field}.{key} is required")
    value = document[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunRecordValidationError(f"{field}.{key} must be an integer")
    return value


def _required_non_negative_int(
    document: Mapping[str, Any], key: str, field: str
) -> int:
    value = _required_int(document, key, field)
    if value < 0:
        raise RunRecordValidationError(f"{field}.{key} must be non-negative")
    return value


def _required_positive_int(
    document: Mapping[str, Any], key: str, field: str
) -> int:
    value = _required_int(document, key, field)
    if value <= 0:
        raise RunRecordValidationError(f"{field}.{key} must be positive")
    return value


def _required_non_negative_number(
    document: Mapping[str, Any], key: str, field: str
) -> int | float:
    if key not in document:
        raise RunRecordValidationError(f"{field}.{key} is required")
    value = document[key]
    if not _is_non_negative_number(value):
        raise RunRecordValidationError(
            f"{field}.{key} must be a finite non-negative number"
        )
    return value


def _validate_optional_non_negative_int(
    document: Mapping[str, Any], key: str, field: str
) -> None:
    if key not in document:
        raise RunRecordValidationError(f"{field} is required")
    value = document[key]
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise RunRecordValidationError(f"{field} must be null or non-negative")


def _validate_optional_non_negative_number(
    document: Mapping[str, Any], key: str, field: str
) -> None:
    if key not in document:
        raise RunRecordValidationError(f"{field} is required")
    value = document[key]
    if value is not None and not _is_non_negative_number(value):
        raise RunRecordValidationError(
            f"{field} must be null or a finite non-negative number"
        )


def _required_datetime(
    document: Mapping[str, Any], key: str, field: str
) -> datetime:
    value = _required_string_value(document, key, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunRecordValidationError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise RunRecordValidationError(f"{field} must include a timezone")
    return parsed


def _parse_enum(enum_type: type[StrEnum], value: str, field: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise RunRecordValidationError(f"{field} must be one of: {allowed}") from error
