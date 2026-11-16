import json
from pathlib import Path

import pytest

from coding_agent.evals.results import (
    BudgetAvailability,
    BudgetStatus,
    FunctionalStatus,
    OverallStatus,
    RecordStatus,
    RunRecordValidationError,
    SafetyStatus,
    determine_outcome,
    evaluate_budget,
    load_run_record,
    validate_run_record,
)


def test_budget_without_limit_is_not_applicable() -> None:
    result = evaluate_budget(
        "cost_usd",
        limit=None,
        observed=None,
        source=None,
    )

    assert result.status is BudgetStatus.NOT_APPLICABLE
    assert result.availability is BudgetAvailability.NOT_APPLICABLE


def test_configured_budget_without_observation_is_unverifiable() -> None:
    result = evaluate_budget(
        "cost_usd",
        limit=1.0,
        observed=None,
        source=None,
    )

    assert result.status is BudgetStatus.UNVERIFIABLE
    assert result.availability is BudgetAvailability.UNAVAILABLE


def test_budget_observation_is_compared_with_limit() -> None:
    within = evaluate_budget("steps", limit=10, observed=10, source="runner")
    exceeded = evaluate_budget("steps", limit=10, observed=11, source="runner")

    assert within.status is BudgetStatus.WITHIN
    assert within.availability is BudgetAvailability.AVAILABLE
    assert exceeded.status is BudgetStatus.EXCEEDED


def test_all_required_axes_and_budgets_pass() -> None:
    outcome = determine_outcome(
        functional=FunctionalStatus.PASS,
        safety=SafetyStatus.PASS,
        record=RecordStatus.COMPLETE,
        budgets=(
            evaluate_budget("steps", limit=10, observed=3, source="runner"),
            evaluate_budget(
                "cost_usd",
                limit=None,
                observed=None,
                source=None,
            ),
        ),
    )

    assert outcome.overall is OverallStatus.PASS
    assert outcome.failure_reasons == ()


@pytest.mark.parametrize(
    ("functional", "safety", "record", "expected_reason"),
    [
        (
            FunctionalStatus.FAIL,
            SafetyStatus.PASS,
            RecordStatus.COMPLETE,
            "functional:fail",
        ),
        (
            FunctionalStatus.ERROR,
            SafetyStatus.PASS,
            RecordStatus.COMPLETE,
            "functional:error",
        ),
        (
            FunctionalStatus.PASS,
            SafetyStatus.FAIL,
            RecordStatus.COMPLETE,
            "safety:fail",
        ),
        (
            FunctionalStatus.PASS,
            SafetyStatus.UNVERIFIABLE,
            RecordStatus.COMPLETE,
            "safety:unverifiable",
        ),
        (
            FunctionalStatus.PASS,
            SafetyStatus.PASS,
            RecordStatus.INCOMPLETE,
            "record:incomplete",
        ),
    ],
)
def test_required_axis_failure_closes_overall_result(
    functional: FunctionalStatus,
    safety: SafetyStatus,
    record: RecordStatus,
    expected_reason: str,
) -> None:
    outcome = determine_outcome(
        functional=functional,
        safety=safety,
        record=record,
        budgets=(),
    )

    assert outcome.overall is OverallStatus.FAIL
    assert expected_reason in outcome.failure_reasons


@pytest.mark.parametrize(
    ("budget", "expected_reason"),
    [
        (
            evaluate_budget("steps", limit=10, observed=11, source="runner"),
            "budget:steps:exceeded",
        ),
        (
            evaluate_budget("cost_usd", limit=1.0, observed=None, source=None),
            "budget:cost_usd:unverifiable",
        ),
    ],
)
def test_budget_failure_preserves_functional_pass_but_fails_overall(
    budget: object,
    expected_reason: str,
) -> None:
    outcome = determine_outcome(
        functional=FunctionalStatus.PASS,
        safety=SafetyStatus.PASS,
        record=RecordStatus.COMPLETE,
        budgets=(budget,),
    )

    assert outcome.functional is FunctionalStatus.PASS
    assert outcome.overall is OverallStatus.FAIL
    assert outcome.failure_reasons == (expected_reason,)


@pytest.mark.parametrize(
    ("filename", "overall", "failure_reason"),
    [
        ("passed.json", OverallStatus.PASS, None),
        (
            "budget-exceeded.json",
            OverallStatus.FAIL,
            "budget:steps:exceeded",
        ),
        ("safety-failed.json", OverallStatus.FAIL, "safety:fail"),
        (
            "record-incomplete.json",
            OverallStatus.FAIL,
            "record:incomplete",
        ),
    ],
)
def test_committed_run_record_examples_are_self_consistent(
    filename: str,
    overall: OverallStatus,
    failure_reason: str | None,
) -> None:
    record = load_run_record(Path("evals/examples/run-records") / filename)

    assert record.outcome.overall is overall
    if failure_reason is not None:
        assert failure_reason in record.outcome.failure_reasons


def test_run_record_rejects_declared_result_that_disagrees_with_evidence() -> None:
    path = Path("evals/examples/run-records/budget-exceeded.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["results"]["overall"] = "pass"
    document["results"]["failure_reasons"] = []

    with pytest.raises(RunRecordValidationError, match="results do not match"):
        validate_run_record(document)


def test_run_record_requires_explicit_budget_observation_fields() -> None:
    path = Path("evals/examples/run-records/passed.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["budgets"]["steps"]["source"]

    with pytest.raises(RunRecordValidationError, match=r"budgets\.steps\.source"):
        validate_run_record(document)


def test_run_record_rejects_budget_availability_mismatch() -> None:
    path = Path("evals/examples/run-records/passed.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["budgets"]["steps"]["availability"] = "unavailable"

    with pytest.raises(RunRecordValidationError, match="availability does not match"):
        validate_run_record(document)


def test_run_record_enforces_total_retention_limit() -> None:
    path = Path("evals/examples/run-records/passed.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["retention_policy"]["max_record_bytes"] = 1

    with pytest.raises(RunRecordValidationError, match="max_record_bytes"):
        validate_run_record(document)


def test_run_record_fails_closed_when_configured_cost_is_unavailable() -> None:
    path = Path("evals/examples/run-records/passed.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["budgets"]["cost_usd"] = {
        "limit": 1.0,
        "observed": None,
        "source": None,
        "availability": "unavailable",
        "status": "unverifiable",
    }
    document["results"] = {
        "functional": "pass",
        "safety": "pass",
        "record": "complete",
        "overall": "fail",
        "failure_reasons": ["budget:cost_usd:unverifiable"],
    }

    record = validate_run_record(document)

    assert record.outcome.functional is FunctionalStatus.PASS
    assert record.outcome.overall is OverallStatus.FAIL
    assert record.outcome.failure_reasons == ("budget:cost_usd:unverifiable",)
