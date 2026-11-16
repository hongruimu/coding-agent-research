import json
from pathlib import Path

import pytest

from coding_agent.evals.results import (
    RunRecordValidationError,
    redact_and_truncate,
    validate_retained_text,
    validate_run_record,
)


def test_secret_is_redacted_before_output_is_truncated() -> None:
    fake_secret = "sk-test-not-a-real-secret"
    retained = redact_and_truncate(
        f"prefix {fake_secret} suffix with additional output",
        secrets={"api_key": fake_secret},
        max_bytes=32,
    )
    serialized = json.dumps(retained.to_dict(), ensure_ascii=False)

    assert fake_secret not in retained.text
    assert fake_secret not in serialized
    assert retained.redactions[0].label == "api_key"
    assert retained.redactions[0].count == 1
    assert retained.truncated is True
    assert retained.stored_bytes <= 32
    assert retained.redacted_bytes > retained.stored_bytes


def test_utf8_truncation_never_persists_partial_character() -> None:
    retained = redact_and_truncate(
        "中文输出边界",
        secrets={},
        max_bytes=7,
    )

    assert retained.text == "中文"
    assert retained.stored_bytes == len(retained.text.encode("utf-8"))
    assert retained.truncated is True


def test_retained_text_metadata_detects_tampering() -> None:
    retained = redact_and_truncate("short output", secrets={}, max_bytes=64).to_dict()
    retained["stored_bytes"] += 1

    with pytest.raises(RunRecordValidationError, match="stored_bytes"):
        validate_retained_text(retained, max_bytes=64, field="stdout")


def test_empty_secret_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        redact_and_truncate(
            "output",
            secrets={"api_key": ""},
            max_bytes=64,
        )


def test_run_record_rejects_unretained_tool_arguments() -> None:
    document = json.loads(
        Path("evals/examples/run-records/passed.json").read_text(encoding="utf-8")
    )
    document["tool_calls"][0]["arguments"] = {}

    with pytest.raises(RunRecordValidationError, match=r"tool_calls\[0\]\.arguments"):
        validate_run_record(document)


def test_tool_arguments_can_be_redacted_before_retention() -> None:
    fake_secret = "fake-tool-secret"
    document = json.loads(
        Path("evals/examples/run-records/passed.json").read_text(encoding="utf-8")
    )
    document["tool_calls"][0]["arguments"] = redact_and_truncate(
        json.dumps({"query": fake_secret}),
        secrets={"query_secret": fake_secret},
        max_bytes=128,
    ).to_dict()

    serialized = json.dumps(document, ensure_ascii=False)
    assert fake_secret not in serialized
    validate_run_record(document)


@pytest.mark.parametrize(
    ("container", "sensitive_value", "expected_field"),
    [
        ("model", {"nested": {"OPENAI_API_KEY": "fake-key"}}, "run.model.config"),
        ("environment", {"service_access_token": "fake-token"}, "run.environment"),
    ],
)
def test_run_record_rejects_obvious_sensitive_configuration_keys(
    container: str,
    sensitive_value: dict[str, object],
    expected_field: str,
) -> None:
    document = json.loads(
        Path("evals/examples/run-records/passed.json").read_text(encoding="utf-8")
    )
    if container == "model":
        document["run"]["model"]["config"] = sensitive_value
    else:
        document["run"]["environment"] = sensitive_value

    with pytest.raises(RunRecordValidationError, match=expected_field):
        validate_run_record(document)


def test_non_sensitive_token_limit_key_remains_valid() -> None:
    document = json.loads(
        Path("evals/examples/run-records/passed.json").read_text(encoding="utf-8")
    )
    document["run"]["model"]["config"] = {"max_tokens": 1024}

    validate_run_record(document)
