from dataclasses import FrozenInstanceError, fields

import pytest

from coding_agent.protocol import (
    CORE_PROTOCOL_VERSION,
    Message,
    MessageRole,
    ModelResponse,
    ProtocolValidationError,
    ResponseKind,
    ToolCall,
    ToolResult,
    validate_message_history,
)


def _call(call_id: str = "call-1", name: str = "read_file") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"path": "README.md"})


def test_protocol_v1_freezes_the_four_structure_fields() -> None:
    assert CORE_PROTOCOL_VERSION == "coding-agent/core-protocol-v1"
    assert [field.name for field in fields(ToolCall)] == ["id", "name", "arguments"]
    assert [field.name for field in fields(ToolResult)] == [
        "tool_call_id",
        "content",
        "is_error",
    ]
    assert [field.name for field in fields(ModelResponse)] == [
        "kind",
        "content",
        "tool_calls",
    ]
    assert [field.name for field in fields(Message)] == [
        "role",
        "content",
        "tool_calls",
        "tool_result",
    ]


def test_protocol_structures_are_frozen_and_snapshot_arguments() -> None:
    arguments = {"path": "README.md", "options": ["number-lines"]}
    call = ToolCall(id="call-1", name="read_file", arguments=arguments)

    arguments["path"] = "changed.py"
    arguments["options"].append("raw")

    assert call.arguments == {"path": "README.md", "options": ["number-lines"]}
    with pytest.raises(FrozenInstanceError):
        call.name = "write_file"  # type: ignore[misc]


@pytest.mark.parametrize(("field", "value"), [("id", ""), ("id", "  "), ("name", "")])
def test_tool_call_requires_non_blank_identity(field: str, value: str) -> None:
    values: dict[str, object] = {
        "id": "call-1",
        "name": "read_file",
        "arguments": {},
    }
    values[field] = value

    with pytest.raises(ProtocolValidationError, match=field):
        ToolCall(**values)  # type: ignore[arg-type]


def test_tool_call_requires_an_arguments_mapping() -> None:
    with pytest.raises(ProtocolValidationError, match="arguments"):
        ToolCall(id="call-1", name="read_file", arguments=["README.md"])  # type: ignore[arg-type]


def test_tool_result_preserves_empty_and_error_outputs() -> None:
    empty = ToolResult(tool_call_id="call-1", content="")
    failed = ToolResult(tool_call_id="call-2", content="not found", is_error=True)

    assert empty.content == ""
    assert empty.is_error is False
    assert failed.is_error is True
    assert failed.to_message() == Message(role=MessageRole.TOOL, tool_result=failed)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tool_call_id": "", "content": "ok"}, "tool_call_id"),
        ({"tool_call_id": "call-1", "content": 3}, "content"),
        ({"tool_call_id": "call-1", "content": "ok", "is_error": 1}, "is_error"),
    ],
)
def test_tool_result_rejects_invalid_fields(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ProtocolValidationError, match=match):
        ToolResult(**kwargs)  # type: ignore[arg-type]


def test_model_response_requires_an_explicit_kind() -> None:
    with pytest.raises(TypeError):
        ModelResponse()  # type: ignore[call-arg]

    with pytest.raises(ProtocolValidationError, match="kind"):
        ModelResponse(kind="final", content="done")  # type: ignore[arg-type]


def test_model_response_final_branch_converts_to_an_assistant_message() -> None:
    response = ModelResponse(kind=ResponseKind.FINAL, content="done")

    assert response.to_message() == Message(role=MessageRole.ASSISTANT, content="done")


def test_model_response_tool_branch_converts_to_an_assistant_message() -> None:
    calls = (_call(), _call("call-2", "list_files"))
    response = ModelResponse(kind=ResponseKind.TOOL_CALLS, tool_calls=calls)

    assert response.to_message() == Message(
        role=MessageRole.ASSISTANT,
        tool_calls=calls,
    )


@pytest.mark.parametrize(
    "response",
    [
        lambda: ModelResponse(kind=ResponseKind.FINAL),
        lambda: ModelResponse(kind=ResponseKind.FINAL, content="   "),
        lambda: ModelResponse(
            kind=ResponseKind.FINAL,
            content="done",
            tool_calls=(_call(),),
        ),
        lambda: ModelResponse(kind=ResponseKind.TOOL_CALLS),
        lambda: ModelResponse(
            kind=ResponseKind.TOOL_CALLS,
            content="also text",
            tool_calls=(_call(),),
        ),
    ],
)
def test_model_response_rejects_ambiguous_or_empty_branches(response: object) -> None:
    with pytest.raises(ProtocolValidationError):
        response()  # type: ignore[operator]


def test_model_response_rejects_duplicate_call_ids() -> None:
    with pytest.raises(ProtocolValidationError, match="duplicate tool call id 'call-1'"):
        ModelResponse(
            kind=ResponseKind.TOOL_CALLS,
            tool_calls=(_call(), _call(name="list_files")),
        )


@pytest.mark.parametrize(
    "message",
    [
        lambda: Message(role=MessageRole.SYSTEM),
        lambda: Message(role=MessageRole.USER, content="  "),
        lambda: Message(role=MessageRole.USER, content="read", tool_calls=(_call(),)),
        lambda: Message(role=MessageRole.ASSISTANT),
        lambda: Message(
            role=MessageRole.ASSISTANT,
            content="done",
            tool_calls=(_call(),),
        ),
        lambda: Message(role=MessageRole.TOOL),
        lambda: Message(
            role=MessageRole.TOOL,
            content="duplicate payload",
            tool_result=ToolResult(tool_call_id="call-1", content="ok"),
        ),
    ],
)
def test_message_roles_reject_missing_or_mixed_payloads(message: object) -> None:
    with pytest.raises(ProtocolValidationError):
        message()  # type: ignore[operator]


def test_message_rejects_unknown_role_and_non_tuple_calls() -> None:
    with pytest.raises(ProtocolValidationError, match="role"):
        Message(role="user", content="hello")  # type: ignore[arg-type]

    with pytest.raises(ProtocolValidationError, match="tool_calls"):
        Message(role=MessageRole.ASSISTANT, tool_calls=[_call()])  # type: ignore[arg-type]


def test_history_accepts_correlated_results_and_pending_calls() -> None:
    first_call = _call()
    pending_call = _call("call-2", "list_files")
    result = ToolResult(tool_call_id=first_call.id, content="contents")
    history = (
        Message(role=MessageRole.SYSTEM, content="You are a coding agent."),
        Message(role=MessageRole.USER, content="Inspect the repository."),
        ModelResponse(
            kind=ResponseKind.TOOL_CALLS,
            tool_calls=(first_call, pending_call),
        ).to_message(),
        result.to_message(),
    )

    assert validate_message_history(history) is None


def test_history_rejects_duplicate_call_ids_across_responses() -> None:
    history = (
        Message(role=MessageRole.ASSISTANT, tool_calls=(_call(),)),
        Message(role=MessageRole.ASSISTANT, tool_calls=(_call(name="list_files"),)),
    )

    with pytest.raises(ProtocolValidationError, match="duplicate tool call id 'call-1'"):
        validate_message_history(history)


def test_history_rejects_unknown_or_not_yet_seen_results() -> None:
    result = ToolResult(tool_call_id="call-1", content="contents").to_message()

    with pytest.raises(ProtocolValidationError, match="unknown or not-yet-seen"):
        validate_message_history((result,))

    with pytest.raises(ProtocolValidationError, match="unknown or not-yet-seen"):
        validate_message_history((result, Message(role=MessageRole.ASSISTANT, tool_calls=(_call(),))))


def test_history_rejects_more_than_one_result_for_a_call() -> None:
    result = ToolResult(tool_call_id="call-1", content="contents").to_message()
    history = (
        Message(role=MessageRole.ASSISTANT, tool_calls=(_call(),)),
        result,
        result,
    )

    with pytest.raises(ProtocolValidationError, match="duplicate result for tool call 'call-1'"):
        validate_message_history(history)


def test_history_rejects_non_message_items() -> None:
    with pytest.raises(ProtocolValidationError, match=r"messages\[0\]"):
        validate_message_history(("not-a-message",))  # type: ignore[arg-type]
