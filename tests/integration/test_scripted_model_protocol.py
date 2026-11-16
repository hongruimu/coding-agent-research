import pytest

from coding_agent.model import ScriptedModel
from coding_agent.protocol import (
    Message,
    MessageRole,
    ModelResponse,
    ProtocolValidationError,
    ResponseKind,
    ToolCall,
    ToolResult,
    validate_message_history,
)


def _initial_messages() -> tuple[Message, ...]:
    return (
        Message(role=MessageRole.SYSTEM, content="You are a coding agent."),
        Message(role=MessageRole.USER, content="Read README.md."),
    )


def _tool_response(call_id: str = "call-1") -> ModelResponse:
    return ModelResponse(
        kind=ResponseKind.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id=call_id,
                name="read_file",
                arguments={"path": "README.md"},
            ),
        ),
    )


def _run_valid_two_round_trace() -> tuple[
    tuple[Message, ...],
    tuple[tuple[Message, ...], ...],
    ModelResponse,
]:
    model = ScriptedModel(
        [
            _tool_response(),
            ModelResponse(kind=ResponseKind.FINAL, content="README inspected."),
        ]
    )
    history = list(_initial_messages())

    tool_response = model.respond(history)
    history.append(tool_response.to_message())
    history.append(
        ToolResult(
            tool_call_id=tool_response.tool_calls[0].id,
            content="# Coding Agent",
        ).to_message()
    )
    validate_message_history(history)

    final_response = model.respond(history)
    history.append(final_response.to_message())
    validate_message_history(history)
    model.assert_complete()

    return tuple(history), model.calls, final_response


def test_correlated_tool_result_enters_the_next_model_round() -> None:
    history, calls, final_response = _run_valid_two_round_trace()

    assert final_response == ModelResponse(
        kind=ResponseKind.FINAL,
        content="README inspected.",
    )
    assert calls[0] == _initial_messages()
    assert calls[1] == history[:-1]
    assert history[2].tool_calls[0].id == "call-1"
    assert history[3].tool_result == ToolResult(
        tool_call_id="call-1",
        content="# Coding Agent",
    )


def test_the_same_script_produces_the_same_structured_trace() -> None:
    first_history, first_calls, first_response = _run_valid_two_round_trace()
    second_history, second_calls, second_response = _run_valid_two_round_trace()

    assert second_history == first_history
    assert second_calls == first_calls
    assert second_response == first_response


def test_wrong_result_id_fails_before_consuming_the_next_model_step() -> None:
    model = ScriptedModel(
        [
            _tool_response(),
            ModelResponse(kind=ResponseKind.FINAL, content="must remain unconsumed"),
        ]
    )
    history = list(_initial_messages())
    history.append(model.respond(history).to_message())
    history.append(
        ToolResult(tool_call_id="missing-call", content="wrong result").to_message()
    )

    with pytest.raises(ProtocolValidationError, match="unknown or not-yet-seen"):
        validate_message_history(history)

    assert len(model.calls) == 1
    assert model.remaining_steps == 1


def test_reusing_a_call_id_in_a_later_model_round_fails_history_validation() -> None:
    model = ScriptedModel([_tool_response(), _tool_response()])
    history = list(_initial_messages())
    first_response = model.respond(history)
    history.append(first_response.to_message())
    history.append(
        ToolResult(
            tool_call_id=first_response.tool_calls[0].id,
            content="# Coding Agent",
        ).to_message()
    )
    validate_message_history(history)

    history.append(model.respond(history).to_message())

    with pytest.raises(ProtocolValidationError, match="duplicate tool call id 'call-1'"):
        validate_message_history(history)

    assert len(model.calls) == 2
    assert model.remaining_steps == 0
