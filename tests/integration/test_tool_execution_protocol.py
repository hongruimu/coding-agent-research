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
from coding_agent.tools import ToolRegistry


def _read_file(path: str) -> str:
    return f"content of {path}"


def _explode() -> str:
    raise RuntimeError("boom")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("read_file", _read_file)
    registry.register("explode", _explode)
    return registry


def _initial_messages() -> tuple[Message, ...]:
    return (
        Message(role=MessageRole.SYSTEM, content="You are a coding agent."),
        Message(role=MessageRole.USER, content="Read README.md."),
    )


def _tool_call_response() -> ModelResponse:
    return ModelResponse(
        kind=ResponseKind.TOOL_CALLS,
        tool_calls=(
            ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"}),
        ),
    )


def _run_trace() -> tuple[
    tuple[Message, ...],
    tuple[tuple[Message, ...], ...],
    ModelResponse,
]:
    model = ScriptedModel(
        [
            _tool_call_response(),
            ModelResponse(kind=ResponseKind.FINAL, content="README inspected."),
        ]
    )
    registry = _registry()
    history = list(_initial_messages())

    response = model.respond(history)
    history.append(response.to_message())
    for tool_call in response.tool_calls:
        history.append(registry.execute(tool_call).to_message())
    validate_message_history(history)

    final_response = model.respond(history)
    history.append(final_response.to_message())
    validate_message_history(history)
    model.assert_complete()

    return tuple(history), model.calls, final_response


def test_executed_tool_result_enters_the_next_model_round() -> None:
    history, calls, final_response = _run_trace()

    assert final_response == ModelResponse(
        kind=ResponseKind.FINAL,
        content="README inspected.",
    )
    assert history[2].tool_calls[0].id == "call-1"
    assert history[3].tool_result == ToolResult(
        tool_call_id="call-1",
        content="content of README.md",
        is_error=False,
    )
    assert calls[0] == _initial_messages()
    assert calls[1] == history[:-1]


def test_the_same_script_and_registry_produce_the_same_trace() -> None:
    first_history, first_calls, first_response = _run_trace()
    second_history, second_calls, second_response = _run_trace()

    assert second_history == first_history
    assert second_calls == first_calls
    assert second_response == first_response


def test_tool_execution_failure_stays_as_correlated_error_data() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-2", name="explode", arguments={}),),
            ),
            ModelResponse(kind=ResponseKind.FINAL, content="handled failure."),
        ]
    )
    registry = _registry()
    history = list(_initial_messages())

    response = model.respond(history)
    history.append(response.to_message())
    result = registry.execute(response.tool_calls[0])
    assert result.is_error is True
    history.append(result.to_message())
    validate_message_history(history)

    final_response = model.respond(history)
    history.append(final_response.to_message())
    model.assert_complete()

    assert final_response.content == "handled failure."
    assert history[3].tool_result.tool_call_id == "call-2"


def test_wrong_result_id_fails_history_validation() -> None:
    model = ScriptedModel(
        [
            _tool_call_response(),
            ModelResponse(kind=ResponseKind.FINAL, content="must remain unconsumed"),
        ]
    )
    history = list(_initial_messages())
    history.append(model.respond(history).to_message())
    history.append(ToolResult(tool_call_id="wrong-id", content="misattached").to_message())

    with pytest.raises(ProtocolValidationError, match="unknown or not-yet-seen"):
        validate_message_history(history)

    assert len(model.calls) == 1
    assert model.remaining_steps == 1
