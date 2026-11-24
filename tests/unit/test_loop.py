import pytest

from coding_agent.loop import LoopResult, StepLimitExceededError, StopReason, run
from coding_agent.model import ScriptExhaustedError, ScriptedModel
from coding_agent.protocol import (
    Message,
    MessageRole,
    ModelResponse,
    ProtocolValidationError,
    ResponseKind,
    ToolCall,
    ToolResult,
)
from coding_agent.tools import ToolNotFoundError, ToolParameterError, ToolRegistry


def _echo(text: str) -> str:
    return f"echo: {text}"


def _upper(text: str) -> str:
    return text.upper()


def _explode() -> str:
    raise RuntimeError("boom")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("echo", _echo)
    registry.register("upper", _upper)
    registry.register("explode", _explode)
    return registry


def _initial_messages() -> tuple[Message, ...]:
    return (
        Message(role=MessageRole.SYSTEM, content="You are a coding agent."),
        Message(role=MessageRole.USER, content="Do a task."),
    )


# --- M1.3-T1 normal path ---


def test_final_response_returns_result() -> None:
    model = ScriptedModel([ModelResponse(kind=ResponseKind.FINAL, content="done.")])

    result = run(model, _registry(), _initial_messages(), max_steps=1)

    assert result == LoopResult(
        messages=(
            Message(role=MessageRole.SYSTEM, content="You are a coding agent."),
            Message(role=MessageRole.USER, content="Do a task."),
            Message(role=MessageRole.ASSISTANT, content="done."),
        ),
        final_text="done.",
        stop_reason=StopReason.FINAL,
    )
    model.assert_complete()


def test_tool_call_round_executes_and_feeds_back() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(
                    ToolCall(id="call-1", name="echo", arguments={"text": "hello"}),
                ),
            ),
            ModelResponse(kind=ResponseKind.FINAL, content="all done."),
        ]
    )

    result = run(model, _registry(), _initial_messages(), max_steps=2)

    assert result.final_text == "all done."
    assert result.stop_reason is StopReason.FINAL
    assert result.messages[2].tool_calls[0].id == "call-1"
    assert result.messages[3].tool_result == ToolResult(
        tool_call_id="call-1",
        content="echo: hello",
        is_error=False,
    )
    assert result.messages[4] == Message(role=MessageRole.ASSISTANT, content="all done.")
    model.assert_complete()


def test_multiple_tool_calls_execute_in_order() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(
                    ToolCall(id="call-1", name="echo", arguments={"text": "a"}),
                    ToolCall(id="call-2", name="upper", arguments={"text": "b"}),
                ),
            ),
            ModelResponse(kind=ResponseKind.FINAL, content="finished."),
        ]
    )

    result = run(model, _registry(), _initial_messages(), max_steps=2)

    tool_result_ids = [
        message.tool_result.tool_call_id
        for message in result.messages
        if message.tool_result is not None
    ]
    assert tool_result_ids == ["call-1", "call-2"]
    assert result.messages[3].tool_result.content == "echo: a"
    assert result.messages[4].tool_result.content == "B"
    assert result.final_text == "finished."
    model.assert_complete()


def test_is_error_result_feeds_back_and_continues() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(
                    ToolCall(id="call-1", name="explode", arguments={}),
                ),
            ),
            ModelResponse(kind=ResponseKind.FINAL, content="recovered."),
        ]
    )

    result = run(model, _registry(), _initial_messages(), max_steps=2)

    assert result.final_text == "recovered."
    assert result.messages[3].tool_result.tool_call_id == "call-1"
    assert result.messages[3].tool_result.is_error is True
    model.assert_complete()


def test_multiple_rounds_before_final() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-1", name="echo", arguments={"text": "one"}),),
            ),
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-2", name="echo", arguments={"text": "two"}),),
            ),
            ModelResponse(kind=ResponseKind.FINAL, content="done."),
        ]
    )

    result = run(model, _registry(), _initial_messages(), max_steps=3)

    assert result.final_text == "done."
    assert len(result.messages) == 7
    assert result.messages[3].tool_result.tool_call_id == "call-1"
    assert result.messages[5].tool_result.tool_call_id == "call-2"
    model.assert_complete()


# --- M1.3-T2 budget and error propagation ---


def test_step_limit_exceeded_raises() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-1", name="echo", arguments={"text": "one"}),),
            ),
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-2", name="echo", arguments={"text": "two"}),),
            ),
            ModelResponse(kind=ResponseKind.FINAL, content="never reached."),
        ]
    )

    with pytest.raises(StepLimitExceededError) as exc_info:
        run(model, _registry(), _initial_messages(), max_steps=2)

    assert exc_info.value.max_steps == 2
    assert exc_info.value.steps == 2
    assert model.remaining_steps == 1  # the FINAL step was not consumed


def test_step_limit_executes_the_last_allowed_step_tools() -> None:
    executed: list[str] = []

    def record(value: str) -> str:
        executed.append(value)
        return value

    registry = ToolRegistry()
    registry.register("record", record)

    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-1", name="record", arguments={"value": "one"}),),
            ),
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-2", name="record", arguments={"value": "two"}),),
            ),
            ModelResponse(kind=ResponseKind.FINAL, content="never reached."),
        ]
    )

    with pytest.raises(StepLimitExceededError):
        run(model, registry, _initial_messages(), max_steps=2)

    assert executed == ["one", "two"]


@pytest.mark.parametrize("bad_max_steps", [0, -1, True, 1.5, "2"])
def test_invalid_max_steps_raises_value_error(bad_max_steps: object) -> None:
    model = ScriptedModel([ModelResponse(kind=ResponseKind.FINAL, content="unused.")])

    with pytest.raises(ValueError, match="max_steps"):
        run(model, _registry(), _initial_messages(), max_steps=bad_max_steps)


def test_tool_not_found_propagates() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-1", name="missing", arguments={}),),
            ),
        ]
    )

    with pytest.raises(ToolNotFoundError):
        run(model, _registry(), _initial_messages(), max_steps=2)


def test_parameter_error_propagates() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-1", name="echo", arguments={}),),
            ),
        ]
    )

    with pytest.raises(ToolParameterError):
        run(model, _registry(), _initial_messages(), max_steps=2)


def test_invalid_initial_history_propagates() -> None:
    model = ScriptedModel([ModelResponse(kind=ResponseKind.FINAL, content="unused.")])
    bad_messages = (
        Message(role=MessageRole.SYSTEM, content="system"),
        Message(
            role=MessageRole.TOOL,
            tool_result=ToolResult(tool_call_id="missing-call", content="orphan"),
        ),
    )

    with pytest.raises(ProtocolValidationError):
        run(model, _registry(), bad_messages, max_steps=2)

    assert model.remaining_steps == 1  # the model was never called


def test_model_exception_propagates() -> None:
    model = ScriptedModel([RuntimeError("model failed")])

    with pytest.raises(RuntimeError, match="model failed"):
        run(model, _registry(), _initial_messages(), max_steps=2)


def test_script_exhausted_propagates() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                kind=ResponseKind.TOOL_CALLS,
                tool_calls=(ToolCall(id="call-1", name="echo", arguments={"text": "x"}),),
            ),
        ]
    )

    with pytest.raises(ScriptExhaustedError):
        run(model, _registry(), _initial_messages(), max_steps=5)
