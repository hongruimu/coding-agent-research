import pytest

from coding_agent.model import (
    Model,
    ScriptExhaustedError,
    ScriptedModel,
    UnconsumedScriptError,
)
from coding_agent.protocol import (
    Message,
    MessageRole,
    ModelResponse,
    ResponseKind,
    ToolCall,
)


def _user_message(content: str = "Inspect the repository.") -> Message:
    return Message(role=MessageRole.USER, content=content)


def _final(content: str) -> ModelResponse:
    return ModelResponse(kind=ResponseKind.FINAL, content=content)


def test_scripted_model_implements_the_minimal_model_protocol() -> None:
    model: Model = ScriptedModel([_final("done")])

    assert isinstance(model, Model)
    assert model.respond((_user_message(),)) == _final("done")


def test_scripted_model_returns_steps_in_fifo_order_and_records_each_call() -> None:
    first = _final("first")
    second = _final("second")
    model = ScriptedModel([first, second])
    first_input = (_user_message("first input"),)
    second_input = (_user_message("second input"),)

    assert model.respond(first_input) == first
    assert model.remaining_steps == 1
    assert model.respond(second_input) == second

    assert model.calls == (first_input, second_input)
    assert model.remaining_steps == 0


def test_scripted_model_snapshots_scripted_responses_at_construction() -> None:
    call = ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})
    configured = ModelResponse(kind=ResponseKind.TOOL_CALLS, tool_calls=(call,))
    model = ScriptedModel([configured])

    call.arguments["path"] = "changed.py"  # type: ignore[index]

    returned = model.respond((_user_message(),))
    assert returned.tool_calls[0].arguments == {"path": "README.md"}
    assert returned is not configured


def test_scripted_model_call_records_are_detached_snapshots() -> None:
    call = ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})
    input_message = Message(role=MessageRole.ASSISTANT, tool_calls=(call,))
    messages = [input_message]
    model = ScriptedModel([_final("done")])

    model.respond(messages)
    messages.clear()
    call.arguments["path"] = "changed.py"  # type: ignore[index]

    first_view = model.calls
    assert first_view[0][0].tool_calls[0].arguments == {"path": "README.md"}
    first_view[0][0].tool_calls[0].arguments["path"] = "mutated-view.py"  # type: ignore[index]
    assert model.calls[0][0].tool_calls[0].arguments == {"path": "README.md"}


def test_scripted_model_raises_the_configured_exception_after_recording_the_call() -> None:
    error = RuntimeError("provider unavailable")
    message = _user_message()
    model = ScriptedModel([error])

    with pytest.raises(RuntimeError, match="provider unavailable") as caught:
        model.respond((message,))

    assert caught.value is error
    assert model.calls == ((message,),)
    assert model.remaining_steps == 0


def test_scripted_model_exhaustion_is_explicit_and_records_the_unexpected_call() -> None:
    message = _user_message()
    model = ScriptedModel([])

    with pytest.raises(ScriptExhaustedError, match="call 0") as caught:
        model.respond((message,))

    assert caught.value.call_index == 0
    assert model.calls == ((message,),)
    assert model.remaining_steps == 0


def test_scripted_model_assert_complete_reports_unconsumed_steps() -> None:
    model = ScriptedModel([_final("first"), _final("second")])

    model.respond((_user_message(),))

    with pytest.raises(UnconsumedScriptError, match="1 unconsumed") as caught:
        model.assert_complete()

    assert caught.value.remaining_steps == 1
    model.respond((_user_message("continue"),))
    assert model.assert_complete() is None


def test_scripted_model_rejects_invalid_steps_before_any_call() -> None:
    with pytest.raises(TypeError, match=r"steps\[1\]"):
        ScriptedModel([_final("valid"), object()])  # type: ignore[list-item]


def test_empty_message_sequence_is_recorded_without_hidden_policy() -> None:
    model = ScriptedModel([_final("done")])

    assert model.respond(()) == _final("done")
    assert model.calls == ((),)
