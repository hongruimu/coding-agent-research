from coding_agent.loop import LoopResult, StopReason, run
from coding_agent.model import ScriptedModel
from coding_agent.protocol import (
    Message,
    MessageRole,
    ModelResponse,
    ResponseKind,
    ToolCall,
    ToolResult,
)
from coding_agent.tools import ToolRegistry


def _read_file(path: str) -> str:
    return f"content of {path}"


def _summarize(text: str) -> str:
    return f"summary: {text}"


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("read_file", _read_file)
    registry.register("summarize", _summarize)
    return registry


def _initial_messages() -> tuple[Message, ...]:
    return (
        Message(role=MessageRole.SYSTEM, content="You are a coding agent."),
        Message(role=MessageRole.USER, content="Read and summarize README.md."),
    )


def _script() -> list[ModelResponse]:
    return [
        ModelResponse(
            kind=ResponseKind.TOOL_CALLS,
            tool_calls=(
                ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"}),
            ),
        ),
        ModelResponse(
            kind=ResponseKind.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="call-2",
                    name="summarize",
                    arguments={"text": "content of README.md"},
                ),
            ),
        ),
        ModelResponse(kind=ResponseKind.FINAL, content="summary: content of README.md"),
    ]


def _run_once() -> tuple[LoopResult, ScriptedModel]:
    model = ScriptedModel(_script())
    result = run(model, _registry(), _initial_messages(), max_steps=3)
    return result, model


def test_run_drives_multi_round_closed_loop() -> None:
    result, model = _run_once()

    assert result.final_text == "summary: content of README.md"
    assert result.stop_reason is StopReason.FINAL
    assert result.messages == (
        Message(role=MessageRole.SYSTEM, content="You are a coding agent."),
        Message(role=MessageRole.USER, content="Read and summarize README.md."),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"}),),
        ),
        Message(
            role=MessageRole.TOOL,
            tool_result=ToolResult(tool_call_id="call-1", content="content of README.md"),
        ),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=(
                ToolCall(id="call-2", name="summarize", arguments={"text": "content of README.md"}),
            ),
        ),
        Message(
            role=MessageRole.TOOL,
            tool_result=ToolResult(
                tool_call_id="call-2", content="summary: content of README.md"
            ),
        ),
        Message(role=MessageRole.ASSISTANT, content="summary: content of README.md"),
    )

    # The first tool result was fed back into the second model call's input.
    assert model.calls[1][3].tool_result == ToolResult(
        tool_call_id="call-1", content="content of README.md"
    )
    model.assert_complete()


def test_same_inputs_produce_field_equal_result() -> None:
    first_result, first_model = _run_once()
    second_result, second_model = _run_once()

    assert first_result == second_result
    assert first_result.messages == second_result.messages
    assert first_result.final_text == second_result.final_text
    assert first_result.stop_reason == second_result.stop_reason

    first_model.assert_complete()
    second_model.assert_complete()
