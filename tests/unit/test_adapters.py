import pytest

from coding_agent.adapters import adapt_openai_response
from coding_agent.protocol import ModelResponse, ResponseKind, ToolCall


# Fixed representative samples of the OpenAI Responses API `response.output`
# format (pinned per the M1.3 plan; the adapter only reads the `output` list).
OPENAI_FINAL_TEXT = {
    "id": "resp_abc123",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_abc123",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Hello!", "annotations": []},
            ],
        }
    ],
}

OPENAI_FUNCTION_CALL = {
    "id": "resp_def456",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "function_call",
            "id": "fc_abc123",
            "call_id": "call_abc123",
            "name": "get_weather",
            "arguments": '{"city": "San Francisco"}',
        }
    ],
}


def test_adapts_final_text_response() -> None:
    result = adapt_openai_response(OPENAI_FINAL_TEXT)

    assert result == ModelResponse(kind=ResponseKind.FINAL, content="Hello!")


def test_concatenates_multiple_output_text_blocks() -> None:
    raw = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "A"},
                    {"type": "output_text", "text": "B"},
                ],
            }
        ]
    }

    result = adapt_openai_response(raw)

    assert result == ModelResponse(kind=ResponseKind.FINAL, content="AB")


def test_adapts_single_function_call() -> None:
    result = adapt_openai_response(OPENAI_FUNCTION_CALL)

    assert result == ModelResponse(
        kind=ResponseKind.TOOL_CALLS,
        tool_calls=(
            ToolCall(id="fc_abc123", name="get_weather", arguments={"city": "San Francisco"}),
        ),
    )


def test_adapts_multiple_function_calls_in_order() -> None:
    raw = {
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "name": "read_file",
                "arguments": '{"path": "a.txt"}',
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "name": "read_file",
                "arguments": '{"path": "b.txt"}',
            },
        ]
    }

    result = adapt_openai_response(raw)

    assert result.kind is ResponseKind.TOOL_CALLS
    assert result.tool_calls == (
        ToolCall(id="fc_1", name="read_file", arguments={"path": "a.txt"}),
        ToolCall(id="fc_2", name="read_file", arguments={"path": "b.txt"}),
    )


def test_missing_output_raises_value_error() -> None:
    with pytest.raises(ValueError, match="output"):
        adapt_openai_response({"id": "resp_1"})


def test_function_call_with_invalid_json_arguments_raises() -> None:
    raw = {
        "output": [
            {"type": "function_call", "id": "fc_1", "name": "f", "arguments": "not json"}
        ]
    }

    with pytest.raises(ValueError, match="JSON"):
        adapt_openai_response(raw)


def test_function_call_arguments_decoding_to_non_object_raises() -> None:
    raw = {
        "output": [
            {"type": "function_call", "id": "fc_1", "name": "f", "arguments": "[1, 2, 3]"}
        ]
    }

    with pytest.raises(ValueError, match="object"):
        adapt_openai_response(raw)


def test_function_call_missing_name_raises() -> None:
    raw = {
        "output": [
            {"type": "function_call", "id": "fc_1", "arguments": "{}"}
        ]
    }

    with pytest.raises(ValueError, match="name"):
        adapt_openai_response(raw)


def test_response_with_no_text_or_calls_raises() -> None:
    raw = {"output": []}

    with pytest.raises(ValueError, match="text or function calls"):
        adapt_openai_response(raw)
