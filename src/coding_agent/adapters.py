"""Provider raw-output to ModelResponse adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping

from coding_agent.protocol import ModelResponse, ResponseKind, ToolCall


def adapt_openai_response(raw: Mapping[str, object]) -> ModelResponse:
    """Map an OpenAI Responses API response object to a ``ModelResponse``.

    ``raw["output"]`` is a list of items. ``function_call`` items become tool
    calls, and assistant ``message`` items contribute their ``output_text``
    blocks as final text. A response with at least one function call maps to
    ``TOOL_CALLS``; otherwise it maps to ``FINAL``.
    """

    output = raw.get("output")
    if not isinstance(output, list):
        raise ValueError("OpenAI response must contain an 'output' list")

    tool_calls: list[ToolCall] = []
    texts: list[str] = []

    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise ValueError(f"output[{index}] must be a mapping")
        item_type = item.get("type")
        if item_type == "function_call":
            tool_calls.append(_parse_function_call(item))
        elif item_type == "message":
            texts.extend(_message_texts(item))

    if tool_calls:
        return ModelResponse(kind=ResponseKind.TOOL_CALLS, tool_calls=tuple(tool_calls))

    content = "".join(texts)
    if not content.strip():
        raise ValueError("OpenAI response contains no text or function calls")
    return ModelResponse(kind=ResponseKind.FINAL, content=content)


def _parse_function_call(item: Mapping[str, object]) -> ToolCall:
    call_id = item.get("id")
    name = item.get("name")
    arguments_raw = item.get("arguments")

    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("function_call requires a non-blank string 'id'")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("function_call requires a non-blank string 'name'")
    if not isinstance(arguments_raw, str):
        raise ValueError("function_call requires a string 'arguments'")

    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"function_call arguments are not valid JSON: {exc}") from exc

    if not isinstance(arguments, dict):
        raise ValueError("function_call arguments must decode to a JSON object")

    return ToolCall(id=call_id, name=name, arguments=arguments)


def _message_texts(item: Mapping[str, object]) -> list[str]:
    content = item.get("content")
    if not isinstance(content, list):
        return []

    texts: list[str] = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "output_text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
    return texts


__all__ = ["adapt_openai_response"]
