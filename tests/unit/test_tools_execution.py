from typing import Optional

import pytest

from coding_agent.protocol import ProtocolValidationError, ToolCall, ToolResult
from coding_agent.tools import (
    ToolNotFoundError,
    ToolParameterError,
    ToolRegistry,
)


def _add(a: int, b: int = 0) -> str:
    return str(a + b)


def _nullable(a: Optional[str]) -> str:
    return "" if a is None else a


def _to_float(x: float) -> str:
    return str(x)


def _to_bool(x: bool) -> str:
    return str(x)


def _explodes() -> str:
    raise ValueError("boom")


def _returns_non_str():
    return 42


def _call(name: str, arguments: dict[str, object], call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def test_execute_unknown_tool_raises_not_found() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError, match="missing"):
        registry.execute(_call("missing", {}))


def test_tool_errors_are_protocol_validation_errors() -> None:
    assert issubclass(ToolNotFoundError, ProtocolValidationError)
    assert issubclass(ToolParameterError, ProtocolValidationError)


def test_execute_success_returns_content_and_is_not_an_error() -> None:
    registry = ToolRegistry()
    registry.register("add", _add)

    result = registry.execute(_call("add", {"a": 2}))

    assert isinstance(result, ToolResult)
    assert result.tool_call_id == "call-1"
    assert result.content == "2"
    assert result.is_error is False


def test_execute_optional_parameter_uses_function_default() -> None:
    registry = ToolRegistry()
    registry.register("add", _add)

    assert registry.execute(_call("add", {"a": 3, "b": 4})).content == "7"
    assert registry.execute(_call("add", {"a": 3})).content == "3"


def test_execute_nullable_parameter_accepts_none() -> None:
    registry = ToolRegistry()
    registry.register("nullable", _nullable)

    assert registry.execute(_call("nullable", {"a": None})).is_error is False
    assert registry.execute(_call("nullable", {"a": "x"})).content == "x"


def test_execute_float_accepts_int_and_float_but_not_bool() -> None:
    registry = ToolRegistry()
    registry.register("to_float", _to_float)

    assert registry.execute(_call("to_float", {"x": 1})).is_error is False
    assert registry.execute(_call("to_float", {"x": 1.0})).is_error is False

    with pytest.raises(ToolParameterError, match="expected float, got bool"):
        registry.execute(_call("to_float", {"x": True}))


def test_execute_bool_accepts_only_bool() -> None:
    registry = ToolRegistry()
    registry.register("to_bool", _to_bool)

    assert registry.execute(_call("to_bool", {"x": True})).is_error is False
    assert registry.execute(_call("to_bool", {"x": False})).is_error is False

    with pytest.raises(ToolParameterError, match="expected bool, got int"):
        registry.execute(_call("to_bool", {"x": 1}))


def test_execute_result_reuses_the_call_id() -> None:
    registry = ToolRegistry()
    registry.register("add", _add)

    result = registry.execute(_call("add", {"a": 1}, call_id="call-9"))

    assert result.tool_call_id == "call-9"


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({}, "'a'.*required"),
        ({"a": "not-an-int"}, "expected int, got str"),
        ({"a": True}, "expected int, got bool"),
        ({"a": None}, "expected int, got NoneType"),
        ({"a": 1, "extra": 2}, "unexpected"),
    ],
)
def test_execute_rejects_invalid_arguments(arguments: dict[str, object], match: str) -> None:
    registry = ToolRegistry()
    registry.register("add", _add)

    with pytest.raises(ToolParameterError, match=match):
        registry.execute(_call("add", arguments))


def test_parameter_error_carries_field_and_reason() -> None:
    registry = ToolRegistry()
    registry.register("add", _add)

    with pytest.raises(ToolParameterError) as caught:
        registry.execute(_call("add", {"a": "not-an-int"}))

    assert caught.value.parameter == "a"
    assert "expected int, got str" in caught.value.reason


def test_execute_exception_becomes_error_result_without_propagating() -> None:
    registry = ToolRegistry()
    registry.register("explodes", _explodes)

    result = registry.execute(_call("explodes", {}))

    assert result.is_error is True
    assert "ValueError" in result.content
    assert "boom" in result.content


def test_execute_non_str_return_becomes_error_result() -> None:
    registry = ToolRegistry()
    registry.register("returns_non_str", _returns_non_str)

    result = registry.execute(_call("returns_non_str", {}))

    assert result.is_error is True
    assert "non-str" in result.content
    assert "int" in result.content
