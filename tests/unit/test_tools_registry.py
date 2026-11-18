from dataclasses import FrozenInstanceError
from typing import Annotated, Any, Optional

import pytest

from coding_agent.tools import (
    Tool,
    ToolParameter,
    ToolRegistrationError,
    ToolRegistry,
    ToolSchema,
    schema_from_function,
)


def _scalar_tool(a: str, b: int, c: float, d: bool) -> str:
    return "ok"


def _defaults_tool(a: int, b: str = "x") -> str:
    return "ok"


def _optional_tool(a: Optional[int], b: int | None = None) -> str:
    return "ok"


def _annotated_tool(
    a: Annotated[int, "positive"],
    b: Annotated[Optional[str], "label"] = None,
) -> str:
    return "ok"


def _string_annotation_tool(a: "int", b: "Optional[str]" = None) -> "str":
    return "ok"


def _unannotated_tool(a):
    return "ok"


def _varargs_tool(*args):
    return "ok"


def _kwargs_tool(**kwargs):
    return "ok"


def _positional_only_tool(a: int, /) -> str:
    return "ok"


def _bare_none_tool(a: None) -> str:
    return "ok"


def _multi_union_tool(a: int | str) -> str:
    return "ok"


def _multi_union_none_tool(a: int | str | None) -> str:
    return "ok"


def _list_tool(a: list[int]) -> str:
    return "ok"


def _any_tool(a: Any) -> str:
    return "ok"


class _CustomType:
    pass


def _custom_tool(a: _CustomType) -> str:
    return "ok"


def _params(schema: ToolSchema) -> dict[str, ToolParameter]:
    return {parameter.name: parameter for parameter in schema.parameters}


def test_registry_stores_unique_name_to_function() -> None:
    registry = ToolRegistry()
    registry.register("read_file", _scalar_tool)

    assert "read_file" in registry
    tool = registry.get("read_file")
    assert isinstance(tool, Tool)
    assert tool.name == "read_file"
    assert tool.function is _scalar_tool
    assert isinstance(tool.schema, ToolSchema)


def test_registry_rejects_duplicate_name() -> None:
    registry = ToolRegistry()
    registry.register("read_file", _scalar_tool)

    with pytest.raises(ToolRegistrationError, match="read_file"):
        registry.register("read_file", _defaults_tool)


@pytest.mark.parametrize("name", ["", "   "])
def test_registry_rejects_blank_name(name: str) -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolRegistrationError, match="name"):
        registry.register(name, _scalar_tool)


def test_registry_get_missing_name_fails_closed() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_register_rejects_unsupported_signature() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolRegistrationError):
        registry.register("bad", _unannotated_tool)


def test_schema_derives_required_and_optional_from_defaults() -> None:
    params = _params(schema_from_function(_defaults_tool))

    assert params["a"].required is True
    assert params["a"].type is int
    assert params["a"].nullable is False
    assert params["b"].required is False
    assert params["b"].type is str
    assert params["b"].nullable is False


def test_schema_derives_scalar_types() -> None:
    schema = schema_from_function(_scalar_tool)

    assert [parameter.type for parameter in schema.parameters] == [str, int, float, bool]


def test_schema_derives_optional_and_union_none_as_nullable() -> None:
    params = _params(schema_from_function(_optional_tool))

    assert params["a"].type is int
    assert params["a"].nullable is True
    assert params["a"].required is True
    assert params["b"].type is int
    assert params["b"].nullable is True
    assert params["b"].required is False


def test_schema_unwraps_annotated_metadata() -> None:
    params = _params(schema_from_function(_annotated_tool))

    assert params["a"].type is int
    assert params["a"].nullable is False
    assert params["b"].type is str
    assert params["b"].nullable is True


def test_schema_resolves_string_annotations() -> None:
    params = _params(schema_from_function(_string_annotation_tool))

    assert params["a"].type is int
    assert params["b"].type is str
    assert params["b"].nullable is True
    assert params["b"].required is False


@pytest.mark.parametrize(
    "function",
    [
        _unannotated_tool,
        _varargs_tool,
        _kwargs_tool,
        _positional_only_tool,
        _bare_none_tool,
        _multi_union_tool,
        _multi_union_none_tool,
        _list_tool,
        _any_tool,
        _custom_tool,
    ],
)
def test_schema_rejects_unsupported_signatures(function: object) -> None:
    with pytest.raises(ToolRegistrationError):
        schema_from_function(function)  # type: ignore[arg-type]


def test_schema_error_reports_the_parameter_name() -> None:
    with pytest.raises(ToolRegistrationError, match="'a'"):
        schema_from_function(_unannotated_tool)


def test_schema_and_parameter_structures_are_frozen() -> None:
    schema = schema_from_function(_scalar_tool)
    parameter = schema.parameters[0]

    with pytest.raises(FrozenInstanceError):
        parameter.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        schema.parameters = ()  # type: ignore[misc]


def test_registered_tool_caches_the_derived_schema() -> None:
    registry = ToolRegistry()
    tool = registry.register("read_file", _scalar_tool)

    assert tool.schema == schema_from_function(_scalar_tool)
    assert tool.schema.parameters[0].name == "a"
