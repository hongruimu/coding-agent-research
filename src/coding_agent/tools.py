"""Tool registry and signature-derived input schemas.

Tools are registered as a unique name to a callable; the parameter schema is
derived once from the callable signature at registration time. Execution
validates arguments against that schema and captures failures as ToolResult
data; it never accesses the network or the filesystem.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import NoneType
from typing import get_args, get_type_hints

from coding_agent.protocol import ProtocolValidationError, ToolCall, ToolResult

_SCALAR_TYPES = (str, int, float, bool)


class ToolRegistrationError(ProtocolValidationError):
    """Raised when a tool cannot be registered (invalid name or signature)."""


class ToolNotFoundError(ProtocolValidationError):
    """Raised when a tool call names a tool that is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"tool {name!r} is not registered")
        self.name = name


class ToolParameterError(ProtocolValidationError):
    """Raised when tool-call arguments fail schema validation."""

    def __init__(self, parameter: str, reason: str) -> None:
        super().__init__(f"parameter {parameter!r}: {reason}")
        self.parameter = parameter
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """One parameter in a tool's input schema, derived from its signature."""

    name: str
    type: type
    required: bool = True
    nullable: bool = False


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """The ordered, signature-derived parameter contract of a tool."""

    parameters: tuple[ToolParameter, ...] = ()


@dataclass(frozen=True, slots=True)
class Tool:
    """A registered tool: its unique name, callable, and cached schema."""

    name: str
    function: Callable[..., str]
    schema: ToolSchema


def _resolve_scalar(name: str, resolved: object) -> tuple[type, bool]:
    """Return ``(inner_type, nullable)`` for one resolved annotation, or raise.

    ``resolved`` comes from ``typing.get_type_hints`` with ``Annotated`` already
    unwrapped. Supported shapes are a bare scalar and ``Optional[T]`` / ``T | None``.
    """

    args = get_args(resolved)
    if not args:
        if resolved in _SCALAR_TYPES:
            return resolved, False
        raise ToolRegistrationError(f"parameter {name!r}: unsupported annotation {resolved!r}")

    if NoneType not in args:
        raise ToolRegistrationError(f"parameter {name!r}: unsupported annotation {resolved!r}")

    non_none = [argument for argument in args if argument is not NoneType]
    if len(non_none) == 1 and non_none[0] in _SCALAR_TYPES:
        return non_none[0], True
    raise ToolRegistrationError(f"parameter {name!r}: unsupported annotation {resolved!r}")


def schema_from_function(function: Callable[..., str]) -> ToolSchema:
    """Derive a frozen input schema from a callable's signature.

    This is the only place that touches ``inspect`` / ``get_type_hints``. Every
    parameter must be annotated with a supported scalar; anything that cannot be
    validated reliably fails at registration instead of being skipped.
    """

    try:
        signature = inspect.signature(function)
        hints = get_type_hints(function)
    except (NameError, TypeError, ValueError) as exc:
        raise ToolRegistrationError(f"cannot inspect tool signature: {exc}") from exc

    parameters: list[ToolParameter] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ToolRegistrationError(f"parameter {name!r}: *args/**kwargs are not supported")
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ToolRegistrationError(
                f"parameter {name!r}: positional-only parameters are not supported"
            )
        if parameter.annotation is inspect.Parameter.empty:
            raise ToolRegistrationError(f"parameter {name!r}: missing type annotation")

        resolved = hints.get(name)
        if resolved is None:
            raise ToolRegistrationError(f"parameter {name!r}: annotation could not be resolved")
        inner_type, nullable = _resolve_scalar(name, resolved)
        parameters.append(
            ToolParameter(
                name=name,
                type=inner_type,
                required=parameter.default is inspect.Parameter.empty,
                nullable=nullable,
            )
        )

    return ToolSchema(parameters=tuple(parameters))


def _describe_type(parameter: ToolParameter) -> str:
    base = parameter.type.__name__
    return f"{base} or None" if parameter.nullable else base


def _matches(parameter: ToolParameter, value: object) -> bool:
    if value is None:
        return parameter.nullable
    if parameter.type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter.type is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, parameter.type)


def _validate_arguments(
    schema: ToolSchema,
    arguments: Mapping[str, object],
) -> dict[str, object]:
    provided = dict(arguments)
    by_name = {parameter.name: parameter for parameter in schema.parameters}

    for parameter in schema.parameters:
        if parameter.required and parameter.name not in provided:
            raise ToolParameterError(parameter.name, "required argument missing")

    for name in provided:
        if name not in by_name:
            raise ToolParameterError(name, "unexpected argument")

    for name, value in provided.items():
        parameter = by_name[name]
        if not _matches(parameter, value):
            raise ToolParameterError(
                name,
                f"expected {_describe_type(parameter)}, got {type(value).__name__}",
            )

    return provided


class ToolRegistry:
    """A unique-name registry that caches each tool's derived schema."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, function: Callable[..., str]) -> Tool:
        if not isinstance(name, str) or not name.strip():
            raise ToolRegistrationError("tool name must be a non-blank string")
        if name in self._tools:
            raise ToolRegistrationError(f"tool {name!r} is already registered")

        schema = schema_from_function(function)
        tool = Tool(name=name, function=function, schema=schema)
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def execute(self, call: ToolCall) -> ToolResult:
        """Validate a tool call, run the tool, and wrap its outcome as a result.

        A ``ProtocolValidationError`` raised by the tool function propagates to
        the caller (it is a request-invalid protocol error, not a result); any
        other exception the function raises is captured as an ``is_error=True``
        result.
        """

        if call.name not in self:
            raise ToolNotFoundError(call.name)

        tool = self._tools[call.name]
        arguments = _validate_arguments(tool.schema, call.arguments)

        try:
            content = tool.function(**arguments)
        except ProtocolValidationError:
            raise
        except Exception as error:
            return ToolResult(
                tool_call_id=call.id,
                content=f"{type(error).__name__}: {error}",
                is_error=True,
            )

        if not isinstance(content, str):
            return ToolResult(
                tool_call_id=call.id,
                content=f"tool {call.name!r} returned non-str {type(content).__name__}",
                is_error=True,
            )

        return ToolResult(tool_call_id=call.id, content=content, is_error=False)


__all__ = [
    "Tool",
    "ToolNotFoundError",
    "ToolParameter",
    "ToolParameterError",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolSchema",
    "schema_from_function",
]
