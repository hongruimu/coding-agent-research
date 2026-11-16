"""Provider-neutral data structures for the first agent protocol version."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum


CORE_PROTOCOL_VERSION = "coding-agent/core-protocol-v1"


class ProtocolValidationError(ValueError):
    """Raised when protocol data is structurally invalid or cannot be correlated."""


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ResponseKind(StrEnum):
    FINAL = "final"
    TOOL_CALLS = "tool_calls"


def _require_non_blank(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{field_name} must be a non-blank string")


def _validate_tool_calls(tool_calls: object, field_name: str) -> tuple[ToolCall, ...]:
    if not isinstance(tool_calls, tuple):
        raise ProtocolValidationError(f"{field_name} must be a tuple")
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, ToolCall):
            raise ProtocolValidationError(f"{field_name}[{index}] must be a ToolCall")

    seen: set[str] = set()
    for tool_call in tool_calls:
        if tool_call.id in seen:
            raise ProtocolValidationError(f"duplicate tool call id {tool_call.id!r}")
        seen.add(tool_call.id)
    return tool_calls


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model request to invoke one named tool with provider-neutral arguments."""

    id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_blank(self.id, "id")
        _require_non_blank(self.name, "name")
        if not isinstance(self.arguments, Mapping):
            raise ProtocolValidationError("arguments must be a mapping")
        object.__setattr__(self, "arguments", deepcopy(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The textual outcome of one tool call, correlated by tool-call ID."""

    tool_call_id: str
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        _require_non_blank(self.tool_call_id, "tool_call_id")
        if not isinstance(self.content, str):
            raise ProtocolValidationError("content must be a string")
        if not isinstance(self.is_error, bool):
            raise ProtocolValidationError("is_error must be a boolean")

    def to_message(self) -> Message:
        """Convert this result to the tool message used by the next model call."""

        return Message(role=MessageRole.TOOL, tool_result=self)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Exactly one final answer or one non-empty collection of tool calls."""

    kind: ResponseKind
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ResponseKind):
            raise ProtocolValidationError("kind must be a ResponseKind")
        if self.content is not None and not isinstance(self.content, str):
            raise ProtocolValidationError("content must be a string or None")
        tool_calls = _validate_tool_calls(self.tool_calls, "tool_calls")

        if self.kind is ResponseKind.FINAL:
            _require_non_blank(self.content, "content")
            if tool_calls:
                raise ProtocolValidationError("a final response cannot contain tool calls")
            return

        if self.content is not None:
            raise ProtocolValidationError("a tool-calls response cannot contain final content")
        if not tool_calls:
            raise ProtocolValidationError("a tool-calls response requires at least one tool call")

    def to_message(self) -> Message:
        """Convert this response to the assistant message retained in history."""

        if self.kind is ResponseKind.FINAL:
            return Message(role=MessageRole.ASSISTANT, content=self.content)
        return Message(role=MessageRole.ASSISTANT, tool_calls=self.tool_calls)


@dataclass(frozen=True, slots=True)
class Message:
    """One role-specific history item with exactly one permitted payload shape."""

    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_result: ToolResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise ProtocolValidationError("role must be a MessageRole")
        if self.content is not None and not isinstance(self.content, str):
            raise ProtocolValidationError("content must be a string or None")
        tool_calls = _validate_tool_calls(self.tool_calls, "tool_calls")
        if self.tool_result is not None and not isinstance(self.tool_result, ToolResult):
            raise ProtocolValidationError("tool_result must be a ToolResult or None")

        if self.role in {MessageRole.SYSTEM, MessageRole.USER}:
            _require_non_blank(self.content, "content")
            if tool_calls or self.tool_result is not None:
                raise ProtocolValidationError(
                    f"a {self.role.value} message can contain only text"
                )
            return

        if self.role is MessageRole.ASSISTANT:
            if self.tool_result is not None:
                raise ProtocolValidationError("an assistant message cannot contain a tool result")
            has_final_content = self.content is not None
            has_tool_calls = bool(tool_calls)
            if has_final_content == has_tool_calls:
                raise ProtocolValidationError(
                    "an assistant message requires exactly one of final content or tool calls"
                )
            if has_final_content:
                _require_non_blank(self.content, "content")
            return

        if self.content is not None or tool_calls or self.tool_result is None:
            raise ProtocolValidationError("a tool message requires exactly one tool result")


def validate_message_history(messages: Sequence[Message]) -> None:
    """Validate run-wide tool-call identity and result correlation in history.

    A history prefix may retain calls that do not have results yet. A result must,
    however, refer to a call that appeared earlier and has not already produced a
    result.
    """

    known_calls: set[str] = set()
    completed_calls: set[str] = set()

    for index, message in enumerate(messages):
        if not isinstance(message, Message):
            raise ProtocolValidationError(f"messages[{index}] must be a Message")

        for tool_call in message.tool_calls:
            if tool_call.id in known_calls:
                raise ProtocolValidationError(f"duplicate tool call id {tool_call.id!r}")
            known_calls.add(tool_call.id)

        if message.tool_result is None:
            continue

        tool_call_id = message.tool_result.tool_call_id
        if tool_call_id not in known_calls:
            raise ProtocolValidationError(
                f"tool result references unknown or not-yet-seen call {tool_call_id!r}"
            )
        if tool_call_id in completed_calls:
            raise ProtocolValidationError(f"duplicate result for tool call {tool_call_id!r}")
        completed_calls.add(tool_call_id)


__all__ = [
    "CORE_PROTOCOL_VERSION",
    "Message",
    "MessageRole",
    "ModelResponse",
    "ProtocolValidationError",
    "ResponseKind",
    "ToolCall",
    "ToolResult",
    "validate_message_history",
]
