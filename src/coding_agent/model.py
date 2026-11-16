"""Minimal model boundary and deterministic scripted implementation."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Protocol, TypeAlias, runtime_checkable

from coding_agent.protocol import Message, ModelResponse


ScriptStep: TypeAlias = ModelResponse | Exception


class ScriptedModelError(RuntimeError):
    """Base error for invalid ScriptedModel consumption."""


class ScriptExhaustedError(ScriptedModelError):
    """Raised when a model call has no configured script step remaining."""

    def __init__(self, *, call_index: int) -> None:
        super().__init__(f"script exhausted before model call {call_index}")
        self.call_index = call_index


class UnconsumedScriptError(ScriptedModelError):
    """Raised when a test ends before every configured step is consumed."""

    def __init__(self, *, remaining_steps: int) -> None:
        super().__init__(f"script has {remaining_steps} unconsumed step(s)")
        self.remaining_steps = remaining_steps


@runtime_checkable
class Model(Protocol):
    """Provider-neutral synchronous model interface.

    Implementations receive protocol messages and return one validated response.
    History validation, tool execution, looping, and stopping remain caller
    responsibilities.
    """

    def respond(self, messages: Sequence[Message]) -> ModelResponse:
        """Return one model response for the supplied history."""

        ...


def _snapshot_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    snapshot = tuple(messages)
    for index, message in enumerate(snapshot):
        if not isinstance(message, Message):
            raise TypeError(f"messages[{index}] must be a Message")
    return deepcopy(snapshot)


def _snapshot_step(step: object, index: int) -> ScriptStep:
    if isinstance(step, ModelResponse):
        return deepcopy(step)
    if isinstance(step, Exception):
        return step
    raise TypeError(f"steps[{index}] must be a ModelResponse or Exception")


class ScriptedModel:
    """Consume a finite FIFO script while recording detached input snapshots."""

    def __init__(self, steps: Iterable[ScriptStep]) -> None:
        self._steps: deque[ScriptStep] = deque(
            _snapshot_step(step, index) for index, step in enumerate(steps)
        )
        self._calls: list[tuple[Message, ...]] = []

    @property
    def calls(self) -> tuple[tuple[Message, ...], ...]:
        """Return detached snapshots of all recorded calls, including over-calls."""

        return deepcopy(tuple(self._calls))

    @property
    def remaining_steps(self) -> int:
        return len(self._steps)

    def respond(self, messages: Sequence[Message]) -> ModelResponse:
        call = _snapshot_messages(messages)
        self._calls.append(call)
        call_index = len(self._calls) - 1

        if not self._steps:
            raise ScriptExhaustedError(call_index=call_index)

        step = self._steps.popleft()
        if isinstance(step, Exception):
            raise step
        return step

    def assert_complete(self) -> None:
        """Require every configured step to have been consumed."""

        if self._steps:
            raise UnconsumedScriptError(remaining_steps=len(self._steps))


__all__ = [
    "Model",
    "ScriptExhaustedError",
    "ScriptedModel",
    "ScriptedModelError",
    "UnconsumedScriptError",
]
