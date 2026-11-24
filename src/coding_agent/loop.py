"""Deterministic execution loop over model, tool, and result rounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from coding_agent.model import Model
from coding_agent.protocol import Message, ResponseKind, validate_message_history
from coding_agent.tools import ToolRegistry


class StopReason(StrEnum):
    """Why a loop run stopped."""

    FINAL = "final"


class StepLimitExceededError(RuntimeError):
    """Raised when the loop would make a model call beyond ``max_steps``."""

    def __init__(self, *, max_steps: int, steps: int) -> None:
        super().__init__(f"step limit exceeded after {steps} step(s) (max_steps={max_steps})")
        self.max_steps = max_steps
        self.steps = steps


@dataclass(frozen=True, slots=True)
class LoopResult:
    """The deterministic outcome of a completed loop run.

    ``messages`` is the full history including the caller-provided input, and
    doubles as the minimal trajectory: each round's model response becomes an
    assistant message and each tool result becomes a tool message, in order.
    ``final_text`` is always equal to ``messages[-1].content``.
    """

    messages: tuple[Message, ...]
    final_text: str
    stop_reason: StopReason


def run(
    model: Model,
    registry: ToolRegistry,
    messages: Sequence[Message],
    *,
    max_steps: int,
) -> LoopResult:
    """Run the model-tool loop until the model returns a final answer.

    Each round asks the model once and, when it requests tool calls, executes
    them one by one in order, feeding each result back before the next round.
    A step is one model call; ``max_steps`` bounds the number of steps. When the
    loop would make a call beyond the budget, it raises ``StepLimitExceededError``
    after the tools requested by the last allowed step have executed. Protocol
    errors, tool-not-found and parameter errors from the registry, and model
    exceptions propagate to the caller.
    """

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    history = list(messages)
    steps = 0
    while True:
        if steps >= max_steps:
            raise StepLimitExceededError(max_steps=max_steps, steps=steps)
        validate_message_history(history)
        response = model.respond(history)
        steps += 1
        history.append(response.to_message())

        if response.kind is ResponseKind.FINAL:
            return LoopResult(
                messages=tuple(history),
                final_text=response.content,
                stop_reason=StopReason.FINAL,
            )

        for tool_call in response.tool_calls:
            history.append(registry.execute(tool_call).to_message())


__all__ = ["LoopResult", "StepLimitExceededError", "StopReason", "run"]
