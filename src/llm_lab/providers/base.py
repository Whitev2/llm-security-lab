"""Provider port: the abstract seam that lets us swap LLM backends.

The rest of the codebase depends only on this interface, never on a concrete
SDK. That makes the security demonstrations testable offline against a
deterministic ``MockProvider`` while remaining runnable against real Claude via
``ClaudeProvider``.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Message roles.

    ``SYSTEM`` here models the **mid-conversation** operator channel: on
    Claude Opus 4.8 an operator instruction can be appended to the ``messages``
    array as ``{"role": "system", ...}`` rather than smuggled into user text.
    Because it carries operator authority that untrusted user/tool text cannot
    forge, it is the injection-safe control channel this lab builds defenses
    around. (For the *initial* prompt, real Claude uses the top-level
    ``system`` parameter; see ``ClaudeProvider``.)
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_RESULT = "tool_result"


@dataclass
class Message:
    """A single conversation turn in provider-neutral form."""

    role: Role
    content: str
    # For TOOL_RESULT messages, the id of the originating tool_use block.
    tool_use_id: str | None = None


@dataclass
class ToolSpec:
    """A tool the model may call.

    ``handler`` runs the tool. ``dangerous`` marks tools whose effects are hard
    to reverse (send email, delete data); the tool-use pattern gates these
    behind human-in-the-loop confirmation. Mirrors the strict tool schema shape
    used by the real Claude API (``strict: True`` + ``additionalProperties:
    False``).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]
    dangerous: bool = False


@dataclass
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Provider-neutral response.

    ``stop_reason`` follows the Claude convention: ``end_turn``, ``tool_use``,
    or ``refusal``. ``refusal`` must be handled before reading ``text``.
    """

    text: str
    stop_reason: str = "end_turn"
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(abc.ABC):
    """The port. Adapters (Claude, Mock) implement this."""

    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Run one turn of the conversation and return the model's response."""
        raise NotImplementedError
