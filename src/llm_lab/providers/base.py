"""Provider port. Всё зависит только от этого интерфейса, не от SDK."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    # SYSTEM = операторский канал mid-conversation ({"role":"system"}), его
    # нельзя подделать из user/tool текста. Начальный system prompt идёт
    # отдельным top-level параметром, см. ClaudeProvider.
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_RESULT = "tool_result"


@dataclass
class Message:
    role: Role
    content: str
    tool_use_id: str | None = None  # только для TOOL_RESULT


@dataclass
class ToolSpec:
    # dangerous => гейт human-in-the-loop перед вызовом
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]
    dangerous: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    # stop_reason: end_turn / tool_use / refusal. refusal обработать до text.
    text: str
    stop_reason: str = "end_turn"
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        raise NotImplementedError
