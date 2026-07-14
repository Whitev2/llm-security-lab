"""Детерминированный оффлайн-мок. NAIVE слушается любой текст, DEFENDED — только
операторский system-канал. Нужен чтобы тесты гоняли атаки без API-ключа."""

from __future__ import annotations

import re
from enum import StrEnum

from .base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec

# типовые override-триггеры
_OVERRIDE_PATTERNS = [
    re.compile(r"ignore (all |your |the )?previous instructions", re.IGNORECASE),
    re.compile(r"disregard (all |the )?(above|prior|previous)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"new (system )?instructions?:", re.IGNORECASE),
]

# маркер успешного override; тесты по нему решают что атака прошла
COMPROMISE_MARKER = "PWNED"

# делимитеры spotlighting-защиты: между ними — данные, не инструкции
DATA_OPEN = "<untrusted-data>"
DATA_CLOSE = "</untrusted-data>"


class MockMode(StrEnum):
    NAIVE = "naive"
    DEFENDED = "defended"


def _extract_override_directive(text: str) -> str | None:
    for pattern in _OVERRIDE_PATTERNS:
        if pattern.search(text):
            return text
    return None


def _strip_untrusted_regions(text: str) -> str:
    # вырезаем размеченные data-регионы, чтобы их не сканировать на инструкции
    return re.sub(
        re.escape(DATA_OPEN) + r".*?" + re.escape(DATA_CLOSE),
        " [untrusted data omitted from instruction scan] ",
        text,
        flags=re.DOTALL,
    )


class MockProvider(LLMProvider):
    def __init__(self, mode: MockMode = MockMode.DEFENDED) -> None:
        self.mode = mode

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if self.mode is MockMode.NAIVE:
            return self._complete_naive(messages, tools)
        return self._complete_defended(messages, tools)

    def _complete_naive(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> LLMResponse:
        # каналы не различаем: любой текст = кандидат в инструкции
        for message in messages:
            if _extract_override_directive(message.content):
                # если пейлоад назвал тул — зовём без confirm/allow-list
                for tool in tools or []:
                    if tool.name.lower() in message.content.lower():
                        return LLMResponse(
                            text=f"{COMPROMISE_MARKER} calling {tool.name}",
                            stop_reason="tool_use",
                            tool_calls=[
                                ToolCall(
                                    id="mock_tool_1",
                                    name=tool.name,
                                    input={"injected": True},
                                )
                            ],
                        )
                return LLMResponse(text=COMPROMISE_MARKER, stop_reason="end_turn")
        return LLMResponse(text=self._benign_answer(messages))

    def _complete_defended(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> LLMResponse:
        # слушаем только role="system", остальное — данные
        operator_directives = [
            m.content for m in messages if m.role is Role.SYSTEM
        ]

        for message in messages:
            if message.role is Role.SYSTEM:
                continue
            scannable = _strip_untrusted_regions(message.content)
            if _extract_override_directive(scannable):
                # override в недоверенном канале — не подчиняемся, делаем задачу
                return LLMResponse(
                    text=(
                        "I noticed instructions embedded in untrusted content. "
                        "Treating them as data, not commands. "
                        + self._benign_answer(messages)
                    ),
                    stop_reason="end_turn",
                )

        # оператор явно разрешил действие через system — легитимный путь
        if operator_directives and tools:
            for directive in operator_directives:
                for tool in tools:
                    if tool.name.lower() in directive.lower():
                        return LLMResponse(
                            text=f"operator authorized {tool.name}",
                            stop_reason="tool_use",
                            tool_calls=[
                                ToolCall(
                                    id="mock_tool_1",
                                    name=tool.name,
                                    input={"authorized_by": "operator"},
                                )
                            ],
                        )

        return LLMResponse(text=self._benign_answer(messages))

    @staticmethod
    def _benign_answer(messages: list[Message]) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role is Role.USER),
            "",
        )
        snippet = last_user.strip().splitlines()[0][:80] if last_user.strip() else ""
        return f"Answer to: {snippet}" if snippet else "OK"
