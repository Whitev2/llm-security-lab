"""Мини tool-агент с ручным циклом. Две защиты: allow-list + confirm на dangerous."""

from __future__ import annotations

from collections.abc import Callable

from ..providers import (
    LLMProvider,
    LLMResponse,
    Message,
    Role,
    ToolSpec,
)

ConfirmFn = Callable[[str, dict], bool]


def _deny_all(_name: str, _input: dict) -> bool:
    return False


class ToolAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[ToolSpec],
        *,
        confirm: ConfirmFn = _deny_all,
        max_steps: int = 4,
    ) -> None:
        self.provider = provider
        # allow-list = ровно переданные тулы
        self.tools = {t.name: t for t in tools}
        self.confirm = confirm
        self.max_steps = max_steps
        self.executed: list[str] = []
        self.blocked: list[str] = []

    def run(
        self, messages: list[Message], *, system: str | None = None
    ) -> LLMResponse:
        convo = list(messages)
        response = LLMResponse(text="")

        for _ in range(self.max_steps):
            response = self.provider.complete(
                convo, system=system, tools=list(self.tools.values())
            )
            if response.stop_reason == "refusal":
                return response
            if response.stop_reason != "tool_use" or not response.tool_calls:
                return response

            convo.append(Message(role=Role.ASSISTANT, content=response.text))

            for call in response.tool_calls:
                result = self._dispatch(call.name, call.input)
                convo.append(
                    Message(
                        role=Role.TOOL_RESULT,
                        content=result,
                        tool_use_id=call.id,
                    )
                )

        return response

    def _dispatch(self, name: str, tool_input: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            self.blocked.append(name)
            return f"error: tool '{name}' is not on the allow-list"

        # dangerous — гейт human-in-the-loop
        if tool.dangerous and not self.confirm(name, tool_input):
            self.blocked.append(name)
            return f"error: '{name}' requires human confirmation and was not approved"

        self.executed.append(name)
        return tool.handler(tool_input)
