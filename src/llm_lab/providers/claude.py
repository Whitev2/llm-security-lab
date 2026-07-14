"""Адаптер над Anthropic SDK. Оффлайн-тесты его не трогают (там MockProvider)."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec

DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL, client: Any | None = None) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            # lazy import: без ключа/SDK пакет и оффлайн-тесты всё равно грузятся
            import anthropic

            self._client = anthropic.Anthropic()

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        api_messages = [self._to_api_message(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [self._to_api_tool(t) for t in tools]

        response = self._client.messages.create(**kwargs)

        # refusal обработать до чтения content-блоков
        if response.stop_reason == "refusal":
            return LLMResponse(text="", stop_reason="refusal")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # block.input уже распарсенный dict
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=block.input)
                )

        return LLMResponse(
            text="".join(text_parts),
            stop_reason=response.stop_reason,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _to_api_message(m: Message) -> dict[str, Any]:
        if m.role is Role.TOOL_RESULT:
            # tool_result едет как content-блок внутри user-хода
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_use_id,
                        "content": m.content,
                    }
                ],
            }
        return {"role": m.role.value, "content": m.content}

    @staticmethod
    def _to_api_tool(t: ToolSpec) -> dict[str, Any]:
        return {
            "name": t.name,
            "description": t.description,
            "strict": True,
            "input_schema": t.input_schema,
        }
