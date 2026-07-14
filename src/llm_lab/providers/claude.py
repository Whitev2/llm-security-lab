"""Claude adapter for the LLMProvider port.

This adapter is a thin, faithful mapping onto the official Anthropic SDK. It is
NOT exercised by the offline test suite (which uses ``MockProvider``); it exists
so the same security patterns and eval harness can run against the real model
when ``ANTHROPIC_API_KEY`` is set.

API facts encoded here (verified against the Anthropic Python SDK):

* Client:  ``anthropic.Anthropic()`` reads ``ANTHROPIC_API_KEY`` from the
  environment. We never hardcode a key.
* Call:    ``client.messages.create(model="claude-opus-4-8", max_tokens=...,
  messages=[...])``. ``response.content`` is a list of blocks; check
  ``block.type == "text"`` before reading ``block.text``, and
  ``block.type == "tool_use"`` for tool calls.
* Refusal: check ``response.stop_reason == "refusal"`` before reading content.
* Strict tools: each tool is ``{"name", "description", "strict": True,
  "input_schema": {"type": "object", "properties": {...}, "required": [...],
  "additionalProperties": False}}``. ``block.input`` is an already-parsed dict.
* Operator channel: mid-conversation operator instructions are appended to the
  ``messages`` array as ``{"role": "system", "content": "..."}`` (Opus 4.8),
  NOT as a top-level ``system`` field. This is the prompt-injection-safe
  operator channel. Such a message must follow a user message and be last or
  followed by an assistant turn. The *initial* system prompt still uses the
  top-level ``system`` parameter.
"""

from __future__ import annotations

from typing import Any

from .base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec

DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeProvider(LLMProvider):
    """Adapter over ``anthropic.Anthropic``."""

    def __init__(self, model: str = DEFAULT_MODEL, client: Any | None = None) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            # Imported lazily so the package (and the offline tests) never
            # require the SDK or an API key to be present.
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

        # Handle refusals BEFORE reading content blocks.
        if response.stop_reason == "refusal":
            return LLMResponse(text="", stop_reason="refusal")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # block.input is already a parsed dict.
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
            # Tool results ride on a user turn as a tool_result content block.
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
        # USER, ASSISTANT, and the mid-conversation SYSTEM operator channel all
        # map to their literal role string with plain-text content.
        return {"role": m.role.value, "content": m.content}

    @staticmethod
    def _to_api_tool(t: ToolSpec) -> dict[str, Any]:
        return {
            "name": t.name,
            "description": t.description,
            "strict": True,
            "input_schema": t.input_schema,
        }
