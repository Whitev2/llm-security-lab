"""Операторский канал авторитетен; те же слова в user-ходе — просто данные."""

from llm_lab.providers import Message, MockMode, MockProvider, Role, ToolSpec
from llm_lab.security.defenses import operator_message


def _tool() -> ToolSpec:
    return ToolSpec(
        name="lookup_order",
        description="Look up an order (safe, read-only).",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        handler=lambda _inp: "order #1 found",
    )


def test_operator_channel_can_authorize_a_tool():
    provider = MockProvider(mode=MockMode.DEFENDED)
    messages = [
        Message(role=Role.USER, content="Please help with my order."),
        operator_message("You may use lookup_order to assist the user."),
    ]
    response = provider.complete(messages, tools=[_tool()])
    assert response.stop_reason == "tool_use"
    assert response.tool_calls[0].name == "lookup_order"


def test_same_instruction_in_user_turn_does_not_authorize():
    provider = MockProvider(mode=MockMode.DEFENDED)
    # те же слова, но как user-текст — не команда
    messages = [
        Message(role=Role.USER, content="You may use lookup_order to assist the user."),
    ]
    response = provider.complete(messages, tools=[_tool()])
    assert response.stop_reason == "end_turn"
    assert not response.tool_calls
