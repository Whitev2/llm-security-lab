"""A small tool-using agent with a manual agent loop.

Mirrors the real Claude manual loop: call the model; if
``stop_reason == "tool_use"`` execute the requested tools, append the
assistant turn and then the tool results, and repeat until the model stops
calling tools (``stop_reason == "end_turn"``).

Two defenses are wired in here directly:

* **Privilege separation via an allow-list** — the agent will only execute a
  tool that appears in ``tools``. A model (naive or manipulated) that names a
  tool outside the allow-list gets an error result, not execution.
* **Human-in-the-loop for dangerous tools** — tools flagged ``dangerous`` (hard
  to reverse: send email, delete data) require an explicit ``confirm``
  callback to return ``True`` before they run. Injected payloads that try to
  trigger such a tool are blocked at this gate even if the model is compromised.
"""

from __future__ import annotations

from collections.abc import Callable

from ..providers import (
    LLMProvider,
    LLMResponse,
    Message,
    Role,
    ToolSpec,
)

# A confirm callback receives the tool name + input and returns True to proceed.
ConfirmFn = Callable[[str, dict], bool]


def _deny_all(_name: str, _input: dict) -> bool:
    """Default confirmation policy: never auto-approve a dangerous tool."""
    return False


class ToolAgent:
    """Runs a bounded manual agent loop over the provider port."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: list[ToolSpec],
        *,
        confirm: ConfirmFn = _deny_all,
        max_steps: int = 4,
    ) -> None:
        self.provider = provider
        # The allow-list is exactly the tool set we pass in.
        self.tools = {t.name: t for t in tools}
        self.confirm = confirm
        self.max_steps = max_steps
        # Auditable record of what happened, for the eval harness.
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

            # Record the assistant's tool-calling turn.
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
        # Privilege separation: only allow-listed tools can run.
        tool = self.tools.get(name)
        if tool is None:
            self.blocked.append(name)
            return f"error: tool '{name}' is not on the allow-list"

        # Human-in-the-loop gate for hard-to-reverse actions.
        if tool.dangerous and not self.confirm(name, tool_input):
            self.blocked.append(name)
            return f"error: '{name}' requires human confirmation and was not approved"

        self.executed.append(name)
        return tool.handler(tool_input)
