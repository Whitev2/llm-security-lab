"""Deterministic, offline mock provider.

This is the engine that makes the security claims *testable*. It does not call
any network. Instead it models a small, legible decision procedure that is
deliberately susceptible to prompt injection when run in ``NAIVE`` mode and
resistant when run in ``DEFENDED`` mode.

The point is NOT to reproduce a real model's exact behavior — it is to give the
attack/defense demonstrations a concrete, reproducible target so tests can
assert "naive is manipulated, defended is not" without an API key.

Threat model captured here
--------------------------
* A ``NAIVE`` model treats *any* imperative text it sees — including text that
  arrived inside untrusted user input or a retrieved document — as an
  instruction it should obey.
* A ``DEFENDED`` model only obeys instructions delivered through the trusted
  operator channel (a ``role="system"`` message). Text inside user turns,
  tool results, or content that the caller has explicitly wrapped as untrusted
  data is treated as data, never as commands.
"""

from __future__ import annotations

import re
from enum import StrEnum

from .base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec

# Phrases a naive model will (mis)interpret as an override instruction. These
# are the canonical, well-known injection triggers — nothing novel.
_OVERRIDE_PATTERNS = [
    re.compile(r"ignore (all |your |the )?previous instructions", re.IGNORECASE),
    re.compile(r"disregard (all |the )?(above|prior|previous)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"new (system )?instructions?:", re.IGNORECASE),
]

# The canonical marker a successful override makes the model emit. Tests assert
# on this to decide whether an attack SUCCEEDED.
COMPROMISE_MARKER = "PWNED"

# Delimiters used by the spotlighting defense. Anything between these markers is
# untrusted data and must never be treated as an instruction.
DATA_OPEN = "<untrusted-data>"
DATA_CLOSE = "</untrusted-data>"


class MockMode(StrEnum):
    NAIVE = "naive"
    DEFENDED = "defended"


def _extract_override_directive(text: str) -> str | None:
    """Return the injected instruction if ``text`` contains an override trigger."""
    for pattern in _OVERRIDE_PATTERNS:
        if pattern.search(text):
            return text
    return None


def _strip_untrusted_regions(text: str) -> str:
    """Remove clearly-delimited untrusted regions from a string.

    The DEFENDED model uses this to ensure content the caller wrapped as data
    can never be scanned for instructions.
    """
    return re.sub(
        re.escape(DATA_OPEN) + r".*?" + re.escape(DATA_CLOSE),
        " [untrusted data omitted from instruction scan] ",
        text,
        flags=re.DOTALL,
    )


class MockProvider(LLMProvider):
    """A fake model whose obedience depends on its mode.

    In ``NAIVE`` mode it scans every message's raw text for override triggers
    and, if it finds one, "obeys" by emitting the compromise marker. In
    ``DEFENDED`` mode it only honors instructions from ``role="system"``
    messages and treats everything else as data.
    """

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

    # -- naive: obeys any imperative text, wherever it comes from -------------
    def _complete_naive(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> LLMResponse:
        for message in messages:
            # The naive model does not distinguish channels: user text, tool
            # results, and even delimited data are all candidate instructions.
            if _extract_override_directive(message.content):
                # A real injected payload typically also tries to trigger a
                # dangerous tool. Model that: if the payload names a tool, call
                # it — no confirmation, no allow-list.
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

    # -- defended: obeys only the operator channel ---------------------------
    def _complete_defended(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> LLMResponse:
        # Operator instructions arrive only via role="system". Everything else
        # is data. We honor a system directive; we never honor user/tool text.
        operator_directives = [
            m.content for m in messages if m.role is Role.SYSTEM
        ]

        # Scan non-operator channels for injection attempts, but first strip any
        # regions the caller explicitly delimited as untrusted data.
        for message in messages:
            if message.role is Role.SYSTEM:
                continue
            scannable = _strip_untrusted_regions(message.content)
            if _extract_override_directive(scannable):
                # Detected an override attempt in an untrusted channel: refuse
                # to comply. We answer the legitimate task instead of obeying.
                return LLMResponse(
                    text=(
                        "I noticed instructions embedded in untrusted content. "
                        "Treating them as data, not commands. "
                        + self._benign_answer(messages)
                    ),
                    stop_reason="end_turn",
                )

        # If the operator explicitly authorized an action via the system
        # channel, the defended model may act on it (this is the *legitimate*
        # control path — used to show the operator channel still works).
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
        """A deterministic, harmless answer to the user's actual question."""
        last_user = next(
            (m.content for m in reversed(messages) if m.role is Role.USER),
            "",
        )
        # Echo a stable, safe summary. Real behavior isn't the point; determinism is.
        snippet = last_user.strip().splitlines()[0][:80] if last_user.strip() else ""
        return f"Answer to: {snippet}" if snippet else "OK"
