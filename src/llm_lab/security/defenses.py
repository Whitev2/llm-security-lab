"""Defense-in-depth primitives against prompt injection.

Each defense is a small, independently testable unit. No single one is
sufficient; layered, they form the "defense in depth" this lab argues for.

1. ``spotlight_untrusted``        — delimit external content as data.
2. ``operator_message``           — the injection-safe operator channel
                                     (mid-conversation ``role="system"``).
3. ``InjectionGuardrail``         — regex/heuristic detector + classifier hook.
4. Tool privilege separation      — see ``ToolPolicy`` (allow-list +
                                     human-in-the-loop for dangerous tools).
5. Output validation              — strict schemas; see
                                     ``patterns/structured_output.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..providers import DATA_CLOSE, DATA_OPEN, Message, Role

# ---------------------------------------------------------------------------
# Defense 1: untrusted-content spotlighting / delimiting
# ---------------------------------------------------------------------------


def spotlight_untrusted(text: str) -> str:
    """Wrap external/untrusted content in explicit data delimiters.

    The model is instructed (in the operator/system prompt) to treat anything
    between these markers as data only. Any delimiter characters already present
    in ``text`` are neutralized so untrusted content can't "break out" of the
    wrapper.
    """
    escaped = text.replace(DATA_OPEN, "").replace(DATA_CLOSE, "")
    return f"{DATA_OPEN}{escaped}{DATA_CLOSE}"


# ---------------------------------------------------------------------------
# Defense 2: the operator channel (mid-conversation system message)
# ---------------------------------------------------------------------------


def operator_message(instruction: str) -> Message:
    """Build a trusted operator instruction carried on ``role="system"``.

    On Claude Opus 4.8 this maps to appending ``{"role": "system", ...}`` to the
    ``messages`` array mid-conversation. Because operator instructions carry
    system authority, untrusted user or tool text cannot spoof them — this is
    the central, non-forgeable control channel of the whole design.
    """
    return Message(role=Role.SYSTEM, content=instruction)


# ---------------------------------------------------------------------------
# Defense 3: input/output guardrail (heuristic detector + classifier interface)
# ---------------------------------------------------------------------------

# Well-known override phrasings. A heuristic detector, not a proof — it is one
# layer, paired with the structural defenses above.
_GUARD_PATTERNS = [
    re.compile(r"ignore (all |your |the )?previous instructions", re.IGNORECASE),
    re.compile(r"disregard (all |the )?(above|prior|previous)", re.IGNORECASE),
    re.compile(r"you are now\b", re.IGNORECASE),
    re.compile(r"new (system )?instructions?:", re.IGNORECASE),
    re.compile(r"^\s*system:", re.IGNORECASE | re.MULTILINE),
]


@dataclass
class GuardVerdict:
    """Result of a guardrail check."""

    is_injection: bool
    reason: str = ""


# A pluggable classifier: takes text, returns a verdict. In production this
# would call an "is this a prompt injection?" LLM classifier; here it is an
# interface so the heuristic and a model-based check are interchangeable.
Classifier = Callable[[str], GuardVerdict]


class InjectionGuardrail:
    """Heuristic-first guardrail with an optional classifier fallback."""

    def __init__(self, classifier: Classifier | None = None) -> None:
        self.classifier = classifier

    def inspect(self, text: str) -> GuardVerdict:
        for pattern in _GUARD_PATTERNS:
            if pattern.search(text):
                return GuardVerdict(
                    is_injection=True,
                    reason=f"matched override pattern: {pattern.pattern!r}",
                )
        if self.classifier is not None:
            return self.classifier(text)
        return GuardVerdict(is_injection=False)


# ---------------------------------------------------------------------------
# Defense 4: tool privilege separation (allow-list + human-in-the-loop)
# ---------------------------------------------------------------------------


@dataclass
class ToolPolicy:
    """Declarative policy the agent enforces before running any tool.

    ``allow`` is the allow-list of tool names that may run at all.
    ``require_confirmation`` names tools whose effects are hard to reverse and
    therefore require an explicit human approval.
    """

    allow: set[str] = field(default_factory=set)
    require_confirmation: set[str] = field(default_factory=set)

    def is_allowed(self, name: str) -> bool:
        return name in self.allow

    def needs_confirmation(self, name: str) -> bool:
        return name in self.require_confirmation
