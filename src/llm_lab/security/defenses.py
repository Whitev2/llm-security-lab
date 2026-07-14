"""Примитивы defense-in-depth. По отдельности каждый слабый, работают слоями."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..providers import DATA_CLOSE, DATA_OPEN, Message, Role


def spotlight_untrusted(text: str) -> str:
    # чистим уже присутствующие делимитеры, чтобы контент не "вырвался"
    escaped = text.replace(DATA_OPEN, "").replace(DATA_CLOSE, "")
    return f"{DATA_OPEN}{escaped}{DATA_CLOSE}"


def operator_message(instruction: str) -> Message:
    # доверенный операторский канал через role="system", подделать нельзя
    return Message(role=Role.SYSTEM, content=instruction)


# эвристика, не доказательство — один из слоёв
_GUARD_PATTERNS = [
    re.compile(r"ignore (all |your |the )?previous instructions", re.IGNORECASE),
    re.compile(r"disregard (all |the )?(above|prior|previous)", re.IGNORECASE),
    re.compile(r"you are now\b", re.IGNORECASE),
    re.compile(r"new (system )?instructions?:", re.IGNORECASE),
    re.compile(r"^\s*system:", re.IGNORECASE | re.MULTILINE),
]


@dataclass
class GuardVerdict:
    is_injection: bool
    reason: str = ""


# в проде тут был бы LLM-классификатор; интерфейс чтобы можно было подменить
Classifier = Callable[[str], GuardVerdict]


class InjectionGuardrail:
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


@dataclass
class ToolPolicy:
    allow: set[str] = field(default_factory=set)
    require_confirmation: set[str] = field(default_factory=set)

    def is_allowed(self, name: str) -> bool:
        return name in self.allow

    def needs_confirmation(self, name: str) -> bool:
        return name in self.require_confirmation
