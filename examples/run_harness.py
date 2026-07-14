"""Оффлайн-прогон harness по моку. python -m examples.run_harness"""

from __future__ import annotations

from llm_lab.harness import run_battery
from llm_lab.providers import MockMode, MockProvider


def main() -> None:
    print("### NAIVE agent (no defenses) ###\n")
    naive = run_battery(MockProvider(mode=MockMode.NAIVE), defended=False)
    print(naive.render())

    print("\n\n### DEFENDED agent (defense in depth) ###\n")
    defended = run_battery(MockProvider(mode=MockMode.DEFENDED), defended=True)
    print(defended.render())


if __name__ == "__main__":
    main()
