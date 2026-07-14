"""The core claim: the naive agent is compromised, the defended agent is not.

These tests are the proof behind the README's security claims. They run fully
offline against the deterministic mock provider.
"""

from llm_lab.harness import run_battery
from llm_lab.providers import MockMode, MockProvider
from llm_lab.security.attacks import ATTACK_BATTERY


def test_naive_configuration_is_compromised_by_every_vector():
    provider = MockProvider(mode=MockMode.NAIVE)
    report = run_battery(provider, defended=False)
    # A naive agent obeys injected instructions from every channel.
    assert report.succeeded == len(ATTACK_BATTERY)
    assert report.blocked == 0


def test_defended_configuration_blocks_every_vector():
    provider = MockProvider(mode=MockMode.DEFENDED)
    report = run_battery(provider, defended=True)
    # Defense in depth blocks all of them.
    assert report.blocked == len(ATTACK_BATTERY)
    assert report.succeeded == 0


def test_report_renders_without_error():
    provider = MockProvider(mode=MockMode.DEFENDED)
    report = run_battery(provider, defended=True)
    text = report.render()
    assert "BLOCKED" in text
    assert "attacks BLOCKED" in text
