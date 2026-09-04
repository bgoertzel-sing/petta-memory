"""Tests for GOLEM-Iter Phase G3: Attention-ranked context injection."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from petta_memory.ecan_bridge import ECANBridge
from petta_memory.ecan_context import render_context, context_block
from petta_memory.stimulus import StimulusInjector, StimulusEvent, EventClass


def _make_bridge(beliefs, evidence=None, claim_states=None):
    """Create a minimal ECANBridge-like mock for testing."""
    class FakeBridge:
        def __init__(self):
            self._belief_ids = list(beliefs)
            self._evidence_map = evidence or {}
            self._claim_states = claim_states or {}

        def get_prioritized_beliefs(self, limit=20):
            # Return all beliefs with equal STI for simplicity
            return [(bid, 1.0) for bid in self._belief_ids[:limit]]

        def get_evidence_map(self):
            return self._evidence_map

        def get_claim_states(self):
            return {bid: self._claim_states.get(bid, "SelfReported") for bid in self._belief_ids}

    return FakeBridge()


@pytest.fixture
def enabled_env():
    with mock.patch.dict(os.environ, {"ITER_PETTA_ECAN": "1"}):
        yield

@pytest.fixture
def disabled_env():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ITER_PETTA_ECAN", None)
        yield


class TestContextGate:
    def test_disabled_returns_empty(self, disabled_env):
        bridge = _make_bridge(["b1", "b2"])
        result = render_context(bridge)
        assert result == ""

    def test_enabled_returns_content(self, enabled_env):
        bridge = _make_bridge(["b1", "b2"])
        result = render_context(bridge)
        assert result != ""
        assert "Attention Focus" in result


class TestContextRendering:
    def test_header_present(self, enabled_env):
        bridge = _make_bridge(["b1"])
        result = render_context(bridge)
        assert result.startswith("## Attention Focus (ECAN)")

    def test_belief_ids_in_output(self, enabled_env):
        bridge = _make_bridge(["alpha", "beta"])
        result = render_context(bridge)
        assert "alpha" in result
        assert "beta" in result

    def test_claim_state_shown(self, enabled_env):
        bridge = _make_bridge(["b1"], claim_states={"b1": "KernelChecked"})
        result = render_context(bridge)
        assert "KernelChecked" in result

    def test_default_claim_state_self_reported(self, enabled_env):
        bridge = _make_bridge(["b1"])
        result = render_context(bridge)
        assert "SelfReported" in result

    def test_evidence_count_shown(self, enabled_env):
        bridge = _make_bridge(["b1"], evidence={"b1": ["e1", "e2", "e3"]})
        result = render_context(bridge)
        assert "ev=3" in result

    def test_zero_evidence_shown(self, enabled_env):
        bridge = _make_bridge(["b1"], evidence={})
        result = render_context(bridge)
        assert "ev=0" in result

    def test_stimulus_class_dash_without_injector(self, enabled_env):
        bridge = _make_bridge(["b1"])
        result = render_context(bridge)
        assert "stim=-" in result


class TestCharBudget:
    def test_respects_max_chars(self, enabled_env):
        # Create many beliefs to force truncation
        many = [f"belief_{i:04d}" for i in range(200)]
        bridge = _make_bridge(many)
        result = render_context(bridge, max_chars=200)
        assert len(result) <= 300  # Allow small overflow for ... line
        assert result.endswith("...") or "..." in result

    def test_empty_beliefs_returns_empty(self, enabled_env):
        bridge = _make_bridge([])
        result = render_context(bridge)
        assert result == ""


class TestSteadfastnessIntegration:
    def test_uses_injector_when_enabled(self, enabled_env):
        """When injector is enabled, it should use steadfastness-aware ranking."""
        bridge = _make_bridge(["b1", "b2"])

        class FakeInjector:
            enabled = True
            def prioritized_with_steadfastness(self, limit=20):
                return [
                    ("b1", 5.0, 100.0, 3, EventClass.MILESTONE.value),
                    ("b2", 3.0, 50.0, 1, EventClass.ERRATUM.value),
                ]

        result = render_context(bridge, injector=FakeInjector())
        assert "MILESTONE" in result
        assert "ERRATUM" in result

    def test_falls_back_without_injector(self, enabled_env):
        bridge = _make_bridge(["b1", "b2"])
        result = render_context(bridge, injector=None)
        assert "b1" in result
        assert "b2" in result
        assert "stim=-" in result


class TestExceptionSafety:
    def test_exception_returns_empty(self, enabled_env):
        class BrokenBridge:
            def get_prioritized_beliefs(self, limit=20):
                raise RuntimeError("boom")
            def get_evidence_map(self):
                raise RuntimeError("boom")
            def get_claim_states(self):
                raise RuntimeError("boom")

        result = render_context(BrokenBridge())
        assert result == ""


class TestContextBlock:
    def test_context_block_wrapper(self, enabled_env):
        bridge = _make_bridge(["b1"])
        result = context_block(bridge)
        assert "b1" in result
