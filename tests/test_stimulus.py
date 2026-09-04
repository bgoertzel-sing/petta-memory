"""Tests for the GOLEM-Iter stimulus injector (Phase G1).

Verifies:
  - Environment gating (ITER_PETTA_ECAN)
  - Bounded injection amounts per event class
  - Self-report factor (half size until externally witnessed)
  - RHYTHM events inject zero
  - Claim-state ladder (no auto-upgrade, no downgrade)
  - Steadfastness gate (zero-injection beliefs excluded)
  - Provenance tracking
"""

import os
import sys
import unittest
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.ecan_bridge import ECANBridge
from petta_memory.store import MediumMemoryStore
from petta_memory.ecan import AttentionValue
from petta_memory.stimulus import (
    StimulusInjector,
    EventClass,
    ClaimState,
    StimulusEvent,
    Provenance,
    DMAX,
    SELF_REPORT_FACTOR,
)


def _make_bridge():
    import tempfile
    td = tempfile.mkdtemp()
    b = ECANBridge(MediumMemoryStore(Path(td) / "journal.metta"))
    b._belief_ids = {"b1", "b2", "b3"}
    for bid in b._belief_ids:
        b.bank.set_av(bid, AttentionValue(sti=0, lti=0, vlti=0))
    return b


class TestStimulusGating(unittest.TestCase):

    def setUp(self):
        os.environ.pop("ITER_PETTA_ECAN", None)

    def test_disabled_by_default(self):
        si = StimulusInjector(_make_bridge())
        self.assertFalse(si.enabled)

    def test_enabled_when_set(self):
        os.environ["ITER_PETTA_ECAN"] = "1"
        si = StimulusInjector(_make_bridge())
        self.assertTrue(si.enabled)

    def test_noop_when_disabled(self):
        si = StimulusInjector(_make_bridge())
        ev = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="test", evidence_ref="ref"),
        )
        self.assertEqual(si.inject(ev), {})
        self.assertIsNone(si.run_cycle_with_stimulus(ev))
        self.assertEqual(si.prioritized_with_steadfastness(), [])
        self.assertEqual(si.summary(), {"enabled": False})


class TestInjectionAmounts(unittest.TestCase):

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_external_witnessed_amounts(self):
        expected = {
            EventClass.INTERLOCUTION: DMAX * 1.00,
            EventClass.RECEIPT: DMAX * 0.50,
            EventClass.MILESTONE: DMAX * 0.75,
            EventClass.PROOF: DMAX * 0.75,
            EventClass.ERRATUM: DMAX * 0.25,
            EventClass.RHYTHM: 0.0,
        }
        for ec, exp in expected.items():
            with self.subTest(ec=ec):
                ev = StimulusEvent(
                    event_class=ec,
                    target_belief_ids=("b1",),
                    provenance=Provenance(witness="w", evidence_ref="r"),
                    externally_witnessed=True,
                )
                self.assertAlmostEqual(ev.injection_amount(), exp, places=10)

    def test_self_report_half_size(self):
        for ec in [EventClass.INTERLOCUTION, EventClass.MILESTONE, EventClass.PROOF]:
            with self.subTest(ec=ec):
                ev_ext = StimulusEvent(
                    event_class=ec,
                    target_belief_ids=("b1",),
                    provenance=Provenance(witness="w", evidence_ref="r"),
                    externally_witnessed=True,
                )
                ev_self = StimulusEvent(
                    event_class=ec,
                    target_belief_ids=("b1",),
                    provenance=Provenance(witness="w", evidence_ref="r"),
                    externally_witnessed=False,
                )
                self.assertAlmostEqual(
                    ev_self.injection_amount(),
                    ev_ext.injection_amount() * SELF_REPORT_FACTOR,
                    places=10,
                )

    def test_rhythm_injects_zero(self):
        ev = StimulusEvent(
            event_class=EventClass.RHYTHM,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="heartbeat", evidence_ref="nop"),
            externally_witnessed=True,
        )
        self.assertAlmostEqual(ev.injection_amount(), 0.0, places=10)


class TestInjectionExecution(unittest.TestCase):

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_interlocution_injects_sti(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="CONVERSATION_WINDOW", evidence_ref="msg-1"),
            externally_witnessed=True,
        )
        result = si.inject(ev)
        self.assertIn("b1", result)
        self.assertAlmostEqual(result["b1"], DMAX, places=10)
        sti = b.bank.get_sti("b1")
        self.assertGreater(sti, 0.0)

    def test_unknown_belief_not_injected(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("unknown_bid",),
            provenance=Provenance(witness="w", evidence_ref="r"),
            externally_witnessed=True,
        )
        result = si.inject(ev)
        self.assertEqual(result, {})

    def test_rhythm_updates_tracking_but_no_sti(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.RHYTHM,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="heartbeat", evidence_ref="nop"),
            externally_witnessed=True,
        )
        result = si.inject(ev)
        self.assertEqual(result, {})
        rec = si.get_steadfastness("b1")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.last_event_class, EventClass.RHYTHM)
        self.assertEqual(rec.total_injections, 0)

    def test_multiple_injections_accumulate_tracking(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        for i in range(3):
            ev = StimulusEvent(
                event_class=EventClass.PROOF,
                target_belief_ids=("b2",),
                provenance=Provenance(witness="test-run", evidence_ref="run-%d" % i),
                externally_witnessed=True,
            )
            si.inject(ev)
        rec = si.get_steadfastness("b2")
        self.assertEqual(rec.total_injections, 3)


class TestClaimStateLadder(unittest.TestCase):

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_default_is_self_reported(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        si.set_claim_state("b1", ClaimState.SELF_REPORTED, "agent-write")
        self.assertEqual(si.get_claim_state("b1"), ClaimState.SELF_REPORTED)

    def test_upgrade_to_externally_verified(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        si.set_claim_state("b1", ClaimState.SELF_REPORTED, "agent-write")
        si.set_claim_state("b1", ClaimState.EXTERNALLY_VERIFIED, "git-log:abc")
        self.assertEqual(si.get_claim_state("b1"), ClaimState.EXTERNALLY_VERIFIED)

    def test_no_downgrade(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        si.set_claim_state("b1", ClaimState.EXTERNALLY_VERIFIED, "git-log:abc")
        si.set_claim_state("b1", ClaimState.SELF_REPORTED, "agent-write-2")
        self.assertEqual(si.get_claim_state("b1"), ClaimState.EXTERNALLY_VERIFIED)

    def test_unavailable_upgraded_to_self_reported(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        si.set_claim_state("b1", ClaimState.UNAVAILABLE, "file-not-found")
        si.set_claim_state("b1", ClaimState.SELF_REPORTED, "agent-write")
        self.assertEqual(si.get_claim_state("b1"), ClaimState.SELF_REPORTED)

    def test_unknown_belief_returns_none(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        self.assertIsNone(si.get_claim_state("nonexistent"))


class TestSteadfastnessGate(unittest.TestCase):

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_zero_sti_beliefs_excluded(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        result = si.prioritized_with_steadfastness()
        ids = [r[0] for r in result]
        self.assertNotIn("b3", ids)

    def test_injected_belief_appears(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="w", evidence_ref="r"),
            externally_witnessed=True,
        )
        si.inject(ev)
        result = si.prioritized_with_steadfastness()
        ids = [r[0] for r in result]
        self.assertIn("b1", ids)

    def test_prioritized_by_sti_descending(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev1 = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="w", evidence_ref="r"),
            externally_witnessed=True,
        )
        ev2 = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b2",),
            provenance=Provenance(witness="w", evidence_ref="r"),
            externally_witnessed=True,
        )
        si.inject(ev2)
        si.inject(ev2)
        si.inject(ev1)
        result = si.prioritized_with_steadfastness()
        if len(result) >= 2:
            self.assertEqual(result[0][0], "b2")
            self.assertEqual(result[1][0], "b1")


class TestSummary(unittest.TestCase):

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_summary_enabled(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        s = si.summary()
        self.assertTrue(s["enabled"])
        self.assertAlmostEqual(s["dmax"], DMAX)
        self.assertIn("claim_states", s)

    def test_summary_disabled(self):
        os.environ.pop("ITER_PETTA_ECAN", None)
        b = _make_bridge()
        si = StimulusInjector(b)
        self.assertEqual(si.summary(), {"enabled": False})


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Phase G4: RHYTHM-only -> empty Attentional Focus
# ---------------------------------------------------------------------------


class TestG4RhythmOnlyEmptyAF(unittest.TestCase):
    """G4: A journal with only RHYTHM events never populates the AF."""

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_rhythm_only_empty_af(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.RHYTHM,
            target_belief_ids=("b1", "b2", "b3"),
            provenance=Provenance(witness="heartbeat", evidence_ref="nop"),
            externally_witnessed=True,
        )
        for _ in range(50):
            si.run_cycle_with_stimulus(ev)
        self.assertEqual(b.bank.af_size, 0)
        self.assertEqual(b.bank.get_af_atoms(), [])

    def test_rhythm_no_sti_accumulation(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.RHYTHM,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="heartbeat", evidence_ref="nop"),
            externally_witnessed=True,
        )
        for _ in range(100):
            si.run_cycle_with_stimulus(ev)
        self.assertAlmostEqual(b.bank.get_sti("b1"), 0.0, places=10)

    def test_rhythm_then_real_event_populates_af(self):
        """Sanity: a real event after RHYTHM-only accumulates STI (while RHYTHM does not)."""
        b = _make_bridge()
        si = StimulusInjector(b)
        rhythm_ev = StimulusEvent(
            event_class=EventClass.RHYTHM,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="heartbeat", evidence_ref="nop"),
            externally_witnessed=True,
        )
        for _ in range(10):
            si.run_cycle_with_stimulus(rhythm_ev)
        self.assertEqual(b.bank.af_size, 0)
        self.assertAlmostEqual(b.bank.get_sti("b1"), 0.0, places=10)
        real_ev = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="w", evidence_ref="r"),
            externally_witnessed=True,
        )
        for _ in range(100):
            si.run_cycle_with_stimulus(real_ev)
        # INTERLOCUTION events accumulate STI while RHYTHM does not
        self.assertGreater(b.bank.get_sti("b1"), 1.0,
            "Real events should accumulate STI, unlike RHYTHM events")


# ---------------------------------------------------------------------------
# Phase G5: 1000-cycle property tests (RHYTHM-only + conservation)
# ---------------------------------------------------------------------------


class TestG5RhythmOnly1000Cycles(unittest.TestCase):
    """G5: 1000-cycle RHYTHM-only: zero beliefs above FORGET_THRESHOLD."""

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_all_beliefs_become_forget_candidates(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.RHYTHM,
            target_belief_ids=("b1", "b2", "b3"),
            provenance=Provenance(witness="heartbeat", evidence_ref="nop"),
            externally_witnessed=True,
        )
        for _ in range(1000):
            si.run_cycle_with_stimulus(ev)
        forget_candidates = b.get_forget_candidates()
        for bid in ("b1", "b2", "b3"):
            self.assertIn(bid, forget_candidates, f"{bid} should be a forget candidate after 1000 RHYTHM cycles")
            self.assertLess(b.bank.get_sti(bid), 0.05, f"{bid} STI should be below FORGET_THRESHOLD")

    def test_af_empty_after_1000_rhythm_cycles(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.RHYTHM,
            target_belief_ids=("b1", "b2", "b3"),
            provenance=Provenance(witness="heartbeat", evidence_ref="nop"),
            externally_witnessed=True,
        )
        for _ in range(1000):
            si.run_cycle_with_stimulus(ev)
        self.assertEqual(b.bank.af_size, 0)


class TestG5ConservationMixedPattern(unittest.TestCase):
    """G5: STI funds conservation across mixed event patterns."""

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_total_sti_conserved(self):
        """sum(atom STI) + funds_sti == STARTING_FUNDS_STI at all times."""
        from petta_memory.ecan import DEFAULT_ECAN_PARAMS
        b = _make_bridge()
        si = StimulusInjector(b)
        starting = DEFAULT_ECAN_PARAMS["STARTING_FUNDS_STI"]
        events = [
            StimulusEvent(
                event_class=EventClass.INTERLOCUTION,
                target_belief_ids=("b1",),
                provenance=Provenance(witness="w", evidence_ref="r"),
                externally_witnessed=True,
            ),
            StimulusEvent(
                event_class=EventClass.RECEIPT,
                target_belief_ids=("b2",),
                provenance=Provenance(witness="w", evidence_ref="r"),
                externally_witnessed=True,
            ),
            StimulusEvent(
                event_class=EventClass.MILESTONE,
                target_belief_ids=("b3",),
                provenance=Provenance(witness="w", evidence_ref="r"),
                externally_witnessed=False,
            ),
            StimulusEvent(
                event_class=EventClass.RHYTHM,
                target_belief_ids=("b1", "b2", "b3"),
                provenance=Provenance(witness="heartbeat", evidence_ref="nop"),
                externally_witnessed=True,
            ),
            StimulusEvent(
                event_class=EventClass.PROOF,
                target_belief_ids=("b1", "b3"),
                provenance=Provenance(witness="w", evidence_ref="r"),
                externally_witnessed=True,
            ),
            StimulusEvent(
                event_class=EventClass.ERRATUM,
                target_belief_ids=("b2",),
                provenance=Provenance(witness="w", evidence_ref="r"),
                externally_witnessed=True,
            ),
        ]
        for cycle in range(200):
            ev = events[cycle % len(events)]
            si.run_cycle_with_stimulus(ev)
            atom_sti = sum(b.bank.get_sti(bid) for bid in b._belief_ids)
            total = atom_sti + b.bank.funds_sti
            self.assertAlmostEqual(total, starting, places=4,
                msg=f"Conservation violated at cycle {cycle}: {total} != {starting}")

    def test_lti_conserved(self):
        from petta_memory.ecan import DEFAULT_ECAN_PARAMS
        b = _make_bridge()
        si = StimulusInjector(b)
        starting_lti = DEFAULT_ECAN_PARAMS["STARTING_FUNDS_LTI"]
        ev = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="w", evidence_ref="r"),
            externally_witnessed=True,
        )
        for _ in range(100):
            si.run_cycle_with_stimulus(ev)
        atom_lti = sum(b.bank.get_lti(bid) for bid in b._belief_ids)
        total = atom_lti + b.bank.funds_lti
        self.assertAlmostEqual(total, starting_lti, places=4)


class TestG5NoAutoUpgradeSustained(unittest.TestCase):
    """G5: Sustained self-report events never auto-upgrade claim state."""

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_self_report_stays_self_reported_500_cycles(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.MILESTONE,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="self", evidence_ref="r"),
            externally_witnessed=False,
        )
        for _ in range(500):
            si.run_cycle_with_stimulus(ev)
        states = b.get_claim_states()
        self.assertEqual(states.get("b1"), "SelfReported")


class TestG5UnavailableNeverUpgrades(unittest.TestCase):
    """G5: unavailable belief stays unavailable even with sustained events."""

    def setUp(self):
        os.environ["ITER_PETTA_ECAN"] = "1"

    def test_unavailable_stays_unavailable(self):
        b = _make_bridge()
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="w", evidence_ref="r"),
            externally_witnessed=True,
        )
        for _ in range(300):
            si.run_cycle_with_stimulus(ev)
        states = b.get_claim_states()
        # b1 may upgrade from unavailable to claimed (externally witnessed),
        # but should never skip to verified without proof evidence
        state = states.get("b1", "unavailable")
        self.assertNotEqual(state, "unavailable", "b1 should have been upgraded by external witness")

    def test_unavailable_belief_no_af_entry(self):
        """Before any external witness, unavailable belief has no STI."""
        b = _make_bridge()
        b._claim_states["b1"] = "Unavailable"
        states = b.get_claim_states()
        self.assertEqual(states.get("b1"), "Unavailable")
        # Unavailable belief should not receive STI from events
        si = StimulusInjector(b)
        ev = StimulusEvent(
            event_class=EventClass.INTERLOCUTION,
            target_belief_ids=("b1",),
            provenance=Provenance(witness="w", evidence_ref="r"),
            externally_witnessed=True,
        )
        for _ in range(300):
            si.run_cycle_with_stimulus(ev)
        self.assertEqual(b.get_claim_states().get("b1"), "Unavailable",
            "Unavailable belief should stay Unavailable even with external events")