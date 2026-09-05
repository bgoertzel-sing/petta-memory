"""Tests for the ECAN bridge connecting store beliefs to attention dynamics."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.store import MediumMemoryStore
from petta_memory.ecan_bridge import ECANBridge
from petta_memory.ecan import AttentionValue


def _make_belief_cluster(
    belief_id: str,
    content: str,
    strength: str = "0.9",
    confidence: str = "0.8",
    evidence_sources: list[str] | None = None,
    cluster_id: str = "mc-test",
) -> str:
    """Build a minimal valid MemoryCluster with a DerivedBelief."""
    atoms = [
        f"(MemoryCluster {cluster_id})",
        f"(SchemaVersion {cluster_id} medium-memory-v1)",
        f"(ClusterType {cluster_id} evidence)",
        f"(ClusterOpenedAt {cluster_id} 2026-01-01T00:00:00Z)",
        f"(ClusterSource {cluster_id} test)",
        f"(Contains {cluster_id} {belief_id})",
        f"(DerivedBelief {belief_id})",
        f"(BeliefContent {belief_id} {content})",
        f"(TruthValue {belief_id} (STV {strength} {confidence}))",
    ]
    if evidence_sources:
        for src in evidence_sources:
            atoms.append(f"(Contains {cluster_id} {src})")
            atoms.append(f"(ObservedEvent {src})")
            atoms.append(f"(EvidenceFor {belief_id} {src})")
            atoms.append(f"(EvidenceSupportCount {belief_id} 1)")
            atoms.append(f"(EvidenceOppositionCount {belief_id} 0)")
    return "\n".join(atoms) + "\n"


def _make_promoted_belief_cluster(
    belief_id: str,
    content: str,
    strength: str = "0.9",
    confidence: str = "0.8",
    cluster_id: str = "mc-prom",
) -> str:
    """Build a cluster with a promoted DerivedBelief (PromotionEvent)."""
    pe_id = f"pe-{belief_id}"
    atoms = [
        f"(MemoryCluster {cluster_id})",
        f"(SchemaVersion {cluster_id} medium-memory-v1)",
        f"(ClusterType {cluster_id} evidence)",
        f"(ClusterOpenedAt {cluster_id} 2026-01-01T00:00:00Z)",
        f"(ClusterSource {cluster_id} test)",
        f"(Contains {cluster_id} {belief_id})",
        f"(Contains {cluster_id} {pe_id})",
        f"(DerivedBelief {belief_id})",
        f"(BeliefContent {belief_id} {content})",
        f"(TruthValue {belief_id} (STV {strength} {confidence}))",
        f"(PromotionEvent {pe_id})",
        f"(PromotesFrom {pe_id} {belief_id})",
        f"(PromotesTo {pe_id} {belief_id})",
        f"(PromotionRule {pe_id} threshold)",
        f"(PromotionTrust {pe_id} 0.9)",
        f"(PromotionDomain {pe_id} test)",
    ]
    return "\n".join(atoms) + "\n"


class TestECANBridgeSync(unittest.TestCase):
    def test_sync_from_empty_store(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            bridge = ECANBridge(store)
            result = bridge.sync_from_store()
            self.assertEqual(result["beliefs"], 0)
            self.assertEqual(result["evidence_links"], 0)

    def test_sync_extracts_beliefs(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable action1)", evidence_sources=["e1", "e2"]
            ))
            bridge = ECANBridge(store)
            result = bridge.sync_from_store()
            self.assertEqual(result["beliefs"], 1)
            self.assertEqual(result["evidence_links"], 2)
            self.assertEqual(result["new_atoms"], 3)  # b1 + e1 + e2

    def test_sync_builds_diffusion_graph(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable action1)", evidence_sources=["e1", "e2"]
            ))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            ev_map = bridge.get_evidence_map()
            self.assertIn("b1", ev_map)
            self.assertEqual(len(ev_map["b1"]), 2)

    def test_multiple_clusters(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable action1)", evidence_sources=["e1"],
                cluster_id="mc-1",
            ))
            store.append_cluster(_make_belief_cluster(
                "b2", "(Acceptable action2)", evidence_sources=["e2", "e3"],
                cluster_id="mc-2",
            ))
            bridge = ECANBridge(store)
            result = bridge.sync_from_store()
            self.assertEqual(result["beliefs"], 2)
            self.assertEqual(result["evidence_links"], 3)


class TestECANBridgeStimulate(unittest.TestCase):
    def test_stimulate_belief(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable action1)", evidence_sources=["e1"],
            ))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            bridge.stimulate_beliefs({"b1": 10.0})
            self.assertGreater(bridge.bank.get_sti("b1"), 0)

    def test_stimulate_unknown_belief_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster("b1", "(Acceptable action1)"))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            bridge.stimulate_beliefs({"unknown": 10.0})
            self.assertEqual(bridge.bank.get_sti("unknown"), 0)


class TestECANBridgeCycle(unittest.TestCase):
    def test_run_cycle_no_stimuli(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable action1)", evidence_sources=["e1"],
            ))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            result = bridge.run_cycle()
            self.assertEqual(result.cycle, 1)

    def test_run_cycle_with_stimuli(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable action1)", evidence_sources=["e1"],
            ))
            bridge = ECANBridge(store, params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
            bridge.sync_from_store()
            # Pass stimuli directly to run_cycle so it counts
            result = bridge.run_cycle(stimuli={"b1": 20.0})
            self.assertEqual(result.stimulated, 1)
            self.assertGreater(bridge.bank.get_sti("b1"), 0)

    def test_diffusion_spreads_to_evidence_sources(self):
        """STI should spread from a stimulated belief to its evidence sources."""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable action1)", evidence_sources=["e1", "e2"],
            ))
            bridge = ECANBridge(store, params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
            bridge.sync_from_store()
            # Stimulate b1 heavily
            bridge.stimulate_beliefs({"b1": 100.0})
            sti_before = bridge.bank.get_sti("e1") + bridge.bank.get_sti("e2")
            # Run cycle - diffusion should spread STI from b1 to e1, e2
            bridge.run_cycle()
            sti_after = bridge.bank.get_sti("e1") + bridge.bank.get_sti("e2")
            self.assertGreater(sti_after, sti_before)

    def test_multiple_cycles(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable action1)", evidence_sources=["e1"],
            ))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            results = bridge.run_cycles(5, stimuli_fn=lambda c: {"b1": 2.0})
            self.assertEqual(len(results), 5)


class TestECANBridgePrioritization(unittest.TestCase):
    def test_prioritized_beliefs_sorted_by_sti(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster("b1", "(Acceptable a1)", cluster_id="mc-1"))
            store.append_cluster(_make_belief_cluster("b2", "(Acceptable a2)", cluster_id="mc-2"))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            # Stimulate b2 more than b1
            bridge.stimulate_beliefs({"b1": 5.0, "b2": 20.0})
            ranked = bridge.get_prioritized_beliefs()
            self.assertEqual(len(ranked), 2)
            self.assertEqual(ranked[0][0], "b2")  # higher STI first
            self.assertEqual(ranked[1][0], "b1")

    def test_prioritized_beliefs_limit(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            for i in range(5):
                store.append_cluster(_make_belief_cluster(
                    f"b{i}", f"(Acceptable a{i})", cluster_id=f"mc-{i}",
                ))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            for i in range(5):
                bridge.stimulate_beliefs({f"b{i}": float(i)})
            ranked = bridge.get_prioritized_beliefs(limit=3)
            self.assertEqual(len(ranked), 3)

    def test_attentional_focus_beliefs(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster("b1", "(Acceptable a1)", cluster_id="mc-1"))
            bridge = ECANBridge(store, params={"MIN_AF_STI": 50, "MAX_AF_SIZE": 100})
            bridge.sync_from_store()
            bridge.stimulate_beliefs({"b1": 20.0})
            # b1 should now be in AF (if STI crossed threshold)
            af = bridge.get_attentional_focus_beliefs()
            # b1 should be in AF after stimulation
            self.assertIn("b1", af)

    def test_summary(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable a1)", evidence_sources=["e1"],
            ))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            s = bridge.summary()
            self.assertEqual(s["total_beliefs"], 1)
            self.assertEqual(s["total_atoms"], 2)
            self.assertEqual(s["evidence_links"], 1)
            self.assertEqual(s["cycle_count"], 0)

    def test_economic_stability_across_cycles(self):
        """STI should not diverge wildly across many cycles."""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster(
                "b1", "(Acceptable a1)", evidence_sources=["e1", "e2"],
                cluster_id="mc-1",
            ))
            store.append_cluster(_make_belief_cluster(
                "b2", "(Acceptable a2)", evidence_sources=["e3", "e4"],
                cluster_id="mc-2",
            ))
            bridge = ECANBridge(store, params={"MIN_AF_STI": 0, "MAX_AF_SIZE": 100})
            bridge.sync_from_store()
            bridge.run_cycles(50, stimuli_fn=lambda c: {"b1": 2.0, "b2": 2.0})
            total_sti = sum(bridge.bank.get_sti(a) for a in ["b1", "b2", "e1", "e2", "e3", "e4"])
            self.assertGreater(total_sti, 0)
            self.assertLess(total_sti, 500000)



class TestClaimStateExtraction(unittest.TestCase):
    """Tests for ClaimState atom extraction (GOLEM-Iter Phase G2)."""

    def _make_cluster_with_claim_state(self, belief_id, state, cluster_id="mc-cs"):
        """Build a valid cluster with a ClaimState atom."""
        atoms = [
            f"(MemoryCluster {cluster_id})",
            f"(SchemaVersion {cluster_id} medium-memory-v1)",
            f"(ClusterType {cluster_id} evidence)",
            f"(ClusterOpenedAt {cluster_id} 2026-01-01T00:00:00Z)",
            f"(ClusterSource {cluster_id} test)",
            f"(Contains {cluster_id} {belief_id})",
            f"(DerivedBelief {belief_id})",
            f"(BeliefContent {belief_id} (Acceptable action1))",
            f"(TruthValue {belief_id} (STV 0.9 0.8))",
            f"(ClaimState {belief_id} {state})",
        ]
        return "\n".join(atoms) + "\n"

    def test_claim_state_extracted(self):
        """ClaimState atoms are extracted by sync_from_store."""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(
                self._make_cluster_with_claim_state("b1", "KernelChecked")
            )
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            self.assertEqual(bridge.get_claim_state("b1"), "KernelChecked")

    def test_claim_state_default_self_reported(self):
        """Beliefs without ClaimState default to 'SelfReported'."""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(_make_belief_cluster("b1", "(Acceptable action1)"))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            self.assertEqual(bridge.get_claim_state("b1"), "SelfReported")

    def test_claim_states_map_includes_all_beliefs(self):
        """get_claim_states() returns a map covering all known beliefs."""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(
                self._make_cluster_with_claim_state("b1", "ExternallyVerified", "mc-1")
            )
            store.append_cluster(_make_belief_cluster("b2", "(Acceptable action2)", cluster_id="mc-2"))
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            states = bridge.get_claim_states()
            self.assertEqual(len(states), 2)
            self.assertEqual(states["b1"], "ExternallyVerified")
            self.assertEqual(states["b2"], "SelfReported")

    def test_claim_state_last_write_wins(self):
        """When multiple ClaimState atoms exist for the same belief, last one wins."""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            atoms = [
                "(MemoryCluster mc-1)",
                "(SchemaVersion mc-1 medium-memory-v1)",
                "(ClusterType mc-1 evidence)",
                "(ClusterOpenedAt mc-1 2026-01-01T00:00:00Z)",
                "(ClusterSource mc-1 test)",
                "(Contains mc-1 b1)",
                "(DerivedBelief b1)",
                f"(ClaimState b1 KernelChecked)",
                f"(ClaimState b1 ExternallyVerified)",
            ]
            store.append_cluster("\n".join(atoms) + "\n")
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            self.assertEqual(bridge.get_claim_state("b1"), "ExternallyVerified")

    def test_claim_state_in_summary(self):
        """Summary dict includes claim_states."""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            store.append_cluster(
                self._make_cluster_with_claim_state("b1", "KernelChecked")
            )
            bridge = ECANBridge(store)
            bridge.sync_from_store()
            result = bridge.run_cycle()
            # summary is in the cycle result or we can call it directly
            # Let's check summary method
            s = bridge.summary()
            self.assertIn("claim_states", s)
            self.assertEqual(s["claim_states"]["b1"], "KernelChecked")

    def test_invalid_claim_state_rejected(self):
        """Invalid ClaimState values raise ValidationError."""
        from petta_memory.store import ValidationError
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "journal.metta")
            with self.assertRaises(ValidationError):
                store.append_cluster(
                    self._make_cluster_with_claim_state("b1", "BogusState")
                )

    def test_all_valid_claim_states(self):
        """All four valid state values are accepted."""
        for state in ["SelfReported", "KernelChecked", "ExternallyVerified", "Unavailable"]:
            with tempfile.TemporaryDirectory() as td:
                store = MediumMemoryStore(Path(td) / "journal.metta")
                store.append_cluster(
                    self._make_cluster_with_claim_state("b1", state)
                )
                bridge = ECANBridge(store)
                bridge.sync_from_store()
                self.assertEqual(bridge.get_claim_state("b1"), state)


class TestCrossClusterEvidence(unittest.TestCase):
    """Regression test: EvidenceFor links in a different cluster than DerivedBelief."""

    def test_cross_cluster_evidence(self):
        """EvidenceFor in cluster B should be found for belief in cluster A."""
        # Cluster A: has the belief but NO evidence
        cluster_a = _make_belief_cluster(
            "bel-cross-test",
            '"cross-cluster belief"',
            cluster_id="mc-belief-only",
        )
        # Cluster B: has the evidence link (no DerivedBelief here)
        cluster_b = "\n".join([
            "(MemoryCluster mc-evidence-only)",
            "(SchemaVersion mc-evidence-only medium-memory-v1)",
            "(ClusterType mc-evidence-only evidence)",
            "(ClusterOpenedAt mc-evidence-only 2026-01-01T00:00:00Z)",
            "(ClusterSource mc-evidence-only test)",
            "(Contains mc-evidence-only ev-cross-1)",
            "(ObservedEvent ev-cross-1)",
            "(EvidenceFor bel-cross-test ev-cross-1)",
            "(EvidenceSupportCount bel-cross-test 1)",
            "(EvidenceOppositionCount bel-cross-test 0)",
            "",
        ])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".metta", delete=False
        ) as f:
            f.write(cluster_a + "\n" + cluster_b)
            f.flush()
            store = MediumMemoryStore(f.name)

        bridge = ECANBridge(store)
        stats = bridge.sync_from_store()

        # Belief should be found
        self.assertIn("bel-cross-test", bridge._belief_ids)
        # Evidence link should be found despite being in a different cluster
        self.assertIn("bel-cross-test", bridge._evidence_map)
        self.assertIn("ev-cross-1", bridge._evidence_map["bel-cross-test"])
        # Stats should reflect the link
        self.assertEqual(stats["evidence_links"], 1)


if __name__ == "__main__":
    unittest.main()
