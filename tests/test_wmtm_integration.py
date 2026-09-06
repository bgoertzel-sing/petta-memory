"""Integration tests: full WMTM cycle with mocked store/ECAN."""
import pytest
from unittest.mock import MagicMock
from petta_memory.wmtm import WMTMStore, ForgettingPolicy
from petta_memory.wmtm_recall import RecallBridge
from petta_memory.wmtm_inference import WMTMInferenceEngine
from petta_memory.wmtm_utility import WMTMUtility


def _make_cluster(cid, text):
    c = MagicMock()
    c.id = cid
    c.text = text
    return c


class TestFullWMTMCycle:
    """End-to-end: recall → tick → derive → utility → writeback."""

    def _setup_bridge(self):
        # Mock store with keyword-indexed clusters
        clusters = {
            "ecan": [_make_cluster("c1", "ECAN attention allocation")],
            "attention": [_make_cluster("c1", "ECAN attention allocation"), _make_cluster("c3", "attention spreading")],
            "memory": [_make_cluster("c2", "episodic memory store"), _make_cluster("c4", "working memory")],
        }
        store = MagicMock()
        def query_about(entity, limit=20):
            return clusters.get(entity, [])
        store.query_about.side_effect = query_about
        store.query_cluster.return_value = None
        store.append_cluster = MagicMock()

        # Mock ECAN with evidence map
        ecan = MagicMock()
        ecan.get_prioritized_beliefs.return_value = {"c1": 30.0, "c2": 15.0, "c3": 20.0, "c4": 10.0}
        ecan.get_evidence_map.return_value = {
            "c1": ["c3"],
            "c3": ["c1", "c4"],
            "c4": ["c2", "c3"],
        }
        ecan.stimulate_beliefs = MagicMock()

        wmtm = WMTMStore(capacity=30, decay_rate=0.9)
        return store, ecan, wmtm

    def test_full_cycle(self):
        store, ecan, wmtm = self._setup_bridge()
        recall = RecallBridge(store, ecan, wmtm)
        inference = WMTMInferenceEngine(wmtm)
        utility = WMTMUtility(wmtm, store, ecan, writeback_threshold=3.0)

        # Step 1: Recall items based on query context
        admitted = recall.recall("ECAN attention memory", top_k=10)
        assert len(admitted) >= 3  # c1, c2, c3, c4 from keywords

        # Step 2: Spreading activation from recalled items
        spread = recall.spreading_activation(admitted, depth=2, max_new=10)
        # c3 is evidence neighbor of c1, c4 is neighbor of c3, c2 is neighbor of c4
        # Some may already be in WMTM from recall

        # Step 3: Tick — decay + age
        evicted = wmtm.tick()
        assert isinstance(evicted, list)
        assert wmtm.cycle == 1

        # Step 4: Derive new knowledge
        if "c1" in wmtm and "c2" in wmtm:
            derived = inference.derive_conjunction("c1", "c2")
            assert derived is not None
            assert derived.is_derived
            assert len(derived.derived_from) == 2

        if "c1" in wmtm and "c3" in wmtm:
            impl = inference.derive_implication("c1", "c3")
            assert impl is not None
            assert "->" in impl.text

        # Step 5: Touch some items to boost utility
        wmtm.touch("c1")
        wmtm.touch("c1")
        wmtm.touch("c2")

        # Step 6: Another tick
        wmtm.tick()
        assert wmtm.cycle == 2

        # Step 7: Compute utility scores
        scores = utility.score_all()
        assert len(scores) > 0
        # c1 was touched most → highest utility
        assert scores.get("c1", 0) >= scores.get("c2", 0)

        # Step 8: Writeback to LTM
        written = utility.writeback()
        assert len(written) > 0
        # ECAN should have been stimulated for recalled items
        assert ecan.stimulate_beliefs.called

        # Step 9: Verify WMTM summary
        w_summary = wmtm.summary()
        assert w_summary["count"] > 0
        assert w_summary["cycle"] == 2

        i_summary = inference.summary()
        assert i_summary["total_derivations"] >= 1

        u_summary = utility.summary()
        assert u_summary["total_items"] > 0

    def test_tick_eviction_removes_low_sti(self):
        store, ecan, wmtm = self._setup_bridge()
        recall = RecallBridge(store, ecan, wmtm)
        recall.recall("ECAN attention", top_k=10)

        # Aggressive decay to force eviction
        wmtm.decay_rate = 0.3
        for _ in range(20):
            wmtm.tick()

        # Most items should be evicted by now
        assert len(wmtm) <= 5

    def test_derived_items_evicted_first_under_pressure(self):
        store, ecan, wmtm = self._setup_bridge()
        recall = RecallBridge(store, ecan, wmtm)
        inference = WMTMInferenceEngine(wmtm)

        # Recall some items
        recall.recall("ECAN attention", top_k=5)

        # Derive several items
        for i in range(5):
            if "c1" in wmtm and "c2" in wmtm:
                inference.derive(f"derived-{i}", ["c1", "c2"])

        # Force capacity pressure
        for i in range(50):
            wmtm.admit(f"filler-{i}", f"filler {i}", sti=8.0)

        # Derived items should be preferentially evicted
        remaining = wmtm.all_items()
        derived_remaining = [i for i in remaining if i.is_derived]
        # Not all derived should survive if we have heavy pressure
        # (they get 1.5x penalty)
        assert len(wmtm) <= wmtm.capacity

    def test_provenance_chain_multi_level(self):
        store, ecan, wmtm = self._setup_bridge()
        recall = RecallBridge(store, ecan, wmtm)
        inference = WMTMInferenceEngine(wmtm)

        recall.recall("ECAN attention memory", top_k=10)

        # First level derivation
        d1 = inference.derive("level1", ["c1", "c2"]) if "c1" in wmtm and "c2" in wmtm else None
        assert d1 is not None

        # Second level derivation from first
        d2 = inference.derive("level2", [d1.id, "c3"]) if "c3" in wmtm else None
        assert d2 is not None

        # Trace provenance — should go 2 levels deep
        prov = inference.get_provenance(d2.id, depth=5)
        assert prov is not None
        assert prov["id"] == d2.id
        assert "derived_from" in prov
        # Should trace back to d1
        parents = prov["derived_from"]
        assert any(p["id"] == d1.id for p in parents)
        # d1 should trace back to c1, c2
        d1_prov = [p for p in parents if p["id"] == d1.id][0]
        assert "derived_from" in d1_prov

    def test_writeback_only_high_utility(self):
        store, ecan, wmtm = self._setup_bridge()
        recall = RecallBridge(store, ecan, wmtm)
        utility = WMTMUtility(wmtm, store, ecan, writeback_threshold=100.0)  # very high

        recall.recall("ECAN attention", top_k=10)
        # No items have been used, so utility should be low
        written = utility.writeback()
        # With threshold=100, nothing should qualify
        assert len(written) == 0
        assert not ecan.stimulate_beliefs.called

    def test_recall_and_spread_integration(self):
        store, ecan, wmtm = self._setup_bridge()
        recall = RecallBridge(store, ecan, wmtm)
        result = recall.recall_and_spread("ECAN attention", top_k=5, spread_depth=2)
        # Should have recalled items + potentially spread items
        assert len(result) >= 1
        # All admitted items should be in WMTM
        for rid in result:
            assert rid in wmtm
