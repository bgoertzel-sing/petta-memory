"""Tests for WMTM RecallBridge."""
import pytest
from unittest.mock import MagicMock, patch
from petta_memory.wmtm import WMTMStore
from petta_memory.wmtm_recall import RecallBridge, extract_keywords


class TestExtractKeywords:
    def test_basic(self):
        kws = extract_keywords("the sky is blue")
        assert "sky" in kws
        assert "blue" in kws
        assert "the" not in kws
        assert "is" not in kws

    def test_min_length(self):
        kws = extract_keywords("ab cd efgh")
        assert "ab" not in kws  # too short
        assert "efgh" in kws

    def test_empty(self):
        assert extract_keywords("") == []

    def test_stop_words(self):
        kws = extract_keywords("what is the status of the project")
        assert "what" not in kws
        assert "status" in kws
        assert "project" in kws

    def test_case_insensitive(self):
        kws = extract_keywords("ECAN Tuning Results")
        assert "ecan" in kws
        assert "tuning" in kws
        assert "results" in kws


class TestRecallBridge:
    def _make_store_mock(self, clusters_by_kw=None):
        """Create a mock MediumMemoryStore."""
        store = MagicMock()
        clusters_by_kw = clusters_by_kw or {}

        def query_about(entity, limit=20):
            return clusters_by_kw.get(entity, [])

        store.query_about.side_effect = query_about
        store.query_cluster.return_value = None
        return store

    def _make_ecan_mock(self, prioritized=None, evidence_map=None):
        """Create a mock ECANBridge."""
        ecan = MagicMock()
        ecan.get_prioritized_beliefs.return_value = prioritized or {}
        ecan.get_evidence_map.return_value = evidence_map or {}
        return ecan

    def _make_cluster(self, cid, text):
        """Create a mock cluster object."""
        c = MagicMock()
        c.id = cid
        c.text = text
        return c

    def test_recall_basic(self):
        clusters = {
            "sky": [self._make_cluster("c1", "sky is blue")],
            "blue": [self._make_cluster("c1", "sky is blue"), self._make_cluster("c2", "ocean is blue")],
        }
        store = self._make_store_mock(clusters)
        ecan = self._make_ecan_mock(prioritized={"c1": 50.0, "c2": 10.0})
        wmtm = WMTMStore(capacity=50)

        bridge = RecallBridge(store, ecan, wmtm)
        admitted = bridge.recall("sky blue", top_k=10)

        assert len(admitted) >= 1
        assert "c1" in admitted
        assert "c2" in admitted
        # c1 matched 2 keywords, c2 matched 1
        item_c1 = wmtm.get("c1")
        item_c2 = wmtm.get("c2")
        assert item_c1.sti >= item_c2.sti

    def test_recall_no_keywords(self):
        store = self._make_store_mock()
        ecan = self._make_ecan_mock()
        wmtm = WMTMStore()
        bridge = RecallBridge(store, ecan, wmtm)
        admitted = bridge.recall("the is a")
        assert admitted == []

    def test_recall_no_matches(self):
        store = self._make_store_mock({"nonexistent": []})
        ecan = self._make_ecan_mock()
        wmtm = WMTMStore()
        bridge = RecallBridge(store, ecan, wmtm)
        admitted = bridge.recall("nonexistent")
        assert admitted == []

    def test_recall_top_k_limit(self):
        clusters = {}
        for i in range(10):
            c = self._make_cluster(f"c{i}", f"item {i}")
            clusters.setdefault("match", []).append(c)
        store = self._make_store_mock(clusters)
        ecan = self._make_ecan_mock()
        wmtm = WMTMStore(capacity=50)
        bridge = RecallBridge(store, ecan, wmtm)
        admitted = bridge.recall("match", top_k=5)
        assert len(admitted) == 5

    def test_spreading_activation_basic(self):
        evidence_map = {
            "c1": ["c2", "c3"],
            "c2": ["c1"],
            "c3": ["c1", "c4"],
            "c4": ["c3"],
        }
        store = self._make_store_mock()
        store.query_cluster.return_value = None
        ecan = self._make_ecan_mock(evidence_map=evidence_map)
        wmtm = WMTMStore(capacity=50)

        # Seed c1 in WMTM
        wmtm.admit("c1", "seed", sti=50.0)

        bridge = RecallBridge(store, ecan, wmtm)
        new_ids = bridge.spreading_activation(["c1"], depth=2, max_new=10)

        assert "c2" in new_ids or "c2" in wmtm
        assert "c3" in new_ids or "c3" in wmtm
        # c4 is 2 hops away
        assert "c4" in new_ids or "c4" in wmtm

    def test_spreading_activation_threshold(self):
        evidence_map = {"c1": ["c2"]}
        store = self._make_store_mock()
        ecan = self._make_ecan_mock(evidence_map=evidence_map)
        wmtm = WMTMStore()
        wmtm.admit("c1", "seed", sti=0.5)  # low STI: 0.5*0.7=0.35 < 0.5 threshold

        bridge = RecallBridge(store, ecan, wmtm)
        new_ids = bridge.spreading_activation(["c1"], depth=2, threshold=0.5)
        assert len(new_ids) == 0

    def test_spreading_activation_already_in_wmtm(self):
        evidence_map = {"c1": ["c2"]}
        store = self._make_store_mock()
        ecan = self._make_ecan_mock(evidence_map=evidence_map)
        wmtm = WMTMStore()
        wmtm.admit("c1", "seed", sti=50.0)
        wmtm.admit("c2", "already here", sti=5.0)

        bridge = RecallBridge(store, ecan, wmtm)
        new_ids = bridge.spreading_activation(["c1"], depth=1)
        # c2 already in WMTM - should boost STI, not re-admit
        assert "c2" not in new_ids
        item = wmtm.get("c2")
        assert item.sti > 5.0  # boosted

    def test_recall_and_spread(self):
        evidence_map = {"c1": ["c2", "c3"]}
        clusters = {"match": [self._make_cluster("c1", "matched item")]}
        store = self._make_store_mock(clusters)
        ecan = self._make_ecan_mock(evidence_map=evidence_map)
        wmtm = WMTMStore(capacity=50)

        bridge = RecallBridge(store, ecan, wmtm)
        result = bridge.recall_and_spread("match", top_k=5, spread_depth=2)
        assert "c1" in result
        assert len(result) >= 2  # c1 + at least one spread

    def test_spreading_activation_no_evidence(self):
        store = self._make_store_mock()
        ecan = self._make_ecan_mock(evidence_map={})
        wmtm = WMTMStore()
        wmtm.admit("c1", "seed")
        bridge = RecallBridge(store, ecan, wmtm)
        result = bridge.spreading_activation(["c1"])
        assert result == []

    def test_recall_ecan_exception_handling(self):
        """RecallBridge should gracefully handle ECAN failures."""
        clusters = {"test": [self._make_cluster("c1", "test item")]}
        store = self._make_store_mock(clusters)
        ecan = MagicMock()
        ecan.get_prioritized_beliefs.side_effect = Exception("ECAN down")
        wmtm = WMTMStore(capacity=50)
        bridge = RecallBridge(store, ecan, wmtm)
        # Should still work, just with STI=0 for all
        admitted = bridge.recall("test")
        assert "c1" in admitted
