"""Tests for WMTM InferenceEngine and Utility."""
import pytest
from unittest.mock import MagicMock
from petta_memory.wmtm import WMTMStore
from petta_memory.wmtm_inference import WMTMInferenceEngine
from petta_memory.wmtm_utility import WMTMUtility


class TestWMTMInferenceEngine:
    def test_derive_basic(self):
        wmtm = WMTMStore(capacity=50)
        wmtm.admit("b1", "sky is blue", sti=20.0)
        wmtm.admit("b2", "ocean is blue", sti=10.0)
        engine = WMTMInferenceEngine(wmtm)
        item = engine.derive("both blue", ["b1", "b2"])
        assert item is not None
        assert item.is_derived
        assert item.derived_from == ["b1", "b2"]
        # STI should be avg * 0.8 = 15 * 0.8 = 12
        assert abs(item.sti - 12.0) < 0.5
        assert engine.derivation_count == 1

    def test_derive_no_sources(self):
        wmtm = WMTMStore()
        engine = WMTMInferenceEngine(wmtm)
        item = engine.derive("test", ["nonexistent"])
        assert item is None

    def test_derive_explicit_sti(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "test", sti=100.0)
        engine = WMTMInferenceEngine(wmtm)
        item = engine.derive("derived", ["b1"], sti=50.0)
        assert abs(item.sti - 50.0) < 0.01

    def test_derive_conjunction(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "cats", sti=20.0)
        wmtm.admit("b2", "dogs", sti=20.0)
        engine = WMTMInferenceEngine(wmtm)
        item = engine.derive_conjunction("b1", "b2")
        assert item is not None
        assert "AND" in item.text

    def test_derive_implication(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "rain", sti=20.0)
        wmtm.admit("b2", "wet ground", sti=20.0)
        engine = WMTMInferenceEngine(wmtm)
        item = engine.derive_implication("b1", "b2")
        assert item is not None
        assert "->" in item.text

    def test_derive_conjunction_missing(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "cats")
        engine = WMTMInferenceEngine(wmtm)
        item = engine.derive_conjunction("b1", "nonexistent")
        assert item is None

    def test_provenance(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "fact A", sti=20.0)
        wmtm.admit("b2", "fact B", sti=20.0)
        engine = WMTMInferenceEngine(wmtm)
        derived = engine.derive("A and B", ["b1", "b2"])
        prov = engine.get_provenance(derived.id)
        assert prov is not None
        assert prov["id"] == derived.id
        assert prov["source_type"] == "derived"
        assert len(prov["derived_from"]) == 2

    def test_provenance_recalled_item(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "simple fact")
        engine = WMTMInferenceEngine(wmtm)
        prov = engine.get_provenance("b1")
        assert prov["source_type"] == "recalled"
        assert "derived_from" not in prov

    def test_provenance_missing(self):
        wmtm = WMTMStore()
        engine = WMTMInferenceEngine(wmtm)
        prov = engine.get_provenance("nonexistent")
        assert prov is None

    def test_summary(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "fact", sti=20.0)
        engine = WMTMInferenceEngine(wmtm)
        engine.derive("derived", ["b1"], sti=15.0)
        s = engine.summary()
        assert s["total_derivations"] == 1
        assert s["active_derived"] == 1


class TestWMTMUtility:
    def _make_mocks(self):
        store = MagicMock()
        ecan = MagicMock()
        return store, ecan

    def test_compute_utility(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "test", sti=20.0)
        item = wmtm.get("b1")
        item.use_count = 3
        store, ecan = self._make_mocks()
        util = WMTMUtility(wmtm, store, ecan)
        score = util.compute_utility(item)
        # 3 * 1.0 + 20 * 0.3 - 0 = 3 + 6 = 9
        assert abs(score - 9.0) < 0.5

    def test_score_all(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "t1", sti=20.0)
        wmtm.admit("b2", "t2", sti=5.0)
        store, ecan = self._make_mocks()
        util = WMTMUtility(wmtm, store, ecan)
        scores = util.score_all()
        assert len(scores) == 2
        assert scores["b1"] > scores["b2"]

    def test_get_writeback_candidates(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "high", sti=50.0)
        item = wmtm.get("b1")
        item.use_count = 5
        wmtm.admit("b2", "low", sti=1.0)
        store, ecan = self._make_mocks()
        util = WMTMUtility(wmtm, store, ecan, writeback_threshold=5.0)
        candidates = util.get_writeback_candidates()
        ids = [c.id for c in candidates]
        assert "b1" in ids
        assert "b2" not in ids

    def test_writeback_recalled(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "fact", sti=20.0, origin_cluster="c1")
        item = wmtm.get("b1")
        item.use_count = 5
        store, ecan = self._make_mocks()
        util = WMTMUtility(wmtm, store, ecan, writeback_threshold=1.0)
        written = util.writeback()
        assert "b1" in written
        # Should have stimulated ECAN
        ecan.stimulate_beliefs.assert_called_once()

    def test_writeback_derived(self):
        from petta_memory.wmtm_inference import WMTMInferenceEngine
        wmtm = WMTMStore()
        wmtm.admit("b1", "source", sti=30.0)
        engine = WMTMInferenceEngine(wmtm)
        derived = engine.derive("new insight", ["b1"], sti=20.0)
        item = wmtm.get(derived.id)
        item.use_count = 5
        store, ecan = self._make_mocks()
        store.append_cluster = MagicMock()
        util = WMTMUtility(wmtm, store, ecan, writeback_threshold=1.0)
        written = util.writeback()
        assert derived.id in written
        store.append_cluster.assert_called_once()

    def test_writeback_specific_ids(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "fact1", sti=20.0, origin_cluster="c1")
        wmtm.admit("b2", "fact2", sti=20.0, origin_cluster="c2")
        store, ecan = self._make_mocks()
        util = WMTMUtility(wmtm, store, ecan)
        written = util.writeback(item_ids=["b1"])
        assert "b1" in written
        assert "b2" not in written

    def test_summary(self):
        wmtm = WMTMStore()
        wmtm.admit("b1", "test", sti=20.0)
        store, ecan = self._make_mocks()
        util = WMTMUtility(wmtm, store, ecan, writeback_threshold=5.0)
        s = util.summary()
        assert s["total_items"] == 1
        assert s["avg_utility"] > 0
