"""Tests for WMTM Coordinator."""
import pytest
from unittest.mock import MagicMock
from petta_memory.wmtm_coordinator import WMTMCoordinator


def _make_cluster(cid, text):
    c = MagicMock()
    c.id = cid
    c.text = text
    return c


class TestWMTMCoordinator:
    def _setup(self, **kwargs):
        clusters = {
            "ecan": [_make_cluster("c1", "ECAN tuning results")],
            "tuning": [_make_cluster("c1", "ECAN tuning results"), _make_cluster("c2", "parameter tuning")],
            "results": [_make_cluster("c3", "test results pass")],
        }
        store = MagicMock()
        def query_about(entity, limit=20):
            return clusters.get(entity, [])
        store.query_about.side_effect = query_about
        store.query_cluster.return_value = None
        store.append_cluster = MagicMock()

        ecan = MagicMock()
        ecan.get_prioritized_beliefs.return_value = {"c1": 30.0, "c2": 15.0, "c3": 20.0}
        ecan.get_evidence_map.return_value = {"c1": ["c3"], "c3": ["c1"]}
        ecan.stimulate_beliefs = MagicMock()

        return WMTMCoordinator(store, ecan, capacity=30, **kwargs)

    def test_on_context(self):
        coord = self._setup()
        admitted = coord.on_context("ECAN tuning results")
        assert len(admitted) >= 1
        assert coord.turn_count == 0

    def test_on_tick(self):
        coord = self._setup()
        coord.on_context("ECAN tuning")
        evicted = coord.on_tick()
        assert isinstance(evicted, list)
        assert coord.wmtm.cycle == 1

    def test_on_derive(self):
        coord = self._setup()
        coord.on_context("ECAN tuning results")
        if "c1" in coord.wmtm and "c3" in coord.wmtm:
            deriv_id = coord.on_derive("combined insight", ["c1", "c3"])
            assert deriv_id is not None
            assert coord.inference.derivation_count == 1

    def test_on_derive_no_sources(self):
        coord = self._setup()
        result = coord.on_derive("test", ["nonexistent"])
        assert result is None

    def test_on_touch(self):
        coord = self._setup()
        coord.on_context("ECAN tuning")
        if "c1" in coord.wmtm:
            coord.on_touch("c1")
            item = coord.wmtm.get("c1")
            assert item.use_count == 1

    def test_on_turn_end(self):
        coord = self._setup()
        coord.on_context("ECAN tuning results")
        coord.on_touch("c1")
        written = coord.on_turn_end()
        assert coord.turn_count == 1
        assert isinstance(written, list)

    def test_on_turn_end_auto_writeback_disabled(self):
        coord = self._setup(auto_writeback=False)
        coord.on_context("ECAN tuning")
        written = coord.on_turn_end()
        assert written == []
        assert coord.turn_count == 1

    def test_get_active_context(self):
        coord = self._setup()
        coord.on_context("ECAN tuning results")
        ctx = coord.get_active_context(limit=5)
        assert isinstance(ctx, list)
        assert len(ctx) >= 1
        assert "id" in ctx[0]
        assert "text" in ctx[0]
        assert "sti" in ctx[0]

    def test_full_summary(self):
        coord = self._setup()
        coord.on_context("ECAN tuning")
        coord.on_turn_end()
        s = coord.full_summary()
        assert "turn" in s
        assert s["turn"] == 1
        assert "wmtm" in s
        assert "inference" in s
        assert "utility" in s

    def test_full_lifecycle(self):
        """Complete lifecycle: context → tick → derive → touch → turn_end."""
        coord = self._setup()
        # Turn 1
        coord.on_context("ECAN tuning results")
        coord.on_tick()
        active = coord.get_active_context()
        assert len(active) > 0

        # Derive something
        ids = [i["id"] for i in active[:2]]
        if len(ids) >= 2:
            deriv = coord.on_derive("synthesis", ids)
            assert deriv is not None

        coord.on_touch(ids[0])
        coord.on_turn_end()
        assert coord.turn_count == 1

        # Turn 2 — new context
        coord.on_context("tuning results")
        coord.on_turn_end()
        assert coord.turn_count == 2

        s = coord.full_summary()
        assert s["wmtm"]["count"] > 0
