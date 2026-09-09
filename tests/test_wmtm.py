"""Tests for WMTM foundational data structures."""
import pytest
from petta_memory.wmtm import WMTMItem, WMTMStore, ForgettingPolicy


class TestWMTMItem:
    def test_creation(self):
        item = WMTMItem(id="b1", text="sky is blue", source_type="recalled")
        assert item.id == "b1"
        assert item.source_type == "recalled"
        assert item.is_derived is False
        assert item.sti == 10.0
        assert item.use_count == 0

    def test_derived_item(self):
        item = WMTMItem(id="d1", text="derived fact", source_type="derived", derived_from=["b1", "b2"])
        assert item.is_derived is True
        assert len(item.derived_from) == 2

    def test_touch(self):
        item = WMTMItem(id="b1", text="test")
        item.touch()
        assert item.use_count == 1
        assert item.last_used == 0
        assert item.utility == 1.0
        item.touch()
        assert item.use_count == 2

    def test_decay_sti(self):
        item = WMTMItem(id="b1", text="test", sti=100.0)
        item.decay_sti(rate=0.85)
        assert abs(item.sti - 85.0) < 0.01
        assert item.age == 1
        item.decay_sti(rate=0.85)
        assert abs(item.sti - 72.25) < 0.01
        assert item.age == 2


class TestForgettingPolicy:
    def test_sti_threshold_eviction(self):
        policy = ForgettingPolicy(sti_threshold=5.0)
        item = WMTMItem(id="b1", text="test", sti=2.0)
        assert policy.should_evict(item) is True

    def test_sti_above_threshold(self):
        policy = ForgettingPolicy(sti_threshold=5.0)
        item = WMTMItem(id="b1", text="test", sti=20.0)
        assert policy.should_evict(item) is False

    def test_max_age_eviction(self):
        policy = ForgettingPolicy(max_age=10)
        item = WMTMItem(id="b1", text="test", sti=100.0)
        item.age = 10
        assert policy.should_evict(item) is True

    def test_capacity_eviction(self):
        policy = ForgettingPolicy(max_items=3)
        items = [
            WMTMItem(id=f"b{i}", text=f"t{i}", sti=float(10 - i))
            for i in range(5)
        ]
        evict = policy.evict_candidates(items)
        assert len(evict) == 2  # 5 items, max 3, evict 2
        # Lowest STI items should be evicted first
        assert "b4" in evict
        assert "b3" in evict

    def test_derived_items_evicted_first(self):
        policy = ForgettingPolicy(max_items=2, derived_eviction_mult=1.5)
        items = [
            WMTMItem(id="r1", text="recalled", sti=10.0, source_type="recalled"),
            WMTMItem(id="d1", text="derived", sti=10.0, source_type="derived"),
            WMTMItem(id="r2", text="recalled2", sti=10.0, source_type="recalled"),
        ]
        evict = policy.evict_candidates(items)
        assert len(evict) == 1
        assert "d1" in evict  # derived item evicted first due to penalty


class TestWMTMStore:
    def test_admit(self):
        store = WMTMStore(capacity=10)
        item = store.admit("b1", "sky is blue", sti=15.0)
        assert item.id == "b1"
        assert item.sti == 15.0
        assert len(store) == 1

    def test_admit_duplicate(self):
        store = WMTMStore()
        store.admit("b1", "test", sti=10.0)
        store.admit("b1", "test", sti=20.0)
        assert len(store) == 1
        item = store.get("b1")
        assert item.sti == 20.0
        assert item.use_count == 1  # touched on re-admit

    def test_get(self):
        store = WMTMStore()
        store.admit("b1", "test")
        assert store.get("b1") is not None
        assert store.get("nonexistent") is None

    def test_touch(self):
        store = WMTMStore()
        store.admit("b1", "test", sti=10.0)
        store.touch("b1")
        item = store.get("b1")
        assert item.use_count == 1

    def test_remove(self):
        store = WMTMStore()
        store.admit("b1", "test")
        store.remove("b1")
        assert len(store) == 0
        assert "b1" not in store

    def test_get_active_set(self):
        store = WMTMStore()
        store.admit("b1", "t1", sti=5.0)
        store.admit("b2", "t2", sti=50.0)
        store.admit("b3", "t3", sti=20.0)
        active = store.get_active_set(limit=2)
        assert len(active) == 2
        assert active[0].id == "b2"  # highest STI
        assert active[1].id == "b3"

    def test_tick_decay(self):
        store = WMTMStore(decay_rate=0.5)
        store.admit("b1", "test", sti=100.0)
        evicted = store.tick()
        assert len(evicted) == 0
        item = store.get("b1")
        assert abs(item.sti - 50.0) < 0.01
        assert item.age == 1

    def test_tick_eviction(self):
        store = WMTMStore(capacity=10, decay_rate=0.3)
        store.admit("b1", "test", sti=10.0)
        # Tick until STI drops below threshold (5.0)
        for _ in range(5):
            store.tick()
        item = store.get("b1")
        if item:
            assert item.sti < 5.0
        # After enough ticks, should be evicted
        for _ in range(10):
            store.tick()
        assert "b1" not in store

    def test_capacity_enforcement(self):
        store = WMTMStore(capacity=3)
        for i in range(5):
            store.admit(f"b{i}", f"t{i}", sti=float(10 - i))
        assert len(store) <= 3

    def test_summary(self):
        store = WMTMStore()
        store.admit("b1", "t1", source_type="recalled", sti=10.0)
        store.admit("b2", "t2", source_type="derived", sti=30.0)
        s = store.summary()
        assert s["count"] == 2
        assert s["recalled"] == 1
        assert s["derived"] == 1
        assert s["avg_sti"] == 20.0
        assert s["max_sti"] == 30.0
        assert s["min_sti"] == 10.0

    def test_contains(self):
        store = WMTMStore()
        store.admit("b1", "test")
        assert "b1" in store
        assert "b2" not in store

    def test_all_items(self):
        store = WMTMStore()
        store.admit("b1", "t1")
        store.admit("b2", "t2")
        items = store.all_items()
        assert len(items) == 2

    def test_derived_item_admission(self):
        store = WMTMStore()
        item = store.admit(
            "d1", "derived fact",
            source_type="derived",
            derived_from=["b1", "b2"],
            sti=25.0,
        )
        assert item.is_derived
        assert item.derived_from == ["b1", "b2"]
        assert item.sti == 25.0
