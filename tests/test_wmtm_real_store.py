"""C01: Real-object regression tests for F01-F08."""
from __future__ import annotations
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from petta_memory.store import MediumMemoryStore
from petta_memory.wmtm import WMTMStore, WMTMItem, ForgettingPolicy
from petta_memory.wmtm_recall import RecallBridge
from petta_memory.wmtm_inference import WMTMInferenceEngine
from petta_memory.wmtm_utility import WMTMUtility
from petta_memory.wmtm_coordinator import WMTMCoordinator
from petta_memory.ecan_bridge import ECANBridge

CA = """
(MemoryCluster mc-a)
(SchemaVersion mc-a medium-memory-v1)
(ClusterType mc-a belief-record)
(ClusterOpenedAt mc-a "2026-09-08T00:00:00Z")
(ClusterSource mc-a local-test)
(Contains mc-a belief-a)
(Contains mc-a event-a)
(DerivedBelief belief-a)
(BeliefContent belief-a (Useful memory))
(About belief-a memory)
(TruthValue belief-a (stv 0.9 0.8))
(ObservedEvent event-a)
(EvidenceFor belief-a event-a)
"""

CB = """
(MemoryCluster mc-b)
(SchemaVersion mc-b medium-memory-v1)
(ClusterType mc-b belief-record)
(ClusterOpenedAt mc-b "2026-09-08T00:00:00Z")
(ClusterSource mc-b local-test)
(Contains mc-b belief-b)
(Contains mc-b event-b)
(DerivedBelief belief-b)
(BeliefContent belief-b (Helpful system))
(About belief-b memory)
(TruthValue belief-b (stv 0.8 0.7))
(ObservedEvent event-b)
(EvidenceFor belief-b event-b)
"""

CM = """
(MemoryCluster mc-c)
(SchemaVersion mc-c medium-memory-v1)
(ClusterType mc-c belief-record)
(ClusterOpenedAt mc-c "2026-09-08T00:00:00Z")
(ClusterSource mc-c local-test)
(Contains mc-c belief-c1)
(Contains mc-c belief-c2)
(Contains mc-c event-c)
(DerivedBelief belief-c1)
(BeliefContent belief-c1 (Fast cache))
(About belief-c1 cache)
(TruthValue belief-c1 (stv 0.7 0.6))
(DerivedBelief belief-c2)
(BeliefContent belief-c2 (Stable service))
(About belief-c2 service)
(TruthValue belief-c2 (stv 0.85 0.9))
(ObservedEvent event-c)
(EvidenceFor belief-c1 event-c)
(EvidenceFor belief-c2 event-c)
"""

def _ms(*ts):
    fd, p = tempfile.mkstemp(suffix=".mem", prefix="wmtm_r_")
    os.close(fd)
    s = MediumMemoryStore(p)
    for t in ts: s.append_cluster(t)
    return s, p

def _cl(p):
    try: os.unlink(p)
    except OSError: pass
class F01_RecallIdentity(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA, CB)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.b = RecallBridge(self.s, self.e, self.w)
    def tearDown(self):
        _cl(self.p)
    def test_no_id_attr(self):
        c = self.s.query_cluster("mc-a")
        self.assertTrue(hasattr(c, "cluster_id"))
        self.assertEqual(c.cluster_id, "mc-a")
        self.assertFalse(hasattr(c, "id"))
    def test_recall_returns_cluster_ids(self):
        a = self.b.recall("memory useful", top_k=10)
        self.assertGreater(len(a), 0)
        for i in a:
            self.assertIn(i, ("mc-a", "mc-b"))
    def test_recall_not_debug_strings(self):
        a = self.b.recall("memory useful", top_k=10)
        for i in a:
            self.assertFalse(i.startswith("MemoryCluster("))

    def test_multi_belief(self):
        s, p = _ms(CM)
        try:
            c = s.query_cluster("mc-c")
            n = [x for x in c.atoms if x.startswith("(Contains")]
            self.assertEqual(len(n), 3)
        finally:
            _cl(p)
class F02_WritebackRole(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.u = WMTMUtility(self.w, self.s, self.e)
    def tearDown(self):
        _cl(self.p)
    def test_writeback_uses_observed_event(self):
        """F02 (current bug): writeback labels derived items as ObservedEvent, not DerivedBelief."""
        it = self.w.admit("d1", "derived conclusion", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        self.u.writeback(["d1"])
        with open(self.p) as f: c = f.read()
        self.assertIn("ObservedEvent", c)
        self.assertIn("observed-event", c)
    def test_writeback_duplicates(self):
        it = self.w.admit("d2", "dup test", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        self.u.writeback(["d2"])
        self.u.writeback(["d2"])
        with open(self.p) as f: c = f.read()
        self.assertGreaterEqual(c.count("dup test"), 2)
    def test_zero_use_high_sti(self):
        it = self.w.admit("d3", "unused", source_type="derived", sti=32.0)
        score = self.u.compute_utility(it)
        self.assertGreaterEqual(score, 5.0)
class F03_Inference(unittest.TestCase):
    def setUp(self):
        self.w = WMTMStore()
        self.eng = WMTMInferenceEngine(self.w)
        self.w.admit("s1", "src", sti=10.0)
    def test_missing_parent_accepted(self):
        """F03 (current bug): derive() accepts missing parent IDs without error."""
        self.assertIsNotNone(self.eng.derive("x", ["s1", "missing"]))
    def test_bogus_rule_accepted(self):
        """F03 (current bug): derive() accepts unknown rule names without error."""
        self.assertIsNotNone(self.eng.derive("x", ["s1"], rule="bogus"))
    def test_conjunction_concat(self):
        self.w.admit("s2", "second", sti=10.0)
        r = self.eng.derive_conjunction("s1", "s2")
        self.assertIn("AND", r.text)
    def test_implication_concat(self):
        self.w.admit("s2", "second", sti=10.0)
        r = self.eng.derive_implication("s1", "s2")
        self.assertIn("->", r.text)
class F04_Provenance(unittest.TestCase):
    def setUp(self):
        self.w = WMTMStore()
        self.eng = WMTMInferenceEngine(self.w)
        self.w.admit("root", "root", sti=20.0)
        self.mid = self.eng.derive("mid", ["root"], sti=15.0)
        self.leaf = self.eng.derive("leaf", [self.mid.id], sti=12.0)
    def test_provenance_resident(self):
        prov = self.eng.get_provenance("deriv-2")
        self.assertEqual(prov["id"], "deriv-2")
    def test_provenance_evicted(self):
        self.w.remove("root")
        prov = self.eng.get_provenance("deriv-2")
        self.assertIsNotNone(prov)
        chain = prov.get("derived_from", [])
        if chain:
            parent = chain[0]
            if isinstance(parent, dict):
                grand = parent.get("derived_from", [])
                if grand and isinstance(grand[0], dict):
                    self.assertIsNone(grand[0].get("id"))
    def test_local_counter(self):
        e2 = WMTMInferenceEngine(WMTMStore())
        e2.wmtm.admit("x", "x", sti=10.0)
        self.assertEqual(e2.derive("y", ["x"]).id, "deriv-1")
class F05_Clocks(unittest.TestCase):
    def test_recency_cycle_0(self):
        w = WMTMStore(capacity=100, forgetting=ForgettingPolicy(sti_threshold=0.0, max_items=999))
        w.admit("t1", "t", sti=1000.0)
        for _ in range(10): w.tick()
        item = w.get("t1")
        self.assertIsNotNone(item)
        r = 0.9 ** max(0, item.age - item.last_used)
        self.assertAlmostEqual(r, 0.9 ** 10, places=2)
    def test_recency_cycle_100(self):
        w = WMTMStore(capacity=100, forgetting=ForgettingPolicy(sti_threshold=0.0, max_items=999))
        for _ in range(100): w.tick()
        w.admit("t2", "t", sti=1000.0)
        for _ in range(10): w.tick()
        item = w.get("t2")
        self.assertIsNotNone(item)
        r = 0.9 ** max(0, item.age - item.last_used)
        self.assertAlmostEqual(r, 1.0, places=3)
    def test_decay_age(self):
        w = WMTMStore()
        w.admit("t", "t", sti=10.0)
        self.assertEqual(w.get("t").age, 0)
        w.tick()
        self.assertEqual(w.get("t").age, 1)
class F06_Capacity(unittest.TestCase):
    def test_capacity_one_large_policy(self):
        w = WMTMStore(capacity=1, forgetting=ForgettingPolicy(max_items=5, sti_threshold=0.0))
        w.admit("a", "a", sti=10.0)
        w.admit("b", "b", sti=10.0)
        self.assertEqual(len(w.all_items()), 2)
    def test_capacity_one_matching(self):
        w = WMTMStore(capacity=1, forgetting=ForgettingPolicy(max_items=1, sti_threshold=0.0))
        w.admit("a", "a", sti=10.0)
        w.admit("b", "b", sti=10.0)
        self.assertEqual(len(w.all_items()), 1)
class F07_Coordinator(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.coord = WMTMCoordinator(self.s, self.e, capacity=100)
    def tearDown(self):
        _cl(self.p)
    def test_tick_advances_cycle(self):
        c0 = self.coord.wmtm._cycle
        self.coord.on_tick()
        self.coord.on_tick()
        self.assertEqual(self.coord.wmtm._cycle - c0, 2)
    def test_atoms_unchanged(self):
        n0 = self.e.bank.num_atoms
        self.coord.on_tick()
        self.coord.on_tick()
        self.assertEqual(self.e.bank.num_atoms, n0)
class F08_Serialization(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.u = WMTMUtility(self.w, self.s, self.e)
    def tearDown(self):
        _cl(self.p)
    def test_writeback_quotes_vs_query_unquoted(self):
        it = self.w.admit("d1", "derived conclusion", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        self.u.writeback(["d1"])
        new = [c for c in self.s.clusters() if c.cluster_id != "mc-a"]
        self.assertGreater(len(new), 0)
        atoms_text = chr(10).join(new[0].atoms)
        self.assertIn("derived conclusion", atoms_text)
        results = self.s.query_about("derived conclusion")
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()

