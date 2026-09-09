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

CA = (
    "(MemoryCluster mc-a)\n"
    "(SchemaVersion mc-a medium-memory-v1)\n"
    "(ClusterType mc-a belief-record)\n"
    "(ClusterOpenedAt mc-a \"2026-09-08T00:00:00Z\")\n"
    "(ClusterSource mc-a local-test)\n"
    "(Contains mc-a belief-a)\n"
    "(Contains mc-a event-a)\n"
    "(DerivedBelief belief-a)\n"
    "(BeliefContent belief-a (Useful memory))\n"
    "(About belief-a memory)\n"
    "(TruthValue belief-a (stv 0.9 0.8))\n"
    "(ObservedEvent event-a)\n"
    "(EvidenceFor belief-a event-a)\n"
)

CB = (
    "(MemoryCluster mc-b)\n"
    "(SchemaVersion mc-b medium-memory-v1)\n"
    "(ClusterType mc-b belief-record)\n"
    "(ClusterOpenedAt mc-b \"2026-09-08T00:00:00Z\")\n"
    "(ClusterSource mc-b local-test)\n"
    "(Contains mc-b belief-b)\n"
    "(Contains mc-b event-b)\n"
    "(DerivedBelief belief-b)\n"
    "(BeliefContent belief-b (Helpful system))\n"
    "(About belief-b memory)\n"
    "(TruthValue belief-b (stv 0.8 0.7))\n"
    "(ObservedEvent event-b)\n"
    "(EvidenceFor belief-b event-b)\n"
)

CM = (
    "(MemoryCluster mc-c)\n"
    "(SchemaVersion mc-c medium-memory-v1)\n"
    "(ClusterType mc-c belief-record)\n"
    "(ClusterOpenedAt mc-c \"2026-09-08T00:00:00Z\")\n"
    "(ClusterSource mc-c local-test)\n"
    "(Contains mc-c belief-c1)\n"
    "(Contains mc-c belief-c2)\n"
    "(Contains mc-c event-c)\n"
    "(DerivedBelief belief-c1)\n"
    "(BeliefContent belief-c1 (Fast inference))\n"
    "(About belief-c1 inference)\n"
    "(TruthValue belief-c1 (stv 0.8 0.7))\n"
    "(DerivedBelief belief-c2)\n"
    "(BeliefContent belief-c2 (Slow inference))\n"
    "(About belief-c2 inference)\n"
    "(TruthValue belief-c2 (stv 0.5 0.4))\n"
    "(ObservedEvent event-c)\n"
    "(EvidenceFor belief-c1 event-c)\n"
    "(EvidenceFor belief-c2 event-c)\n"
)

def _ms(*texts):
    fd, p = tempfile.mkstemp(suffix=".mem", prefix="wmtm_r_")
    os.close(fd)
    s = MediumMemoryStore(p)
    for t in texts:
        s.append_cluster(t)
    return s, p

def _cl(p):
    try:
        os.unlink(p)
    except OSError:
        pass

class F01(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA, CB)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.b = RecallBridge(self.s, self.e, self.w)
    def tearDown(self):
        _cl(self.p)
    def test_cluster_id_not_id(self):
        c = self.s.query_cluster("mc-a")
        self.assertTrue(hasattr(c, "cluster_id"))
        self.assertEqual(c.cluster_id, "mc-a")
        self.assertFalse(hasattr(c, "id"))
    def test_recall_returns_cluster_id(self):
        a = self.b.recall("memory useful", top_k=10)
        self.assertGreater(len(a), 0)
        for i in a:
            self.assertIn(i, ("mc-a", "mc-b"))
    def test_multi_belief(self):
        s, p = _ms(CM)
        try:
            c = s.query_cluster("mc-c")
            n = [x for x in c.atoms if x.startswith("(Contains")]
            self.assertEqual(len(n), 3)
        finally:
            _cl(p)

class F02(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.u = WMTMUtility(self.w, self.s, self.e, writeback_threshold=0.0)
    def tearDown(self):
        _cl(self.p)
    def test_writeback_singleword_works(self):
        """F02: writeback works for single-word text."""
        it = self.w.admit("d1", "conclusion", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        r = self.u.writeback(["d1"])
        self.assertIn("d1", r)
        new = [c for c in self.s.clusters() if c.cluster_id != "mc-a"]
        self.assertGreater(len(new), 0)
    def test_writeback_multiword_succeeds(self):
        """F02/F08 FIXED: writeback succeeds for multi-word text (About atom quoted)."""
        it = self.w.admit("d2", "derived conclusion", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        r = self.u.writeback(["d2"])
        self.assertIn("d2", r)
        new = [c for c in self.s.clusters() if c.cluster_id != "mc-a"]
        self.assertGreater(len(new), 0)
    def test_writeback_uses_observed_event(self):
        """F02: writeback labels items as ObservedEvent, not DerivedBelief."""
        it = self.w.admit("d3", "result", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        self.u.writeback(["d3"])
        with open(self.p) as f:
            content = f.read()
        self.assertIn("ObservedEvent", content)
    def test_writeback_has_derived_belief(self):
        """F02 FIXED: writeback cluster now contains DerivedBelief predicate."""
        it = self.w.admit("d4", "finding", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        self.u.writeback(["d4"])
        new = [c for c in self.s.clusters() if c.cluster_id != "mc-a"]
        self.assertGreater(len(new), 0)
        atoms_text = "\n".join(new[0].atoms)
        self.assertIn("DerivedBelief", atoms_text)
    def test_zero_use_high_sti(self):
        """F02: compute_utility with zero use_count returns sti*0.3."""
        it = self.w.admit("d5", "unused", source_type="derived", sti=32.0)
        score = self.u.compute_utility(it)
        self.assertAlmostEqual(score, 32.0 * 0.3, places=1)

class F03(unittest.TestCase):
    def setUp(self):
        self.w = WMTMStore()
        self.eng = WMTMInferenceEngine(self.w)
        self.w.admit("s1", "cats", sti=10.0)
    def test_missing_parent_rejected(self):
        """F03 (fixed): derive() rejects missing parent IDs."""
        self.assertIsNone(self.eng.derive("x", ["s1", "missing"]))
    def test_bogus_rule_rejected(self):
        """F03 (fixed): derive() rejects unknown rules."""
        self.assertIsNone(self.eng.derive("x", ["s1"], rule="bogus"))
    def test_conjunction_concat(self):
        """F03: conjunction concatenates texts with AND."""
        self.w.admit("s2", "dogs", sti=10.0)
        r = self.eng.derive_conjunction("s1", "s2")
        self.assertEqual(r.text, "cats AND dogs")
    def test_implication_concat(self):
        """F03: implication concatenates texts with ->."""
        self.w.admit("s2", "wet ground", sti=10.0)
        r = self.eng.derive_implication("s1", "s2")
        self.assertEqual(r.text, "cats -> wet ground")

class F04(unittest.TestCase):
    def setUp(self):
        self.w = WMTMStore()
        self.eng = WMTMInferenceEngine(self.w)
        self.w.admit("root", "root source", sti=100.0)
        self.mid = self.eng.derive("middle", ["root"], sti=50.0)
        self.leaf = self.eng.derive("leaf result", [self.mid.id], sti=50.0)
    def test_provenance_resident(self):
        prov = self.eng.get_provenance(self.leaf.id)
        self.assertIsNotNone(prov)
        self.assertEqual(prov["id"], self.leaf.id)
    def test_provenance_root_evicted(self):
        self.w.remove("root")
        prov = self.eng.get_provenance(self.leaf.id)
        self.assertIsNotNone(prov)
    def test_local_counter(self):
        e2 = WMTMInferenceEngine(WMTMStore())
        e2.wmtm.admit("x", "x", sti=10.0)
        self.assertEqual(e2.derive("y", ["x"]).id, "deriv-1")

class F05(unittest.TestCase):
    def test_recency_cycle_0(self):
        w = WMTMStore(capacity=100, forgetting=ForgettingPolicy(sti_threshold=0.0, max_items=999))
        w.admit("t1", "t", sti=1000.0)
        for _ in range(10):
            w.tick()
        item = w.get("t1")
        self.assertIsNotNone(item)
        r = 0.9 ** max(0, item.age - item.last_used)
        self.assertAlmostEqual(r, 0.9 ** 10, places=2)
    def test_recency_cycle_100_bug(self):
        """F05: last_used stores global cycle, age starts at 0 -> recency=1.0."""
        w = WMTMStore(capacity=100, forgetting=ForgettingPolicy(sti_threshold=0.0, max_items=999))
        for _ in range(100):
            w.tick()
        w.admit("t2", "t", sti=1000.0)
        for _ in range(10):
            w.tick()
        item = w.get("t2")
        self.assertIsNotNone(item)
        r = 0.9 ** max(0, item.age - item.last_used)
        self.assertAlmostEqual(r, 0.9 ** 10, places=2)
    def test_decay_age(self):
        w = WMTMStore()
        w.admit("t", "t", sti=10.0)
        self.assertEqual(w.get("t").age, 0)
        w.tick()
        self.assertEqual(w.get("t").age, 1)

class F06(unittest.TestCase):
    def test_capacity_one_large_policy(self):
        """F06 (fixed): capacity=1 evicts even when max_items=5."""
        w = WMTMStore(capacity=1, forgetting=ForgettingPolicy(max_items=5, sti_threshold=0.0))
        w.admit("a", "a", sti=10.0)
        w.admit("b", "b", sti=10.0)
        self.assertEqual(len(w.all_items()), 1)
    def test_capacity_one_matching(self):
        w = WMTMStore(capacity=1, forgetting=ForgettingPolicy(max_items=1, sti_threshold=0.0))
        w.admit("a", "a", sti=10.0)
        w.admit("b", "b", sti=10.0)
        self.assertEqual(len(w.all_items()), 1)

class F07(unittest.TestCase):
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

class F08(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.u = WMTMUtility(self.w, self.s, self.e, writeback_threshold=0.0)
    def tearDown(self):
        _cl(self.p)
    def test_writeback_multiword_succeeds(self):
        """F08 FIXED: Multi-word text now works (About atom is quoted)."""
        it = self.w.admit("d1", "derived conclusion", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        r = self.u.writeback(["d1"])
        self.assertIn("d1", r)
        new = [c for c in self.s.clusters() if c.cluster_id != "mc-a"]
        self.assertGreater(len(new), 0)
    def test_writeback_singleword_succeeds(self):
        """F08 FIXED: Single-word text works and query_about matches."""
        it = self.w.admit("d2", "conclusion", source_type="derived", derived_from=["mc-a"], sti=32.0)
        it.use_count = 1
        r = self.u.writeback(["d2"])
        self.assertIn("d2", r)
        new = [c for c in self.s.clusters() if c.cluster_id != "mc-a"]
        self.assertGreater(len(new), 0)
        atoms_text = "\n".join(new[0].atoms)
        self.assertIn("conclusion", atoms_text)
        results = self.s.query_about("conclusion")
        self.assertGreater(len(results), 0)

if __name__ == "__main__":
    unittest.main()

