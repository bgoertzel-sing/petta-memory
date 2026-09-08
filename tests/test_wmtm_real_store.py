"""C01: Real-object regression fixtures for WMTM defects F01-F08."""
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

CLUSTER_A = (
    "(MemoryCluster mc-a)\n"
"(SchemaVersion mc-a medium-memory-v1)\n"
"(ClusterType mc-a belief-record)\n"
'(ClusterOpenedAt mc-a "2026-09-08T00:00:00Z")\n'
"(ClusterSource mc-a local-test)\n"
"(Contains mc-a belief-a)\n"
"(Contains mc-a event-a)\n"
"(DerivedBelief belief-a)\n"
"(BeliefContent belief-a (Useful memory))\n"
"(About belief-a memory)\n"
"(TruthValue belief-a (stv 0.9 0.8))\n"
"(ObservedEvent event-a)\n"
"(EvidenceFor belief-a event-a)\n"

CLUSTER_B = (
    "(MemoryCluster mc-b)\n"
"(SchemaVersion mc-b medium-memory-v1)\n"
"(ClusterType mc-b belief-record)\n"
'(ClusterOpenedAt mc-b "2026-09-08T00:00:00Z")\n'
"(ClusterSource mc-b local-test)\n"
"(Contains mc-b belief-b)\n"
"(Contains mc-b event-b)\n"
"(DerivedBelief belief-b)\n"
"(BeliefContent belief-b (Helpful system))\n"
"(About belief-b memory)\n"
"(TruthValue belief-b (stv 0.8 0.7))\n"
"(ObservedEvent event-b)\n"
"(EvidenceFor belief-b event-b)\n"

CLUSTER_MULTI = (
    "(MemoryCluster mc-c)\n"
"(SchemaVersion mc-c medium-memory-v1)\n"
"(ClusterType mc-c belief-record)\n"
'(ClusterOpenedAt mc-c "2026-09-08T00:00:00Z")\n'
"(ClusterSource mc-c local-test)\n"
"(Contains mc-c belief-c1)\n"
"(Contains mc-c belief-c2)\n"
"(Contains mc-c event-c)\n"
"(DerivedBelief belief-c1)\n"
"(BeliefContent belief-c1 (Fast cache))\n"
"(About belief-c1 cache)\n"
"(TruthValue belief-c1 (stv 0.7 0.6))\n"
"(DerivedBelief belief-c2)\n"
"(BeliefContent belief-c2 (Stable service))\n"
"(About belief-c2 service)\n"
"(TruthValue belief-c2 (stv 0.85 0.9))\n"
"(ObservedEvent event-c)\n"
"(EvidenceFor belief-c1 event-c)\n"
"(EvidenceFor belief-c2 event-c)\n"

def _make_store(*texts):
    fd, path = tempfile.mkstemp(suffix=".mem", prefix="wmtm_real_")
    os.close(fd)
    s = MediumMemoryStore(path)
    for t in texts:
        s.append_cluster(t)
    return s, path

def _cleanup(path):
    try: os.unlink(path)
    except OSError: pass

class F01_RecallIdentity(unittest.TestCase):
    def setUp(self):
        self.store, self.path = _make_store(CLUSTER_A, CLUSTER_B)
        self.ecan = ECANBridge(self.store)
        self.ecan.sync_from_store()
        self.wmtm = WMTMStore()
        self.bridge = RecallBridge(self.store, self.ecan, self.wmtm)
    def tearDown(self):
        _cleanup(self.path)
    def test_cluster_has_cluster_id_not_id(self):
        c = self.store.query_cluster("mc-a")
        self.assertTrue(hasattr(c, "cluster_id"))
        self.assertEqual(c.cluster_id, "mc-a")
        self.assertFalse(hasattr(c, "id"))
    def test_recall_uses_debug_string_id(self):
        admitted = self.bridge.recall("memory useful", top_k=5)
        self.assertGreater(len(admitted), 0)
        for item_id in admitted:
            self.assertTrue(item_id.startswith("MemoryCluster("))
    def test_recall_does_not_use_cluster_id(self):
        admitted = self.bridge.recall("memory useful", top_k=5)
        for item_id in admitted:
            self.assertNotIn(item_id, ("mc-a", "mc-b"))
    def test_multi_belief_cluster(self):
        store, path = _make_store(CLUSTER_MULTI)
        try:
            cluster = store.query_cluster("mc-c")
            contains = [a for a in cluster.atoms if a.startswith("(Contains")]
            self.assertEqual(len(contains), 3)
        finally:
            _cleanup(path)


class F06_Capacity(unittest.TestCase):
    def test_capacity_one_with_large_policy_holds_two(self):
        wmtm = WMTMStore(capacity=1, forgetting=ForgettingPolicy(max_items=5, sti_threshold=0.0))
        wmtm.admit("a", "a", sti=10.0)
        wmtm.admit("b", "b", sti=10.0)
        self.assertEqual(len(wmtm.all_items()), 2)
    def test_capacity_one_with_matching_policy_holds_one(self):
        wmtm = WMTMStore(capacity=1, forgetting=ForgettingPolicy(max_items=1, sti_threshold=0.0))
        wmtm.admit("a", "a", sti=10.0)
        wmtm.admit("b", "b", sti=10.0)
        self.assertEqual(len(wmtm.all_items()), 1)

class F07_Coordinator(unittest.TestCase):
    def setUp(self):
        self.store, self.path = _make_store(CLUSTER_A)
        self.ecan = ECANBridge(self.store)
        self.ecan.sync_from_store()
        self.coord = WMTMCoordinator(self.store, self.ecan, capacity=100)
    def tearDown(self):
        _cleanup(self.path)
    def test_on_tick_ticks_wmtm(self):
        cycle_before = self.coord.wmtm._cycle
        self.coord.on_tick()
        self.coord.on_tick()
        self.assertEqual(self.coord.wmtm._cycle - cycle_before, 2)
    def test_ecan_not_run_by_coordinator(self):
        atoms_before = self.ecan.bank.num_atoms
        self.coord.on_tick()
        self.coord.on_tick()
        atoms_after = self.ecan.bank.num_atoms
        self.assertEqual(atoms_before, atoms_after)

class F08_Serialization(unittest.TestCase):
    def setUp(self):
        self.store, self.path = _make_store(CLUSTER_A)
        self.ecan = ECANBridge(self.store)
        self.ecan.sync_from_store()
        self.wmtm = WMTMStore()
        self.utility = WMTMUtility(self.wmtm, self.store, self.ecan)
    def tearDown(self):
        _cleanup(self.path)
    def test_writeback_uses_quoted_text(self):
        self.wmtm.admit("d1", "memory is useful", source_type="derived", derived_from=["mc-a"], sti=32.0)
        self.utility.writeback(["d1"])
        new = [c for c in self.store.clusters() if c.cluster_id != "mc-a"]
        self.assertGreater(len(new), 0)
        atoms_text = chr(10).join(new[0].atoms)
        self.assertIn("memory is useful", atoms_text)
    def test_query_about_finds_symbol(self):
        results = self.store.query_about("memory")
        self.assertGreater(len(results), 0)
    def test_query_about_misses_quoted_sentence(self):
        self.wmtm.admit("d2", "complex sentence about memory", source_type="derived", derived_from=["mc-a"], sti=32.0)
        self.utility.writeback(["d2"])
        store2 = MediumMemoryStore(self.path)
        results = store2.query_about("complex sentence about memory")
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()
