"""
C02: Object identity tests.

Tests the ObjectIndex that maps cognitive object IDs (beliefs, events, etc.)
to their containing clusters.  Verifies that:
- Multi-belief clusters expose individual objects
- Object lookup resolves by type
- Unknown IDs return None (not debug strings)
- ECAN attention can use object IDs
"""
from __future__ import annotations
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from petta_memory.store import MediumMemoryStore
from petta_memory.object_index import ObjectIndex
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
    fd, p = tempfile.mkstemp(suffix=".mem", prefix="oi_")
    os.close(fd)
    s = MediumMemoryStore(p)
    for t in ts: s.append_cluster(t)
    return s, p

def _cl(p):
    try: os.unlink(p)
    except OSError: pass


class TestObjectIndex(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA, CM)
        self.idx = ObjectIndex()
        self.idx.build_from_store(self.s)
    def tearDown(self):
        _cl(self.p)

    def test_single_belief_cluster(self):
        r = self.idx.lookup("belief-a")
        self.assertEqual(r, ("mc-a", "DerivedBelief"))

    def test_event_lookup(self):
        r = self.idx.lookup("event-a")
        self.assertEqual(r, ("mc-a", "ObservedEvent"))

    def test_multi_belief_cluster_two_beliefs(self):
        r1 = self.idx.lookup("belief-c1")
        r2 = self.idx.lookup("belief-c2")
        self.assertEqual(r1, ("mc-c", "DerivedBelief"))
        self.assertEqual(r2, ("mc-c", "DerivedBelief"))
        self.assertNotEqual(r1[0], "mc-a")

    def test_objects_in_cluster(self):
        objs = self.idx.objects_in_cluster("mc-c")
        ids = [o[0] for o in objs]
        self.assertIn("belief-c1", ids)
        self.assertIn("belief-c2", ids)
        self.assertIn("event-c", ids)
        self.assertEqual(len(objs), 3)

    def test_cluster_of(self):
        self.assertEqual(self.idx.cluster_of("belief-a"), "mc-a")
        self.assertEqual(self.idx.cluster_of("belief-c1"), "mc-c")

    def test_object_type(self):
        self.assertEqual(self.idx.object_type("belief-a"), "DerivedBelief")
        self.assertEqual(self.idx.object_type("event-a"), "ObservedEvent")

    def test_unknown_returns_none(self):
        self.assertIsNone(self.idx.lookup("nonexistent"))
        self.assertIsNone(self.idx.cluster_of("nonexistent"))
        self.assertIsNone(self.idx.object_type("nonexistent"))

    def test_len(self):
        self.assertEqual(len(self.idx), 5)

    def test_all_objects(self):
        objs = self.idx.all_objects()
        self.assertIn("belief-a", objs)
        self.assertIn("event-a", objs)
        self.assertIn("belief-c1", objs)
        self.assertIn("belief-c2", objs)
        self.assertIn("event-c", objs)

    def test_no_debug_string_identity(self):
        """Object IDs are never debug strings like 'MemoryCluster(...'."""
        for obj_id in self.idx.all_objects():
            self.assertFalse(obj_id.startswith("MemoryCluster("))


class TestECANObjectIDs(unittest.TestCase):
    def setUp(self):
        self.s, self.p = _ms(CA, CM)
        self.ecan = ECANBridge(self.s)
        self.ecan.sync_from_store()
        self.idx = ObjectIndex()
        self.idx.build_from_store(self.s)
    def tearDown(self):
        _cl(self.p)

    def test_ecan_belief_ids_are_object_ids(self):
        """ECAN should track belief object IDs, not cluster IDs."""
        beliefs = self.ecan._belief_ids
        self.assertIn("belief-a", beliefs)
        self.assertIn("belief-c1", beliefs)
        self.assertIn("belief-c2", beliefs)

    def test_ecan_stimulate_uses_object_id(self):
        self.ecan.stimulate_beliefs({"belief-a": 5.0})
        sti = self.ecan.bank.get_sti("belief-a")
        self.assertGreater(sti, 0)

    def test_ecan_prioritized_beliefs_use_object_ids(self):
        self.ecan.stimulate_beliefs({"belief-c1": 10.0})
        prioritized = dict(self.ecan.get_prioritized_beliefs(limit=10))
        self.assertIn("belief-c1", prioritized)


if __name__ == "__main__":
    unittest.main()
