"""Review-v4 regression tests: derivation identity, writeback idempotency, provenance recovery, ECAN cycle."""
from __future__ import annotations
import os, sys, tempfile, unittest, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from petta_memory.store import MediumMemoryStore
from petta_memory.wmtm import WMTMStore, WMTMItem, ForgettingPolicy
from petta_memory.wmtm_recall import RecallBridge
from petta_memory.wmtm_inference import WMTMInferenceEngine
from petta_memory.wmtm_utility import WMTMUtility, _stable_derivation_id
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

def _ms(*texts):
    fd, p = tempfile.mkstemp(suffix=".mem", prefix="wmtm_rv4_")
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


class F02Dedup(unittest.TestCase):
    """Writeback idempotency: same content written twice creates one cluster."""
    def setUp(self):
        self.s, self.p = _ms(CA)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.u = WMTMUtility(self.w, self.s, self.e, writeback_threshold=0.0)
    def tearDown(self):
        _cl(self.p)
    def test_writeback_same_content_one_cluster(self):
        """Two WMTM items with same text/derived_from produce one cluster."""
        it_a = self.w.admit('item-a', 'shared conclusion', source_type='derived', derived_from=['mc-a'], sti=32.0)
        it_a.use_count = 1
        it_b = self.w.admit('item-b', 'shared conclusion', source_type='derived', derived_from=['mc-a'], sti=32.0)
        it_b.use_count = 1
        r = self.u.writeback(['item-a', 'item-b'])
        self.assertIn('item-a', r)
        self.assertIn('item-b', r)
        new = [c for c in self.s.clusters() if c.cluster_id != 'mc-a']
        self.assertEqual(len(new), 1, f'Expected 1 new cluster, got {len(new)}')
    def test_writeback_idempotent_across_calls(self):
        """Calling writeback twice on same item does not create duplicate clusters."""
        it = self.w.admit('d1', 'conclusion', source_type='derived', derived_from=['mc-a'], sti=32.0)
        it.use_count = 1
        r1 = self.u.writeback(['d1'])
        self.assertIn('d1', r1)
        r2 = self.u.writeback(['d1'])
        self.assertNotIn('d1', r2)
        new = [c for c in self.s.clusters() if c.cluster_id != 'mc-a']
        self.assertEqual(len(new), 1)


class F06StableIdentity(unittest.TestCase):
    """Derivation identity is content-deterministic and collision-free."""
    def setUp(self):
        self.s, self.p = _ms(CA, CB)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.b = RecallBridge(self.s, self.e, self.w)
        self.b.recall('memory', top_k=5)
    def tearDown(self):
        _cl(self.p)
    def test_same_conclusion_same_id(self):
        """Deriving same text from same sources yields same item ID."""
        eng1 = WMTMInferenceEngine(self.w)
        eng2 = WMTMInferenceEngine(self.w)
        r1 = eng1.derive('conclusion A', ['mc-a'], sti=50.0)
        self.w.remove(r1.id)
        r2 = eng2.derive('conclusion A', ['mc-a'], sti=50.0)
        self.assertEqual(r1.id, r2.id,
            f'Same conclusion should get same ID: {r1.id} != {r2.id}')
    def test_different_conclusion_different_id(self):
        """Different conclusions get different IDs."""
        eng = WMTMInferenceEngine(self.w)
        r1 = eng.derive('conclusion A', ['mc-a'], sti=50.0)
        r2 = eng.derive('conclusion B', ['mc-a'], sti=50.0)
        self.assertNotEqual(r1.id, r2.id)
    def test_cross_engine_no_collision(self):
        """Two engines on same store deriving same text get same item."""
        eng1 = WMTMInferenceEngine(self.w)
        eng2 = WMTMInferenceEngine(self.w)
        r1 = eng1.derive('shared conclusion', ['mc-a'], sti=50.0)
        r2 = eng2.derive('shared conclusion', ['mc-a'], sti=50.0)
        self.assertEqual(r1.id, r2.id)
        self.assertIs(r1, r2)


class F04ProvenanceRecovery(unittest.TestCase):
    """Provenance chain preserves evidence ancestry after writeback."""
    def setUp(self):
        self.s, self.p = _ms(CA, CB)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.b = RecallBridge(self.s, self.e, self.w)
        self.b.recall('memory useful', top_k=5)
        self.u = WMTMUtility(self.w, self.s, self.e, writeback_threshold=0.0)
    def tearDown(self):
        _cl(self.p)
    def test_provenance_recoverable_after_writeback(self):
        """After writeback, parent evidence remains traceable in LTM."""
        eng = WMTMInferenceEngine(self.w)
        sources = [i.id for i in self.w.all_items() if i.source_type == 'recalled'][:2]
        self.assertGreater(len(sources), 0, 'No recalled items to derive from')
        deriv = eng.derive('test conclusion', sources, sti=50.0)
        self.assertIsNotNone(deriv)
        deriv.use_count = 1
        r = self.u.writeback([deriv.id])
        self.assertIn(deriv.id, r)
        cluster_id = _stable_derivation_id('test conclusion', sources)
        cluster = self.s.query_cluster(cluster_id)
        self.assertIsNotNone(cluster, 'Writeback cluster not found in LTM')
        atoms_text = '\n'.join(cluster.atoms)
        for parent_id in sources:
            self.assertIn(f'(EvidenceFor {cluster_id}-bl {parent_id})', atoms_text,
                f'EvidenceFor link to parent {parent_id} not found in cluster')
        self.assertIn('ReasoningEvent', atoms_text)
        self.assertNotIn('ObservedEvent', atoms_text)


class F07ECANCycle(unittest.TestCase):
    """ECAN cycle actually executes during coordinator tick."""
    def setUp(self):
        self.s, self.p = _ms(CA)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.coord = WMTMCoordinator(self.s, self.e, capacity=100)
    def tearDown(self):
        _cl(self.p)
    def test_ecan_sti_changes(self):
        """ECAN run_cycle changes STI values, proving it executes."""
        sti_before = self.e.bank.get_sti('belief-a')
        self.coord.on_tick()
        sti_after = self.e.bank.get_sti('belief-a')
        self.assertNotEqual(sti_before, sti_after,
            'ECAN run_cycle did not change STI - cycle may not be executing')
    def test_wmtm_and_ecan_both_advance(self):
        """on_tick advances WMTM cycle."""
        c0 = self.coord.wmtm._cycle
        self.coord.on_tick()
        self.assertEqual(self.coord.wmtm._cycle - c0, 1)


if __name__ == "__main__":
    unittest.main()
