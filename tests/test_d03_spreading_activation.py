"""D03 regression test: spreading_activation must resolve cluster→belief IDs.

The bug: BFS seeds were cluster IDs (mc-*) but adjacency was keyed by
belief IDs (belief-*).  Spreading activation found 0 neighbors.
"""
from __future__ import annotations
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from petta_memory.store import MediumMemoryStore
from petta_memory.wmtm import WMTMStore
from petta_memory.wmtm_recall import RecallBridge
from petta_memory.ecan_bridge import ECANBridge

# mc-a: belief-a with evidence event-a
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

# mc-b: belief-b with cross-cluster evidence to event-a (from mc-a)
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
    "(EvidenceFor belief-b event-a)\n"  # cross-cluster evidence link
)

# Isolated cluster
CX = (
    "(MemoryCluster mc-x)\n"
    "(SchemaVersion mc-x medium-memory-v1)\n"
    "(ClusterType mc-x belief-record)\n"
    "(ClusterOpenedAt mc-x \"2026-09-08T00:00:00Z\")\n"
    "(ClusterSource mc-x local-test)\n"
    "(Contains mc-x belief-x)\n"
    "(Contains mc-x event-x)\n"
    "(DerivedBelief belief-x)\n"
    "(BeliefContent belief-x (Isolated concept))\n"
    "(About belief-x isolated)\n"
    "(TruthValue belief-x (stv 0.5 0.5))\n"
    "(ObservedEvent event-x)\n"
    "(EvidenceFor belief-x event-x)\n"
)

def _ms(*texts):
    fd, p = tempfile.mkstemp(suffix=".mem", prefix="d03_")
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

class D03SpreadingActivation(unittest.TestCase):
    """D03: spreading_activation must resolve cluster IDs to belief IDs."""

    def setUp(self):
        self.s, self.p = _ms(CA, CB)
        self.e = ECANBridge(self.s)
        self.e.sync_from_store()
        self.w = WMTMStore()
        self.b = RecallBridge(self.s, self.e, self.w)

    def tearDown(self):
        _cl(self.p)

    def test_evidence_map_keyed_by_belief_id(self):
        """Evidence map keys are belief IDs, not cluster IDs."""
        evmap = self.e.get_evidence_map()
        assert "belief-a" in evmap, f"evidence map keys: {list(evmap.keys())}"
        assert "mc-a" not in evmap, "evidence map should NOT have cluster IDs"

    def test_spreading_activation_finds_neighbors(self):
        """BFS from mc-a should reach mc-b via shared event-a.

        adjacency: belief-a ↔ event-a, belief-b ↔ event-b, belief-b ↔ event-a
        BFS (depth=2): belief-a → event-a → belief-b → resolve to mc-b
        """
        cluster_a = self.s.query_cluster("mc-a")
        text_a = getattr(cluster_a, "text", "mc-a") if cluster_a else "mc-a"
        self.w.admit("mc-a", text=text_a, source_type="query",
                     origin_cluster="mc-a", sti=10.0)

        admitted = self.b.spreading_activation(seed_ids=["mc-a"], depth=2)
        assert len(admitted) > 0, \
            "spreading activation found 0 neighbors — D03 bug!"
        assert "mc-b" in admitted, f"mc-b not in admitted: {admitted}"

    def test_spreading_activation_isolated_finds_nothing(self):
        """Isolated cluster should find 0 neighbors."""
        s, p = _ms(CX)
        try:
            e = ECANBridge(s)
            e.sync_from_store()
            w = WMTMStore()
            b = RecallBridge(s, e, w)
            cluster_x = s.query_cluster("mc-x")
            text_x = getattr(cluster_x, "text", "mc-x") if cluster_x else "mc-x"
            w.admit("mc-x", text=text_x, source_type="query",
                    origin_cluster="mc-x", sti=10.0)
            admitted = b.spreading_activation(seed_ids=["mc-x"], depth=2)
            assert len(admitted) == 0, f"expected 0 neighbors, got {admitted}"
        finally:
            _cl(p)

    def test_spreading_activation_boosts_existing(self):
        """If neighbor already in WMTM, boost STI instead of re-admitting."""
        for cid in ("mc-a", "mc-b"):
            cluster = self.s.query_cluster(cid)
            text = getattr(cluster, "text", cid) if cluster else cid
            self.w.admit(cid, text=text, source_type="query",
                         origin_cluster=cid, sti=10.0)

        item_b_before = self.w.get("mc-b")
        sti_before = item_b_before.sti if item_b_before else 0.0

        admitted = self.b.spreading_activation(seed_ids=["mc-a"], depth=2)

        item_b_after = self.w.get("mc-b")
        sti_after = item_b_after.sti if item_b_after else 0.0
        assert sti_after > sti_before, \
            f"STI not boosted: {sti_before} -> {sti_after}"
        # mc-b was already in WMTM so should NOT be in admitted
        assert "mc-b" not in admitted, \
            "mc-b should be boosted, not re-admitted"
