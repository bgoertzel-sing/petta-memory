import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory import MediumMemoryStore, ValidationError


VALID_CLUSTER = """
(MemoryCluster mc1)
(SchemaVersion mc1 medium-memory-v1)
(ClusterType mc1 episode-record)
(ClusterOpenedAt mc1 "2026-06-27 14:10 PDT")
(ClusterSource mc1 src-test)
(Contains mc1 e1)
(ClusterStatus mc1 active)
(ObservedEvent e1)
(EpistemicRole e1 observed-event)
(About e1 MediumPeTTaMemory)
(HasStatus e1 recorded)
"""

QUOTE_CLUSTER = """
(MemoryCluster mc2)
(SchemaVersion mc2 medium-memory-v1)
(ClusterType mc2 episode-record)
(ClusterOpenedAt mc2 "2026-06-27 14:11 PDT")
(ClusterSource mc2 src-test)
(Contains mc2 se1)
(Contains mc2 qc1)
(ClusterStatus mc2 active)
(SpeechEvent se1)
(RawUtterance se1 "raw quote should not be PLN premise")
(QuotedClaim qc1)
(EpistemicRole qc1 quoted-utterance)
(ClaimText qc1 "quoted claim text")
(About qc1 PLN)
(ClaimStatus qc1 quoted-unverified)
(DerivedBelief b1)
(BeliefContent b1 (Requires MediumPeTTaMemory PLNReadyViews))
(TruthValue b1 (stv 0.90 0.70))
(EvidenceFor b1 qc1)
(PromotionEvent pe1)
(PromotesTo pe1 b1)
"""


class MediumMemoryStoreTests(unittest.TestCase):
    def test_append_valid_cluster_and_read_back(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            cluster = store.append_cluster(VALID_CLUSTER)
            self.assertEqual(cluster.cluster_id, "mc1")
            self.assertEqual(len(store.clusters()), 1)
            self.assertIn("(ObservedEvent e1)", store.tail())

    def test_reject_malformed_atom(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            with self.assertRaises(ValidationError):
                store.append_cluster("""
(MemoryCluster mc1)
(SchemaVersion mc1 medium-memory-v1)
(ClusterType mc1 episode-record)
(ClusterOpenedAt mc1 "now")
(ClusterSource mc1 src-test)
(Contains mc1 e1)
(ObservedEvent e1
""")

    def test_reject_oversized_atom(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta", max_atom_chars=40)
            with self.assertRaises(ValidationError):
                store.append_cluster(VALID_CLUSTER)

    def test_reject_missing_cluster_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            with self.assertRaisesRegex(ValidationError, "ClusterType"):
                store.append_cluster("""
(MemoryCluster mc1)
(SchemaVersion mc1 medium-memory-v1)
(ClusterOpenedAt mc1 "now")
(ClusterSource mc1 src-test)
(Contains mc1 e1)
""")

    def test_query_by_id_type_about_status_role(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(VALID_CLUSTER)
            self.assertEqual([c.cluster_id for c in store.query_id("e1")], ["mc1"])
            self.assertEqual([c.cluster_id for c in store.query_type("ObservedEvent")], ["mc1"])
            self.assertEqual([c.cluster_id for c in store.query_about("MediumPeTTaMemory")], ["mc1"])
            self.assertEqual([c.cluster_id for c in store.query_status("active")], ["mc1"])
            self.assertEqual([c.cluster_id for c in store.query_role("observed-event")], ["mc1"])

    def test_prompt_view_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(VALID_CLUSTER)
            view = store.prompt_view(limit_chars=30)
            self.assertLessEqual(len(view), 30)
            self.assertTrue(view.startswith("(About") or view.startswith("(ClusterStatus") or view.startswith("(HasStatus"))

    def test_pln_view_filters_raw_quotes_and_unpromoted_claims(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(QUOTE_CLUSTER)
            view = store.pln_view()
            self.assertNotIn("RawUtterance", view)
            self.assertNotIn("ClaimText", view)
            self.assertNotIn("QuotedClaim", view)
            self.assertIn("DerivedBelief b1", view)
            self.assertIn("TruthValue b1", view)


if __name__ == "__main__":
    unittest.main()
