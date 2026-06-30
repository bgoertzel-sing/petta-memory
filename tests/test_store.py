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
"""

PROMOTED_BELIEF_CLUSTER = """
(MemoryCluster mc3)
(SchemaVersion mc3 medium-memory-v1)
(ClusterType mc3 belief-promotion)
(ClusterOpenedAt mc3 "2026-06-27 14:12 PDT")
(ClusterSource mc3 src-test)
(Contains mc3 pe1)
(Contains mc3 b2)
(ClusterStatus mc3 active)
(PromotionEvent pe1)
(PromotesFrom pe1 qc1)
(PromotesTo pe1 b2)
(PromotionRule pe1 explicit-test-promotion)
(DerivedBelief b2)
(BeliefContent b2 (Requires MediumPeTTaMemory PLNReadyViews))
(TruthValue b2 (stv 0.90 0.70))
(EvidenceFor b2 qc1)
"""

STATUS_OPEN = """
(MemoryCluster mc-status-open)
(SchemaVersion mc-status-open medium-memory-v1)
(ClusterType mc-status-open status-update)
(ClusterOpenedAt mc-status-open "2026-06-27 14:12 PDT")
(ClusterSource mc-status-open src-test)
(Contains mc-status-open st1)
(ClusterStatus mc-status-open active)
(StatusEvent st1)
(StatusSubject st1 c1)
(StatusValue st1 active)
"""

STATUS_RESOLVED = """
(MemoryCluster mc-status-resolved)
(SchemaVersion mc-status-resolved medium-memory-v1)
(ClusterType mc-status-resolved status-update)
(ClusterOpenedAt mc-status-resolved "2026-06-27 14:13 PDT")
(ClusterSource mc-status-resolved src-test)
(Contains mc-status-resolved st2)
(ClusterStatus mc-status-resolved active)
(StatusEvent st2)
(StatusSubject st2 c1)
(StatusValue st2 resolved)
(Supersedes st2 st1)
"""

OLD_LOW_SALIENCE_CLUSTER = """
(MemoryCluster mc-old-low)
(SchemaVersion mc-old-low medium-memory-v1)
(ClusterType mc-old-low prompt-test)
(ClusterOpenedAt mc-old-low "2026-06-27 14:00 PDT")
(ClusterSource mc-old-low src-test)
(Contains mc-old-low old-commitment)
(ClusterStatus mc-old-low active)
(Commitment old-commitment)
(CommitmentText old-commitment "old but relevant")
(About old-commitment MediumPeTTaMemory)
(SalienceEvent sal-old)
(SalienceSubject sal-old old-commitment)
(SalienceValue sal-old low)
"""

NEW_OTHER_TOPIC_CLUSTER = """
(MemoryCluster mc-new-other-topic)
(SchemaVersion mc-new-other-topic medium-memory-v1)
(ClusterType mc-new-other-topic prompt-test)
(ClusterOpenedAt mc-new-other-topic "2026-06-27 14:01 PDT")
(ClusterSource mc-new-other-topic src-test)
(Contains mc-new-other-topic new-question)
(ClusterStatus mc-new-other-topic active)
(OpenQuestion new-question)
(QuestionText new-question "new but unrelated")
(About new-question OtherTopic)
"""

HIGH_SALIENCE_DONE_CLUSTER = """
(MemoryCluster mc-high-done)
(SchemaVersion mc-high-done medium-memory-v1)
(ClusterType mc-high-done prompt-test)
(ClusterOpenedAt mc-high-done "2026-06-27 14:02 PDT")
(ClusterSource mc-high-done src-test)
(Contains mc-high-done done-artifact)
(ClusterStatus mc-high-done resolved)
(Artifact done-artifact)
(ArtifactPath done-artifact artifacts/done.txt)
(About done-artifact MediumPeTTaMemory)
(SalienceEvent sal-done)
(SalienceSubject sal-done done-artifact)
(SalienceValue sal-done high)
"""


class MediumMemoryStoreTests(unittest.TestCase):
    def test_append_valid_cluster_and_read_back(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            cluster = store.append_cluster(VALID_CLUSTER)
            self.assertEqual(cluster.cluster_id, "mc1")
            self.assertEqual(len(store.clusters()), 1)
            self.assertIn(";;; BEGIN MemoryCluster mc1", store.tail())
            self.assertIn("(ObservedEvent e1)", store.tail())

    def test_reject_malformed_atom_does_not_alter_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "medium_memory.metta"
            store = MediumMemoryStore(path)
            store.append_cluster(VALID_CLUSTER)
            before = path.read_text()
            with self.assertRaises(ValidationError):
                store.append_cluster("""
(MemoryCluster mc-bad)
(SchemaVersion mc-bad medium-memory-v1)
(ClusterType mc-bad episode-record)
(ClusterOpenedAt mc-bad "now")
(ClusterSource mc-bad src-test)
(Contains mc-bad e1)
(ObservedEvent e1
""")
            self.assertEqual(path.read_text(), before)

    def test_parse_multiline_nested_atom_and_ignore_comment_in_string(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            cluster = store.append_cluster('''
(MemoryCluster mc-multiline)
(SchemaVersion mc-multiline medium-memory-v1)
(ClusterType mc-multiline parser-test)
(ClusterOpenedAt mc-multiline "2026-06-29 18:00 PDT")
(ClusterSource mc-multiline src-test)
(Contains mc-multiline b-multiline)
(ClusterStatus mc-multiline active)
; comment between atoms should be ignored
(DerivedBelief b-multiline)
(BeliefContent b-multiline
  (And
    (Requires MediumPeTTaMemory PLNReadyViews)
    (Note "parentheses in string (ok) and semicolon ; ok")))
(TruthValue b-multiline (stv 0.80 0.60))
(EvidenceFor b-multiline src-test)
(PromotionEvent pe-multiline)
(PromotesTo pe-multiline b-multiline)
''')
            self.assertEqual(cluster.cluster_id, "mc-multiline")
            self.assertIn(
                '(BeliefContent b-multiline (And (Requires MediumPeTTaMemory PLNReadyViews) (Note "parentheses in string (ok) and semicolon ; ok")))',
                cluster.atoms,
            )

    def test_reject_extra_tokens_after_complete_atom(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            with self.assertRaises(ValidationError):
                store.append_cluster('''
(MemoryCluster mc-bad-extra)
(SchemaVersion mc-bad-extra medium-memory-v1)
(ClusterType mc-bad-extra parser-test)
(ClusterOpenedAt mc-bad-extra "now")
(ClusterSource mc-bad-extra src-test)
(Contains mc-bad-extra e1) stray-token
(ObservedEvent e1)
''')

    def test_reject_top_level_string_or_symbol(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            with self.assertRaisesRegex(ValidationError, "top-level form"):
                store.append_cluster('''
(MemoryCluster mc-bad-top)
(SchemaVersion mc-bad-top medium-memory-v1)
(ClusterType mc-bad-top parser-test)
(ClusterOpenedAt mc-bad-top "now")
(ClusterSource mc-bad-top src-test)
(Contains mc-bad-top e1)
"not an atom"
''')

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

    def test_reject_missing_schema_version(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            with self.assertRaisesRegex(ValidationError, "SchemaVersion"):
                store.append_cluster("""
(MemoryCluster mc1)
(ClusterType mc1 episode-record)
(ClusterOpenedAt mc1 "now")
(ClusterSource mc1 src-test)
(Contains mc1 e1)
""")

    def test_query_by_id_type_about_status_role_and_cluster(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(VALID_CLUSTER)
            self.assertEqual(store.query_cluster("mc1").cluster_id, "mc1")
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
            self.assertTrue(view.startswith("(ClusterStatus") or view.startswith("(About") or view.startswith("(HasStatus"))

    def test_prompt_view_prefers_topic_status_salience_then_recency(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(OLD_LOW_SALIENCE_CLUSTER)
            store.append_cluster(NEW_OTHER_TOPIC_CLUSTER)
            store.append_cluster(HIGH_SALIENCE_DONE_CLUSTER)
            view = store.prompt_view(topics={"MediumPeTTaMemory"}, statuses={"active"}, limit_chars=2000)
            self.assertLess(view.find("old but relevant"), view.find("done.txt"))
            self.assertLess(view.find("done.txt"), view.find("new but unrelated"))

    def test_pln_view_filters_raw_quotes_and_unpromoted_claims(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(QUOTE_CLUSTER)
            view = store.pln_view()
            self.assertNotIn("RawUtterance", view)
            self.assertNotIn("ClaimText", view)
            self.assertNotIn("QuotedClaim", view)
            self.assertNotIn("DerivedBelief b1", view)
            self.assertNotIn("BeliefContent b1", view)
            self.assertNotIn("SpeechEvent se1", view)
            self.assertNotIn("About qc1 PLN", view)

    def test_pln_view_includes_explicitly_promoted_belief(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(QUOTE_CLUSTER)
            store.append_cluster(PROMOTED_BELIEF_CLUSTER)
            view = store.pln_view()
            self.assertIn("PromotionEvent pe1", view)
            self.assertIn("DerivedBelief b2", view)
            self.assertIn("BeliefContent b2", view)
            self.assertIn("TruthValue b2", view)

    def test_current_status_uses_supersession_events(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(STATUS_OPEN)
            self.assertEqual(store.current_status("c1"), "active")
            store.append_cluster(STATUS_RESOLVED)
            self.assertEqual(store.current_status("c1"), "resolved")
            self.assertEqual([c.cluster_id for c in store.query_status("resolved")], ["mc-status-resolved"])

    def test_delimiter_mismatch_rejected_on_read(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "medium_memory.metta"
            path.write_text(""";;; BEGIN MemoryCluster mc1
(MemoryCluster mc1)
(SchemaVersion mc1 medium-memory-v1)
(ClusterType mc1 episode-record)
(ClusterOpenedAt mc1 "now")
(ClusterSource mc1 src-test)
(Contains mc1 e1)
;;; END MemoryCluster other
""")
            store = MediumMemoryStore(path)
            with self.assertRaises(ValidationError):
                store.clusters()

    def test_reject_duplicate_ids_on_append(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(VALID_CLUSTER)
            with self.assertRaisesRegex(ValidationError, "duplicate ids"):
                store.append_cluster(VALID_CLUSTER.replace("mc1", "mc-duplicate"))


if __name__ == "__main__":
    unittest.main()
