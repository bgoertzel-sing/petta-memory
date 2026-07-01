import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory import LiveWriteDisabled, MediumMemoryStore, OmegaClawMemoryBridge, OmegaClawMemoryPolicy, ValidationError
from petta_memory.sexpr import parse_top_level_lists


VALID_CLUSTER = """
(MemoryCluster mc-oc1)
(SchemaVersion mc-oc1 medium-memory-v1)
(ClusterType mc-oc1 integration-sketch)
(ClusterOpenedAt mc-oc1 "2026-06-29 11:10 PDT")
(ClusterSource mc-oc1 local-test)
(Contains mc-oc1 cmt-oc1)
(ClusterStatus mc-oc1 active)
(Commitment cmt-oc1)
(CommitmentText cmt-oc1 "Keep OmegaClaw writes disabled")
(About cmt-oc1 OmegaClawMemoryBoundary)
(HasStatus cmt-oc1 open)
"""

OTHER_CLUSTER = """
(MemoryCluster mc-oc2)
(SchemaVersion mc-oc2 medium-memory-v1)
(ClusterType mc-oc2 integration-sketch)
(ClusterOpenedAt mc-oc2 "2026-06-29 11:11 PDT")
(ClusterSource mc-oc2 local-test)
(Contains mc-oc2 q-oc2)
(ClusterStatus mc-oc2 active)
(OpenQuestion q-oc2)
(QuestionText q-oc2 "Unrelated prompt fragment")
(About q-oc2 OtherTopic)
"""


class OmegaClawMemoryBridgeTests(unittest.TestCase):
    def test_prompt_view_reads_are_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(VALID_CLUSTER)
            bridge = OmegaClawMemoryBridge(store)
            self.assertEqual(bridge.prompt_view_metta(generated_at="fixed"), "")

    def test_prompt_view_read_wrapper_is_bounded_and_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(VALID_CLUSTER)
            bridge = OmegaClawMemoryBridge(
                store,
                OmegaClawMemoryPolicy(prompt_view_reads_enabled=True, prompt_view_limit_chars=120),
            )
            view = bridge.prompt_view_metta(generated_at="2026-06-29T18:10:00+00:00")
            self.assertIn(";;; BEGIN OmegaClawPromptView oc-prompt-memory-view", view)
            self.assertIn("(PromptViewMode oc-prompt-memory-view read-only)", view)
            self.assertIn('(PromptViewGeneratedAt oc-prompt-memory-view "2026-06-29T18:10:00+00:00")', view)
            self.assertIn("(CommitmentText cmt-oc1", view)
            self.assertNotIn("MemoryCluster mc-oc1", view)

    def test_prompt_view_policy_can_prefer_topics_and_statuses(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(OTHER_CLUSTER)
            store.append_cluster(VALID_CLUSTER)
            bridge = OmegaClawMemoryBridge(
                store,
                OmegaClawMemoryPolicy(
                    prompt_view_reads_enabled=True,
                    prompt_topics=frozenset({"OmegaClawMemoryBoundary"}),
                    prompt_statuses=frozenset({"open"}),
                ),
            )
            view = bridge.prompt_view_metta(generated_at="fixed")
            self.assertLess(view.find("OmegaClawMemoryBoundary"), view.find("OtherTopic"))

    def test_prompt_view_wrapper_escapes_generated_timestamp(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(VALID_CLUSTER)
            bridge = OmegaClawMemoryBridge(store, OmegaClawMemoryPolicy(prompt_view_reads_enabled=True))
            view = bridge.prompt_view_metta(generated_at='bad "timestamp"\\with newline\nkept')
            self.assertIn('(PromptViewGeneratedAt oc-prompt-memory-view "bad \\"timestamp\\"\\\\with newline\\nkept")', view)
            parse_top_level_lists(view)

    def test_invalid_prompt_view_policy_ids_are_rejected(self):
        with self.assertRaises(ValidationError):
            OmegaClawMemoryPolicy(view_id='bad id)')

    def test_autonomous_write_flag_and_write_method_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            with self.assertRaises(LiveWriteDisabled):
                OmegaClawMemoryPolicy(autonomous_writes_enabled=True)
            bridge = OmegaClawMemoryBridge(store)
            with self.assertRaises(LiveWriteDisabled):
                bridge.append_from_omegaclaw(VALID_CLUSTER)


if __name__ == "__main__":
    unittest.main()
