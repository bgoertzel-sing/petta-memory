import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory import LiveWriteDisabled, MediumMemoryStore, OmegaClawMemoryBridge, OmegaClawMemoryPolicy


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
