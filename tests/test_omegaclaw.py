import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory import LiveWriteDisabled, MediumMemoryStore, OmegaClawMemoryBridge, OmegaClawMemoryPolicy, ValidationError
from petta_memory.sexpr import parse_top_level_lists


ROOT = Path(__file__).resolve().parents[1]


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
            self.assertEqual(bridge.index_view_metta(generated_at="fixed"), "")

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

    def test_omegaclaw_style_fixture_produces_prompt_view_and_index_context(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            fixture = ROOT / "fixtures" / "omegaclaw_prompt_context.metta"
            for cluster_text in _fixture_clusters(fixture):
                store.append_cluster(cluster_text)
            bridge = OmegaClawMemoryBridge(
                store,
                OmegaClawMemoryPolicy(
                    prompt_view_reads_enabled=True,
                    prompt_view_limit_chars=420,
                    prompt_topics=frozenset({"ProtomegabotMemory"}),
                    prompt_statuses=frozenset({"active"}),
                    view_id="oc-ggb-memory-gate",
                ),
            )

            prompt_context = bridge.prompt_view_metta(generated_at="2026-07-01T15:30:00+00:00")
            index_context = store.index_view(limit_chars=2000)

            self.assertIn("(OmegaClawPromptView oc-ggb-memory-gate)", prompt_context)
            self.assertIn("(PromptViewMode oc-ggb-memory-gate read-only)", prompt_context)
            self.assertIn("ProtomegabotMemory", prompt_context)
            self.assertIn("Keep OmegaClaw memory reads bounded", prompt_context)
            self.assertNotIn("Unrelated scheduling detail", prompt_context)
            self.assertNotIn("MemoryCluster mc-oc-fixture-boundary", prompt_context)
            self.assertIn("(MM-index-about ProtomegabotMemory cmt-oc-boundary mc-oc-fixture-boundary)", index_context)
            self.assertIn("(MM-index-about ProtomegabotMemory q-oc-next mc-oc-fixture-task)", index_context)
            self.assertIn("(MM-index-status active cmt-oc-boundary mc-oc-fixture-boundary)", index_context)
            parse_top_level_lists(prompt_context)
            parse_top_level_lists(index_context)

    def test_index_view_wrapper_is_separately_flagged_bounded_and_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            store.append_cluster(VALID_CLUSTER)
            bridge = OmegaClawMemoryBridge(
                store,
                OmegaClawMemoryPolicy(
                    index_view_reads_enabled=True,
                    index_view_limit_chars=140,
                    index_view_id="oc-test-index-view",
                ),
            )
            view = bridge.index_view_metta(generated_at='bad "timestamp"\nkept')
            self.assertIn(";;; BEGIN OmegaClawIndexView oc-test-index-view", view)
            self.assertIn("(IndexViewMode oc-test-index-view read-only-derived)", view)
            self.assertIn('(IndexViewGeneratedAt oc-test-index-view "bad \\"timestamp\\"\\nkept")', view)
            self.assertIn("(MM-index-id mc-oc1 mc-oc1)", view)
            self.assertNotIn("(MemoryCluster mc-oc1)", view)
            parse_top_level_lists(view)

    def test_invalid_prompt_view_policy_ids_are_rejected(self):
        with self.assertRaises(ValidationError):
            OmegaClawMemoryPolicy(view_id='bad id)')

        with self.assertRaises(ValidationError):
            OmegaClawMemoryPolicy(index_view_id='bad id)')

    def test_negative_index_view_policy_limit_is_rejected(self):
        with self.assertRaises(ValidationError):
            OmegaClawMemoryPolicy(index_view_limit_chars=-1)

    def test_autonomous_write_flag_and_write_method_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "medium_memory.metta")
            with self.assertRaises(LiveWriteDisabled):
                OmegaClawMemoryPolicy(autonomous_writes_enabled=True)
            bridge = OmegaClawMemoryBridge(store)
            with self.assertRaises(LiveWriteDisabled):
                bridge.append_from_omegaclaw(VALID_CLUSTER)


def _fixture_clusters(path: Path) -> list[str]:
    return [
        chunk.strip()
        for chunk in path.read_text(encoding="utf-8").split(";;; --- cluster ---")
        if "(MemoryCluster" in chunk
    ]


if __name__ == "__main__":
    unittest.main()
