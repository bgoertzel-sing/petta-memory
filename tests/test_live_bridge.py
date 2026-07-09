import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.live_bridge import run_petta_memory_goalchainer_live_bridge
from petta_memory.store import MediumMemoryStore, ValidationError


class LiveBridgeTests(unittest.TestCase):
    def test_live_bridge_reads_journal_and_runs_goalchainer_memory_probe(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            bridge = run_petta_memory_goalchainer_live_bridge(
                journal,
                goalchainer_repo=repo,
                cache_id="bridge-test",
                query_target="(Acceptable publish_redacted_summary)",
                require_query_relevance=True,
            )

        self.assertEqual(bridge["schema"], "petta-memory-goalchainer-live-bridge-v1")
        self.assertEqual(bridge["mode"], "read-only-live-journal-to-local-goalchainer")
        self.assertEqual(bridge["cluster_count"], 1)
        self.assertGreaterEqual(bridge["input_counts"]["goalchainer_items"], 2)
        self.assertEqual(bridge["goalchainer_gate"]["recommended_action"], "publish_redacted_summary")
        self.assertEqual(bridge["goalchainer_gate"]["recommended_status"], "recommended")
        self.assertTrue(bridge["goalchainer_gate"]["heuristic_memory_probe"]["memory_proof_present"])
        self.assertTrue(bridge["checks"]["admitted_handoff_built"])
        self.assertTrue(bridge["checks"]["no_task_or_directive_claim"])
        self.assertTrue(bridge["checks"]["no_memory_write"])

    def test_live_bridge_rejects_empty_journal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValidationError, "non-empty journal"):
                run_petta_memory_goalchainer_live_bridge(Path(td) / "empty.metta")


if __name__ == "__main__":
    unittest.main()
