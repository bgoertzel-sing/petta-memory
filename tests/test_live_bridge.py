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
        self.assertEqual(bridge["patham9_runtime_gate"]["status"], "skipped")
        self.assertTrue(bridge["checks"]["no_task_or_directive_claim"])
        self.assertTrue(bridge["checks"]["no_memory_write"])

    def test_live_bridge_can_run_patham9_runtime_gate_over_admitted_handoff(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        calls = []

        def fake_patham9_runner(handoff, *, pln_repo, timeout_sec):
            calls.append((handoff, pln_repo, timeout_sec))
            return {
                "schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-result-v1",
                "status": "passed",
                "returncode": 0,
                "semantic_markers": {"passed_true_count": 1, "passed_false_count": 0, "semantic_passed": True},
                "program": {"schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1"},
            }

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            bridge = run_petta_memory_goalchainer_live_bridge(
                journal,
                goalchainer_repo=repo,
                cache_id="bridge-patham9-test",
                query_target="(Acceptable publish_redacted_summary)",
                require_query_relevance=True,
                include_patham9_runtime=True,
                pln_repo="/tmp/patham9-pln",
                patham9_timeout_sec=7.0,
                patham9_runner=fake_patham9_runner,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["schema"], "petta-memory-patham9-pln-handoff-v1")
        self.assertEqual(calls[0][0]["item_count"], bridge["input_counts"]["admitted_items"])
        self.assertEqual(calls[0][1], "/tmp/patham9-pln")
        self.assertEqual(calls[0][2], 7.0)
        self.assertTrue(bridge["patham9_runtime_gate"]["enabled"])
        self.assertEqual(bridge["patham9_runtime_gate"]["status"], "passed")
        self.assertTrue(bridge["checks"]["patham9_runtime_passed_or_skipped"])

    def test_live_bridge_rejects_empty_journal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValidationError, "non-empty journal"):
                run_petta_memory_goalchainer_live_bridge(Path(td) / "empty.metta")


if __name__ == "__main__":
    unittest.main()
