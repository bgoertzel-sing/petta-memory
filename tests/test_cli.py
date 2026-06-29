import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "petta_memory.cli"]

CLUSTER = """
(MemoryCluster mc-cli)
(SchemaVersion mc-cli medium-memory-v1)
(ClusterType mc-cli cli-test)
(ClusterOpenedAt mc-cli "2026-06-27 14:20 PDT")
(ClusterSource mc-cli src-test)
(Contains mc-cli b1)
(ClusterStatus mc-cli active)
(DerivedBelief b1)
(EpistemicRole b1 derived-belief)
(BeliefContent b1 (Requires MediumPeTTaMemory CLI))
(TruthValue b1 (stv 0.80 0.60))
(EvidenceFor b1 src-test)
(PromotionEvent pe-cli)
(PromotesTo pe-cli b1)
(About b1 MediumPeTTaMemory)
"""

QUOTE_CLUSTER = """
(MemoryCluster mc-quote-cli)
(SchemaVersion mc-quote-cli medium-memory-v1)
(ClusterType mc-quote-cli quote-test)
(ClusterOpenedAt mc-quote-cli "2026-06-27 14:24 PDT")
(ClusterSource mc-quote-cli src-test)
(Contains mc-quote-cli qc-cli)
(ClusterStatus mc-quote-cli active)
(QuotedClaim qc-cli)
(ClaimText qc-cli "quoted claim text")
(About qc-cli PLN)
(RawUtterance qc-cli "raw quote")
"""

OTHER_TOPIC_CLUSTER = """
(MemoryCluster mc-other-cli)
(SchemaVersion mc-other-cli medium-memory-v1)
(ClusterType mc-other-cli prompt-test)
(ClusterOpenedAt mc-other-cli "2026-06-27 14:25 PDT")
(ClusterSource mc-other-cli src-test)
(Contains mc-other-cli q-other-cli)
(ClusterStatus mc-other-cli active)
(OpenQuestion q-other-cli)
(QuestionText q-other-cli "other topic question")
(About q-other-cli OtherTopic)
"""


class CliTests(unittest.TestCase):
    def run_cli(self, args, *, input_text=None):
        env = {"PYTHONPATH": str(ROOT / "src")}
        return subprocess.run(CLI + args, input=input_text, text=True, capture_output=True, env=env, check=False)

    def test_append_query_and_pln_view(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "medium_memory.metta")
            result = self.run_cli(["--store", store, "append"], input_text=CLUSTER)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("mc-cli", result.stdout)

            query = self.run_cli(["--store", store, "query", "about", "MediumPeTTaMemory"])
            self.assertEqual(query.returncode, 0, query.stderr)
            self.assertIn("DerivedBelief b1", query.stdout)

            pln = self.run_cli(["--store", store, "pln-view"])
            self.assertEqual(pln.returncode, 0, pln.stderr)
            self.assertIn("TruthValue b1", pln.stdout)

    def test_pln_view_exclude_extends_default_exclusions(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "medium_memory.metta")
            self.assertEqual(self.run_cli(["--store", store, "append"], input_text=QUOTE_CLUSTER).returncode, 0)
            pln = self.run_cli(["--store", store, "pln-view", "--exclude", "About"])
            self.assertEqual(pln.returncode, 0, pln.stderr)
            self.assertNotIn("ClaimText", pln.stdout)
            self.assertNotIn("RawUtterance", pln.stdout)
            self.assertNotIn("About qc-cli PLN", pln.stdout)

    def test_prompt_view_accepts_topic_and_status_preferences(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "medium_memory.metta")
            self.assertEqual(self.run_cli(["--store", store, "append"], input_text=OTHER_TOPIC_CLUSTER).returncode, 0)
            self.assertEqual(self.run_cli(["--store", store, "append"], input_text=CLUSTER).returncode, 0)
            prompt = self.run_cli(["--store", store, "prompt-view", "--topic", "MediumPeTTaMemory", "--status", "active"])
            self.assertEqual(prompt.returncode, 0, prompt.stderr)
            self.assertLess(prompt.stdout.find("MediumPeTTaMemory"), prompt.stdout.find("OtherTopic"))


if __name__ == "__main__":
    unittest.main()
