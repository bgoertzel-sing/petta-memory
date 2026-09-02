import json
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
(EvidenceSupportCount b1 3.0)
(EvidenceOppositionCount b1 1.0)
(PromotionEvent pe-cli)
(PromotesTo pe-cli b1)
(PromotionRule pe-cli explicit-cli-test)
(PromotionTrust pe-cli 0.75)
(PromotionDomain pe-cli memory-cli)
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

            pln = self.run_cli(["--store", store, "pln-view", "--normalized"])
            self.assertEqual(pln.returncode, 0, pln.stderr)
            self.assertIn("TruthValue b1", pln.stdout)
            self.assertIn("MM-PLNTrust b1 0.75", pln.stdout)

            pettachainer = self.run_cli(["--store", store, "pettachainer-view"])
            self.assertEqual(pettachainer.returncode, 0, pettachainer.stderr)
            self.assertIn("(: b1 (Requires MediumPeTTaMemory CLI) (STV 0.80 0.60))", pettachainer.stdout)

            packets = self.run_cli(["--store", store, "pettachainer-packets-view"])
            self.assertEqual(packets.returncode, 0, packets.stderr)
            self.assertIn(
                "(EvidencePacket (Requires MediumPeTTaMemory CLI) (EC 3.0 1.0) "
                "((domain memory-cli) (promotion-rule explicit-cli-test)) pe-cli)",
                packets.stdout,
            )

            handoff = self.run_cli(["--store", store, "pettachainer-handoff-cache"])
            self.assertEqual(handoff.returncode, 0, handoff.stderr)
            cache = json.loads(handoff.stdout)
            self.assertEqual(cache["mode"], "non-live-precompiled-statement-cache")
            self.assertEqual(cache["item_count"], 2)
            self.assertEqual(cache["items"][0]["kind"], "pettachainer-stv-statement")
            self.assertEqual(cache["items"][1]["kind"], "pettachainer-evidence-packet")

            goal_handoff = self.run_cli(["--store", store, "goalchainer-handoff-cache"])
            self.assertEqual(goal_handoff.returncode, 0, goal_handoff.stderr)
            goal_cache = json.loads(goal_handoff.stdout)
            self.assertEqual(goal_cache["schema"], "petta-memory-goalchainer-handoff-v1")
            self.assertEqual(goal_cache["decision_gate"], "disabled-no-live-omegaclaw-skill-no-task-claim")
            self.assertEqual(goal_cache["items"][0]["goalchainer_slot"], "acceptability-belief-evidence")
            self.assertEqual(goal_cache["items"][1]["goalchainer_slot"], "contextual-appraisal-evidence")

            patham9_handoff = self.run_cli(["--store", store, "patham9-pln-handoff"])
            self.assertEqual(patham9_handoff.returncode, 0, patham9_handoff.stderr)
            pln_cache = json.loads(patham9_handoff.stdout)
            self.assertEqual(pln_cache["schema"], "petta-memory-patham9-pln-handoff-v1")
            self.assertEqual(pln_cache["sentence_format"], "(Sentence $Term (stv S C) ($EvidenceID))")
            self.assertIn("(Sentence (Requires MediumPeTTaMemory CLI) (stv 0.80 0.60)", pln_cache["items"][0]["atom"])
            self.assertEqual(pln_cache["items"][0]["pi_pln_extension"]["contextual_evidence_packets"][0]["support"], "3.0")

            ranked_plan = self.run_cli([
                "--store",
                store,
                "pi-pln-ranked-plan",
                "--query-target",
                "MediumPeTTaMemory",
                "--require-query-relevance",
                "--controller-min-strength",
                "0.5",
                "--controller-min-confidence",
                "0.5",
                "--seed",
                "7",
            ])
            self.assertEqual(ranked_plan.returncode, 0, ranked_plan.stderr)
            plan = json.loads(ranked_plan.stdout)
            self.assertEqual(plan["schema"], "petta-memory-pi-pln-ranked-inference-control-plan-v1")
            self.assertEqual(plan["recommended_count"], 1)
            self.assertEqual(plan["recommended_branches"][0]["belief_id"], "b1")
            self.assertIn("no PLN.Query/PLN.Derive call", plan["boundary"])

            admitted = self.run_cli([
                "--store",
                store,
                "pi-pln-admitted-handoff",
                "--query-target",
                "MediumPeTTaMemory",
                "--require-query-relevance",
                "--controller-min-strength",
                "0.5",
                "--controller-min-confidence",
                "0.5",
                "--seed",
                "7",
            ])
            self.assertEqual(admitted.returncode, 0, admitted.stderr)
            admitted_payload = json.loads(admitted.stdout)
            self.assertEqual(admitted_payload["schema"], "petta-memory-pi-pln-admitted-handoff-v1")
            self.assertEqual(admitted_payload["admitted_count"], 1)
            self.assertEqual(admitted_payload["admitted_handoff"]["items"][0]["belief_id"], "b1")
            self.assertIn("no PLN.Query/PLN.Derive call", admitted_payload["boundary"])

    def test_audit_view_preserves_complete_records_and_rejects_negative_limit(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "medium_memory.metta")
            self.assertEqual(self.run_cli(["--store", store, "append"], input_text=CLUSTER).returncode, 0)
            full = self.run_cli(["--store", store, "audit-view"])
            self.assertEqual(full.returncode, 0, full.stderr)
            self.assertIn(";;; BEGIN MemoryCluster mc-cli", full.stdout)
            self.assertIn(";;; END MemoryCluster mc-cli", full.stdout)
            self.assertIn("(DerivedBelief b1)", full.stdout)

            tiny = self.run_cli(["--store", store, "audit-view", "--limit-chars", "1"])
            self.assertEqual(tiny.returncode, 0, tiny.stderr)
            self.assertEqual(tiny.stdout, "")

            negative = self.run_cli(["--store", store, "audit-view", "--limit-chars", "-1"])
            self.assertEqual(negative.returncode, 2)
            self.assertIn("limit_chars must be non-negative", negative.stderr)

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

    def test_prompt_view_rejects_negative_limit(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "medium_memory.metta")
            self.assertEqual(self.run_cli(["--store", store, "append"], input_text=CLUSTER).returncode, 0)
            prompt = self.run_cli(["--store", store, "prompt-view", "--limit-chars", "-1"])
            self.assertEqual(prompt.returncode, 2)
            self.assertIn("limit_chars must be non-negative", prompt.stderr)

    def test_index_view_generates_retrieval_atoms(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "medium_memory.metta")
            self.assertEqual(self.run_cli(["--store", store, "append"], input_text=CLUSTER).returncode, 0)
            index = self.run_cli(["--store", store, "index-view"])
            self.assertEqual(index.returncode, 0, index.stderr)
            self.assertIn("(MM-index-id b1 mc-cli)", index.stdout)
            self.assertIn("(MM-index-type DerivedBelief b1 mc-cli)", index.stdout)
            self.assertIn("(MM-index-about MediumPeTTaMemory b1 mc-cli)", index.stdout)
            self.assertIn("(MM-index-role derived-belief b1 mc-cli)", index.stdout)

    def test_pln_view_limit_chars_preserves_complete_atom_lines(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "medium_memory.metta")
            self.assertEqual(self.run_cli(["--store", store, "append"], input_text=CLUSTER).returncode, 0)
            full = self.run_cli(["--store", store, "pln-view", "--normalized"])
            self.assertEqual(full.returncode, 0, full.stderr)
            first_line = full.stdout.splitlines()[0]
            bounded = self.run_cli(["--store", store, "pln-view", "--normalized", "--limit-chars", str(len(first_line) + 1)])
            self.assertEqual(bounded.returncode, 0, bounded.stderr)
            self.assertEqual(bounded.stdout, first_line + "\n")

    def test_pln_view_rejects_negative_limit(self):
        with tempfile.TemporaryDirectory() as td:
            store = str(Path(td) / "medium_memory.metta")
            self.assertEqual(self.run_cli(["--store", store, "append"], input_text=CLUSTER).returncode, 0)
            pln = self.run_cli(["--store", store, "pln-view", "--limit-chars", "-1"])
            self.assertEqual(pln.returncode, 2)
            self.assertIn("limit_chars must be non-negative", pln.stderr)


if __name__ == "__main__":
    unittest.main()
