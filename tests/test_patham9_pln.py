import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.patham9_pln import (
    classify_smoke_result,
    classify_smoke_result_with_retry,
    parse_metta_test_output,
    patham9_pln_handoff_sentences,
    patham9_pln_query_smoke_program,
    summarize_smoke_results,
    summarize_smoke_results_file,
)


class Patham9PlnSmokeGateTests(unittest.TestCase):
    def test_parse_metta_test_output_counts_semantic_failures(self):
        parsed = parse_metta_test_output(
            "[(TestResult (Passed: #t))]\n"
            "[(TestResult (Passed: #f))]\n"
            "[(Error (import! &self PLN) Failed to resolve module top:PLN)]\n"
        )

        self.assertEqual(parsed["passed_true_count"], 1)
        self.assertEqual(parsed["passed_false_count"], 1)
        self.assertEqual(parsed["error_markers"], 1)
        self.assertFalse(parsed["semantic_passed"])
        self.assertEqual(len(parsed["diagnostic_lines"]), 3)

    def test_classify_smoke_result_treats_shell_success_with_passed_false_as_failure(self):
        result = classify_smoke_result(
            {
                "test": "ruletests/example.metta",
                "returncode": 0,
                "output": "[((Is: x) (Should: y) (Passed: #f))]",
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("Passed: #f", " ".join(result["reasons"]))

    def test_classify_smoke_result_requires_at_least_one_passed_true_marker(self):
        result = classify_smoke_result({"test": "empty.metta", "returncode": 0, "output": "[()]"})

        self.assertEqual(result["status"], "failed")
        self.assertIn("no Passed: #t markers", result["reasons"])

    def test_summarize_smoke_results_reports_failed_tests(self):
        summary = summarize_smoke_results(
            [
                {"test": "examples/Smokes.metta", "returncode": 0, "passed_true_count": 1},
                {"test": "ruletests/inversion.metta", "returncode": 0, "error_markers": 1},
            ]
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["results"][1]["test"], "ruletests/inversion.metta")
        self.assertIn("Passed: #f and Error atoms are failures", summary["gate"])

    def test_summarize_smoke_results_file_loads_artifact_list(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.json"
            path.write_text(
                '[{"test": "examples/FlyingRaven.metta", "returncode": 0, "passed_true_count": 2}]',
                encoding="utf-8",
            )

            summary = summarize_smoke_results_file(path)

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["total"], 1)

    def test_classify_smoke_result_with_retry_distinguishes_harness_drift(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "ruletest.log"
            retry = Path(td) / "ruletest.retry.log"
            log.write_text("[(Error (import! ModuleSpace(GroundingSpace-top) PLN) Failed)]", encoding="utf-8")
            retry.write_text("[((Is: x) (Should: x) (Passed: #t))]", encoding="utf-8")

            result = classify_smoke_result_with_retry(
                {
                    "test": "ruletests/inversion.metta",
                    "returncode": 0,
                    "error_markers": 1,
                    "log": str(log),
                }
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["attempt"], "retry")
        self.assertEqual(result["primary_status"], "failed")
        self.assertEqual(result["classification"], "harness-or-environment-drift")

    def test_summarize_smoke_results_can_include_retry_logs(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "broken.log"
            retry = Path(td) / "broken.retry.log"
            log.write_text("[(Error bad import)]", encoding="utf-8")
            retry.write_text("[((Is: x) (Should: x) (Passed: #t))]", encoding="utf-8")

            summary = summarize_smoke_results(
                [{"test": "ruletests/broken.metta", "returncode": 0, "error_markers": 1, "log": str(log)}],
                include_retries=True,
            )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["results"][0]["attempt"], "retry")

    def test_patham9_pln_handoff_sentences_preserve_stv_and_contextual_packets(self):
        cache = {
            "schema": "petta-memory-pettachainer-handoff-v1",
            "cache_id": "pm-cache-test",
            "items": [
                {
                    "kind": "pettachainer-stv-statement",
                    "atom": "(: b2 (Requires MediumPeTTaMemory PLNReadyViews) (STV 0.90 0.70))",
                    "belief_id": "b2",
                    "cluster_id": "mc3",
                    "promotion_event": "pe1",
                    "promotion_rule": "explicit-test-promotion",
                    "promotion_domain": "memory-architecture",
                    "item_status": "pln-ready-input-not-inferred-belief",
                },
                {
                    "kind": "pettachainer-evidence-packet",
                    "atom": "(EvidencePacket (Requires MediumPeTTaMemory PLNReadyViews) (EC 8.0 2.0) ((domain memory-architecture) (promotion-rule explicit-test-promotion)) pe1)",
                    "belief_id": "b2",
                    "cluster_id": "mc3",
                    "promotion_rule": "explicit-test-promotion",
                    "promotion_domain": "memory-architecture",
                },
            ],
        }

        handoff = patham9_pln_handoff_sentences(cache)

        self.assertEqual(handoff["schema"], "petta-memory-patham9-pln-handoff-v1")
        self.assertEqual(handoff["item_count"], 1)
        item = handoff["items"][0]
        self.assertEqual(
            item["atom"],
            "(Sentence (Requires MediumPeTTaMemory PLNReadyViews) (stv 0.90 0.70) "
            "((PMEvidence b2 mc3 pe1 explicit-test-promotion memory-architecture)))",
        )
        self.assertEqual(item["stv"], {"strength": "0.90", "confidence": "0.70"})
        packets = item["pi_pln_extension"]["contextual_evidence_packets"]
        self.assertEqual(packets[0]["support"], "8.0")
        self.assertEqual(packets[0]["opposition"], "2.0")
        self.assertIn("not appended to memory", handoff["boundary"])

    def test_patham9_pln_handoff_sentences_reject_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "expected petta-memory-pettachainer-handoff-v1"):
            patham9_pln_handoff_sentences({"schema": "other", "items": []})

    def test_patham9_pln_query_smoke_program_uses_numeric_stamp_and_preserves_provenance(self):
        handoff = {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "items": [
                {
                    "atom": "(Sentence (Requires MediumPeTTaMemory PLNReadyViews) (stv 0.90 0.70) ((PMEvidence b2 mc3 pe1 rule domain)))",
                    "term": "(Requires MediumPeTTaMemory PLNReadyViews)",
                    "stv": {"strength": "0.90", "confidence": "0.70"},
                    "evidence_id": "(PMEvidence b2 mc3 pe1 rule domain)",
                    "pi_pln_extension": {"contextual_evidence_packets": [{"support": "8", "opposition": "2"}]},
                }
            ],
        }

        program = patham9_pln_query_smoke_program(handoff)

        self.assertIn("(Sentence ((Requires MediumPeTTaMemory PLNReadyViews) (stv 0.90 0.70)) (0))", program["program"])
        self.assertIn("(PLN.Query", program["program"])
        self.assertIn("!(Test", program["program"])
        self.assertEqual(program["runtime_stamp"], "(0)")
        self.assertEqual(program["source_evidence_id"], "(PMEvidence b2 mc3 pe1 rule domain)")
        self.assertEqual(program["source_item"]["pi_pln_extension"]["contextual_evidence_packets"][0]["support"], "8")


if __name__ == "__main__":
    unittest.main()
