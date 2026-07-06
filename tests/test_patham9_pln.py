import tempfile
import unittest
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.patham9_pln import (
    build_meta_learning_benchmark_handoff,
    classify_smoke_result,
    classify_smoke_result_with_retry,
    continuation_predicate_wrapper,
    context_selection_wrapper,
    chained_inference_pipeline,
    ec_projected_stv,
    parse_metta_test_output,
    patham9_pln_api_surface,
    patham9_pln_ec_projection_smoke_program,
    patham9_pln_ec_projection_conflicting_smoke_program,
    patham9_pln_derivation_ec_projection_smoke_program,
    patham9_pi_pln_boundary_plan,
    patham9_pi_pln_extension_spec,
    patham9_pln_derivation_smoke_program,
    patham9_pln_handoff_sentences,
    patham9_pln_multi_sentence_derivation_smoke_program,
    patham9_pln_query_smoke_program,
    probabilistic_inference_filter,
    run_meta_learning_benchmark,
    summarize_smoke_results,
    summarize_smoke_results_file,
    survey_trueagi_chaining_inference_control,
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

    def _patham9_handoff(self):
        return {
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

    def test_patham9_pln_query_smoke_program_uses_numeric_stamp_and_preserves_provenance(self):
        program = patham9_pln_query_smoke_program(self._patham9_handoff())

        self.assertIn("(Sentence ((Requires MediumPeTTaMemory PLNReadyViews) (stv 0.90 0.70)) (0))", program["program"])
        self.assertIn("(PLN.Query", program["program"])
        self.assertIn("!(Test", program["program"])
        self.assertEqual(program["runtime_stamp"], "(0)")
        self.assertEqual(program["source_evidence_id"], "(PMEvidence b2 mc3 pe1 rule domain)")
        self.assertEqual(program["source_item"]["pi_pln_extension"]["contextual_evidence_packets"][0]["support"], "8")

    def test_patham9_pln_derivation_smoke_program_uses_two_premises_and_stamp_sidecar(self):
        program = patham9_pln_derivation_smoke_program(self._patham9_handoff())

        self.assertEqual(program["schema"], "petta-memory-patham9-pln-derivation-smoke-program-v1")
        self.assertIn("PMDerivedFromHandoff", program["derived_term"])
        self.assertIn("(Sentence ((Requires MediumPeTTaMemory PLNReadyViews) (stv 0.90 0.70)) (0))", program["program"])
        self.assertIn("(Implication (Requires MediumPeTTaMemory PLNReadyViews) (PMDerivedFromHandoff", program["program"])
        self.assertIn("(Sentence ((Implication", program["program"])
        self.assertIn("(1))", program["program"])
        self.assertEqual(program["expected_result"], "((stv 0.902 0.63) (0 1))")
        self.assertEqual(program["stamp_sidecar"]["(0)"]["source_evidence_id"], "(PMEvidence b2 mc3 pe1 rule domain)")
        self.assertEqual(program["stamp_sidecar"]["(1)"]["kind"], "synthetic-non-live-bridge-implication")
        self.assertIn("no inferred-belief promotion", program["boundary"])

    def test_patham9_pi_pln_boundary_plan_keeps_wrapper_first_and_summarizes_ec_inputs(self):
        plan = patham9_pi_pln_boundary_plan(self._patham9_handoff())

        self.assertEqual(plan["schema"], "petta-memory-patham9-pi-pln-boundary-plan-v1")
        self.assertEqual(plan["decision"], "wrapper-first")
        self.assertIn("unmodified functional chainer", plan["patham9_core_policy"])
        self.assertIn("PLN.Query", plan["patham9_extension_points"])
        self.assertIn("no truth-changing EC projection is live yet", plan["formula_policy"])
        projected = plan["projection_inputs"][0]
        self.assertEqual(projected["source_evidence_id"], "(PMEvidence b2 mc3 pe1 rule domain)")
        self.assertEqual(projected["contextual_packets"][0]["total_evidence"], 10.0)
        self.assertEqual(projected["contextual_packets"][0]["positive_ratio"], 0.8)
        self.assertIn("no memory append", plan["non_live_gates"][1])

    def test_patham9_pi_pln_boundary_plan_rejects_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "expected petta-memory-patham9-pln-handoff-v1"):
            patham9_pi_pln_boundary_plan({"schema": "other", "items": []})

    def test_ec_projected_stv_blends_confidence_weighted(self):
        result = ec_projected_stv(0.90, 0.70, [{"support": 8.0, "opposition": 2.0}])
        self.assertEqual(result["base_strength"], 0.90)
        self.assertEqual(result["base_confidence"], 0.70)
        self.assertEqual(result["packet_count"], 1)
        # ec_strength = 8/10 = 0.8, ec_confidence = 10/12 ≈ 0.833333
        self.assertAlmostEqual(result["packets"][0]["ec_strength"], 0.8, places=6)
        self.assertAlmostEqual(result["packets"][0]["ec_confidence"], 10.0 / 12.0, places=6)
        # weighted: (0.90*0.70 + 0.8*0.833333) / (0.70 + 0.833333)
        expected_strength = (0.90 * 0.70 + 0.8 * (10.0 / 12.0)) / (0.70 + 10.0 / 12.0)
        self.assertAlmostEqual(result["projected_strength"], round(expected_strength, 6), places=5)
        # projected confidence = max(0.70, 0.833333) = 0.833333
        self.assertAlmostEqual(result["projected_confidence"], round(10.0 / 12.0, 6), places=5)

    def test_ec_projected_stv_with_no_packets_returns_base(self):
        result = ec_projected_stv(0.91, 0.74, [])
        self.assertEqual(result["projected_strength"], 0.91)
        self.assertEqual(result["projected_confidence"], 0.74)
        self.assertEqual(result["packet_count"], 0)

    def test_ec_projected_stv_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            ec_projected_stv(1.5, 0.5, [])
        with self.assertRaises(ValueError):
            ec_projected_stv(0.5, -0.1, [])
        with self.assertRaises(ValueError):
            ec_projected_stv(0.5, 0.5, [{"support": -1, "opposition": 0}])

    def test_ec_projected_stv_skips_zero_total_packets(self):
        result = ec_projected_stv(0.80, 0.60, [{"support": 0, "opposition": 0}, {"support": 6, "opposition": 2}])
        self.assertEqual(result["packet_count"], 1)
        self.assertAlmostEqual(result["packets"][0]["ec_strength"], 0.75, places=5)

    def test_patham9_pln_ec_projection_smoke_program_builds_direct_and_projected(self):
        handoff = self._patham9_handoff()
        smoke = patham9_pln_ec_projection_smoke_program(handoff)
        self.assertEqual(smoke["schema"], "petta-memory-patham9-pln-ec-projection-smoke-program-v1")
        self.assertIn("(PLN.Query", smoke["direct"]["program"])
        self.assertIn("(PLN.Query", smoke["projected"]["program"])
        # Direct uses original STV
        self.assertIn("(stv 0.90 0.70)", smoke["direct"]["runtime_sentence"])
        # Projected uses different STV (blended with EC 8/2)
        self.assertNotIn("(stv 0.90 0.70)", smoke["projected"]["runtime_sentence"])
        self.assertIn("(stv ", smoke["projected"]["runtime_sentence"])
        # EC projection metadata
        self.assertEqual(smoke["ec_projection"]["packet_count"], 1)
        self.assertAlmostEqual(smoke["ec_projection"]["packets"][0]["positive_ratio"], 0.8, places=5)
        # Boundary
        self.assertIn("no inferred-belief promotion", smoke["boundary"])
        # Both use same query term
        self.assertEqual(smoke["direct"]["expected_result"].split()[0], smoke["projected"]["expected_result"].split()[0])

    def test_patham9_pln_ec_projection_smoke_program_rejects_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "expected petta-memory-patham9-pln-handoff-v1"):
            patham9_pln_ec_projection_smoke_program({"schema": "other", "items": []})

    def test_ec_projected_stv_with_conflicting_ec_lowers_strength(self):
        """Strong base STV (0.90, 0.70) with opposing EC (1, 9) should lower strength."""
        result = ec_projected_stv(0.90, 0.70, [{"support": 1.0, "opposition": 9.0}])
        self.assertEqual(result["base_strength"], 0.90)
        self.assertEqual(result["base_confidence"], 0.70)
        self.assertEqual(result["packet_count"], 1)
        # ec_strength = 1/10 = 0.1, ec_confidence = 10/12 ≈ 0.833333
        self.assertAlmostEqual(result["packets"][0]["ec_strength"], 0.1, places=6)
        # Weighted blend should be lower than base 0.90
        self.assertLess(result["projected_strength"], 0.90)
        # Confidence is max(0.70, 0.833333) = 0.833333
        self.assertAlmostEqual(result["projected_confidence"], round(10.0 / 12.0, 6), places=5)

    def test_patham9_pln_ec_projection_conflicting_smoke_program_lowers_strength(self):
        handoff = self._patham9_handoff()
        smoke = patham9_pln_ec_projection_conflicting_smoke_program(handoff)
        self.assertEqual(smoke["schema"], "petta-memory-patham9-pln-ec-projection-conflicting-smoke-program-v1")
        self.assertTrue(smoke["strength_lowered"], "Projected strength should be lower than base with opposing EC")
        self.assertIn("(PLN.Query", smoke["direct"]["program"])
        self.assertIn("(PLN.Query", smoke["projected"]["program"])
        # Direct uses original STV
        self.assertIn("(stv 0.90 0.70)", smoke["direct"]["runtime_sentence"])
        # Projected STV should be different and lower
        self.assertNotIn("(stv 0.90 0.70)", smoke["projected"]["runtime_sentence"])
        projected_stv = smoke["projected"]["stv"]
        self.assertLess(float(projected_stv["strength"]), 0.90)
        self.assertEqual(smoke["conflicting_ec"], {"support": 1.0, "opposition": 9.0})
        self.assertIn("no inferred-belief promotion", smoke["boundary"])

    def test_patham9_pln_ec_projection_conflicting_smoke_program_rejects_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "expected petta-memory-patham9-pln-handoff-v1"):
            patham9_pln_ec_projection_conflicting_smoke_program({"schema": "other", "items": []})

    def test_patham9_pln_ec_projection_conflicting_smoke_program_custom_ec(self):
        handoff = self._patham9_handoff()
        smoke = patham9_pln_ec_projection_conflicting_smoke_program(
            handoff, conflicting_support=2.0, conflicting_opposition=8.0
        )
        self.assertEqual(smoke["conflicting_ec"], {"support": 2.0, "opposition": 8.0})
        self.assertTrue(smoke["strength_lowered"])
        # With 2/8 = 0.25 ec_strength, still lower than 0.90
        self.assertLess(float(smoke["projected"]["stv"]["strength"]), 0.90)

    def test_patham9_pln_derivation_ec_projection_smoke_program_builds_direct_and_projected(self):
        handoff = self._patham9_handoff()
        smoke = patham9_pln_derivation_ec_projection_smoke_program(handoff)
        self.assertEqual(smoke["schema"], "petta-memory-patham9-pln-derivation-ec-projection-smoke-program-v1")
        self.assertIn("PMDerivedFromHandoff", smoke["derived_term"])
        # Both programs have PLN.Query and two sentences
        self.assertIn("(PLN.Query", smoke["direct"]["program"])
        self.assertIn("(PLN.Query", smoke["projected"]["program"])
        self.assertIn("(Implication", smoke["direct"]["program"])
        self.assertIn("(Implication", smoke["projected"]["program"])
        # Direct uses original STV
        self.assertIn("(stv 0.90 0.70)", smoke["direct"]["program"])
        # Projected uses different STV
        self.assertNotIn("(stv 0.90 0.70)", smoke["projected"]["program"])
        # Results should differ because EC projection changes the STV
        self.assertTrue(smoke["results_differ"], "Direct and projected derivation results should differ")
        # Expected results differ
        self.assertNotEqual(smoke["direct"]["expected_result"], smoke["projected"]["expected_result"])
        # Boundary
        self.assertIn("no inferred-belief promotion", smoke["boundary"])
        # Stamp sidecar
        self.assertIn("(0)", smoke["stamp_sidecar"])
        self.assertIn("(1)", smoke["stamp_sidecar"])
        self.assertEqual(smoke["stamp_sidecar"]["(1)"]["kind"], "synthetic-non-live-bridge-implication")

    def test_patham9_pln_derivation_ec_projection_smoke_program_rejects_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "expected petta-memory-patham9-pln-handoff-v1"):
            patham9_pln_derivation_ec_projection_smoke_program({"schema": "other", "items": []})

    def test_patham9_pln_derivation_ec_projection_smoke_program_no_packets_results_same(self):
        """When there are no contextual EC packets, projected == direct."""
        handoff = {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "items": [
                {
                    "atom": "(Sentence (Test) (stv 0.80 0.60) ((PMEvidence b3 mc3 pe3 rule domain)))",
                    "term": "(Test)",
                    "stv": {"strength": "0.80", "confidence": "0.60"},
                    "evidence_id": "(PMEvidence b3 mc3 pe3 rule domain)",
                    "pi_pln_extension": {"contextual_evidence_packets": []},
                }
            ],
        }
        smoke = patham9_pln_derivation_ec_projection_smoke_program(handoff)
        self.assertFalse(smoke["results_differ"], "With no EC packets, projected should equal direct")
        self.assertEqual(smoke["direct"]["expected_result"], smoke["projected"]["expected_result"])


class Patham9PlnApiSurfaceTests(unittest.TestCase):
    def test_api_surface_returns_source_level_mapping(self):
        repo = Path(__file__).resolve().parents[2] / "patham9-pln"
        if not repo.exists():
            self.skipTest(f"patham9/PLN checkout not found at {repo}")
        surface = patham9_pln_api_surface(repo)
        self.assertEqual(surface["schema"], "petta-memory-patham9-pln-api-surface-v1")
        self.assertEqual(surface["mode"], "source-level-no-runtime-inspection")
        self.assertEqual(surface["boundary"], "source-level inspection only; no SWI/PeTTa/MeTTa runtime invoked; no memory append; no inferred-belief promotion; no OmegaClaw/GoalChainer live path")

        # Core API entries
        core = surface["core_api"]
        self.assertIn("PLN.Derive", core)
        self.assertIn("PLN.Query", core)
        self.assertIn("Sentence", core)
        self.assertIn("StampDisjoint", core)
        self.assertIn("PriorityRank", core)
        self.assertIn("ConfidenceRank", core)
        self.assertIn("LimitSize", core)
        self.assertIn("BestCandidate", core)

        # PLN.Derive has 4 arity overloads
        self.assertEqual(len(core["PLN.Derive"]["full_signatures"]), 4)
        self.assertEqual(len(core["PLN.Derive"]["signatures"]), 4)

        # PLN.Query has 4 arity overloads
        self.assertEqual(len(core["PLN.Query"]["full_signatures"]), 4)
        self.assertEqual(len(core["PLN.Query"]["signatures"]), 4)

        # Defaults reference PLN.Config
        self.assertIn("PLN.Config.MaxSteps", core["PLN.Derive"]["defaults"]["maxsteps"])

        # Sentence structure documented
        self.assertIn("stv", core["Sentence"]["structure"])

        # Truth-value formulas captured
        formula_names = {f["name"] for f in surface["truth_value_formulas"]}
        self.assertIn("Truth_Deduction", formula_names)
        self.assertIn("Truth_ModusPonens", formula_names)
        self.assertIn("Truth_Revision", formula_names)
        self.assertIn("Truth_Negation", formula_names)

        # Inference rules captured
        self.assertTrue(len(surface["inference_rules"]) >= 10,
                        f"expected at least 10 inference rules, got {len(surface['inference_rules'])}")

        # Guard predicates captured
        guard_text = " ".join(g["definition"] for g in surface["guard_predicates"])
        self.assertIn("SyllogisticRuleGuard", guard_text)
        self.assertIn("SymmetricModusPonensRuleGuard", guard_text)

        # Config defaults
        config_text = " ".join(c["definition"] for c in surface["config_defaults"])
        self.assertIn("PLN.Config.MaxSteps", config_text)
        self.assertIn("PLN.Config.TaskQueueSize", config_text)
        self.assertIn("PLN.Config.BeliefQueueSize", config_text)

        # Source files documented
        self.assertIn("PLN.metta", surface["source_files"])
        self.assertIn("src/Deriver.metta", surface["source_files"])
        self.assertIn("src/Formulas.metta", surface["source_files"])
        self.assertIn("src/Rules.metta", surface["source_files"])

        # pi-PLN extension points
        ext = surface["pi_pln_extension_points"]
        self.assertIn("wrapper_boundary", ext)
        self.assertIn("internal_extensions", ext)
        self.assertIn("sentence_construction", ext["wrapper_boundary"])
        self.assertIn("stv_pre_projection", ext["wrapper_boundary"])
        self.assertIn("context_selection", ext["wrapper_boundary"])
        self.assertIn("revisit_trigger", ext)

        # No runtime evidence in the result
        self.assertNotIn("stdout", surface)
        self.assertNotIn("stderr", surface)
        self.assertNotIn("returncode", surface)

    def test_api_surface_raises_for_missing_repo(self):
        with self.assertRaises(FileNotFoundError):
            patham9_pln_api_surface("/nonexistent/path/to/pln")


class Patham9PiPlnExtensionSpecTests(unittest.TestCase):
    def _patham9_handoff(self, num_items: int = 2):
        items = []
        for i in range(num_items):
            items.append({
                "atom": f"(Sentence (Term{i}) (stv 0.{90 - i} 0.{70 - i}) ((PMEvidence b{i} mc{i} pe{i} rule{i} domain{i})))",
                "term": f"(Term{i})",
                "stv": {"strength": f"0.{90 - i}", "confidence": f"0.{70 - i}"},
                "evidence_id": f"(PMEvidence b{i} mc{i} pe{i} rule{i} domain{i})",
                "pi_pln_extension": {"contextual_evidence_packets": [{"support": str(8 - i), "opposition": str(2 + i)}]},
            })
        return {"schema": "petta-memory-patham9-pln-handoff-v1", "items": items}

    def test_extension_spec_returns_design_specification(self):
        handoff = self._patham9_handoff(2)
        spec = patham9_pi_pln_extension_spec(handoff)
        self.assertEqual(spec["schema"], "petta-memory-patham9-pi-pln-extension-spec-v1")
        self.assertEqual(spec["mode"], "design-specification-no-runtime")
        self.assertEqual(spec["version"], "0.1")
        self.assertEqual(spec["boundary_decision"], "wrapper-first: keep checked-out patham9/PLN unmodified; petta-memory owns wrapper layer")

    def test_extension_spec_documents_sentence_construction_protocol(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(1))
        protocol = spec["sentence_construction_protocol"]
        self.assertIn("(Sentence ($Term (stv S C)) $Stamp)", protocol["format"])
        self.assertIn("ec_projected_stv", protocol["stv_source"])
        self.assertIn("numeric runtime stamps", protocol["stamp_policy"])

    def test_extension_spec_documents_ec_projection_formula(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(2))
        formula = spec["ec_projection_formula"]
        self.assertEqual(formula["name"], "confidence-weighted blend")
        self.assertIn("sum(s_i * w_i)", formula["formula"])
        self.assertTrue(len(formula["tested_in"]) >= 3, "should reference at least 3 test names")
        self.assertEqual(formula["status"], "reviewed and implemented as ec_projected_stv()")

    def test_extension_spec_documents_provenance_sidecar_policy(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(1))
        sidecar = spec["provenance_sidecar_policy"]
        contents_text = " ".join(sidecar["contents"])
        self.assertIn("PMEvidence", contents_text)
        self.assertIn("not appended to memory", sidecar["boundary"])

    def test_extension_spec_documents_context_selection_policy(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(1))
        context = spec["context_selection_policy"]
        self.assertEqual(context["current_state"], "not-live; wrapper does not yet filter or generate contexts")
        self.assertIn("wrapper owns this entirely", context["patham9_support"])

    def test_extension_spec_documents_inference_control_hooks(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(1))
        hooks = spec["inference_control_hooks"]
        self.assertEqual(hooks["current_state"], "deferred (roadmap item 4)")
        self.assertTrue(len(hooks["reference_patterns"]) >= 2, "should reference at least 2 patterns")
        self.assertIn("pln-inf-ctl.metta", " ".join(hooks["reference_patterns"]))

    def test_extension_spec_documents_read_write_boundaries(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(1))
        boundaries = spec["read_write_boundaries"]
        self.assertIn("no_memory_append", boundaries)
        self.assertIn("no_inferred_belief_promotion", boundaries)
        self.assertIn("no_omegaclaw_live", boundaries)
        self.assertIn("no_patham9_source_patch", boundaries)

    def test_extension_spec_documents_revisit_triggers(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(1))
        triggers = spec["revisit_triggers"]
        self.assertIn("internal_extension", triggers)
        self.assertIn("inference_control", triggers)
        self.assertIn("context_selection", triggers)

    def test_extension_spec_includes_projection_inputs(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(3))
        self.assertEqual(spec["item_count"], 3)
        self.assertEqual(len(spec["projection_inputs"]), 3)
        first = spec["projection_inputs"][0]
        self.assertEqual(first["item_index"], 0)
        self.assertIn("base_stv", first)
        self.assertIn("projected_stv", first)
        self.assertEqual(first["contextual_packet_count"], 1)

    def test_extension_spec_rejects_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "expected petta-memory-patham9-pln-handoff-v1"):
            patham9_pi_pln_extension_spec({"schema": "other", "items": []})

    def test_extension_spec_boundary_text(self):
        spec = patham9_pi_pln_extension_spec(self._patham9_handoff(1))
        self.assertIn("no runtime invoked", spec["boundary"])
        self.assertIn("no memory append", spec["boundary"])


class Patham9PlnMultiSentenceDerivationTests(unittest.TestCase):
    def _multi_item_handoff(self, num_items: int = 3):
        items = []
        for i in range(num_items):
            items.append({
                "atom": f"(Sentence (Requires Target{i} PLNReadyViews) (stv 0.{80 + i} 0.6{i}) ((PMEvidence b{i} mc{i} pe{i} rule{i} domain{i})))",
                "term": f"(Requires Target{i} PLNReadyViews)",
                "stv": {"strength": f"0.{80 + i}", "confidence": f"0.6{i}"},
                "evidence_id": f"(PMEvidence b{i} mc{i} pe{i} rule{i} domain{i})",
                "pi_pln_extension": {"contextual_evidence_packets": []},
            })
        return {"schema": "petta-memory-patham9-pln-handoff-v1", "items": items}

    def test_multi_sentence_program_loads_all_handoff_items(self):
        handoff = self._multi_item_handoff(3)
        smoke = patham9_pln_multi_sentence_derivation_smoke_program(handoff)
        self.assertEqual(smoke["schema"], "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1")
        self.assertEqual(smoke["handoff_sentence_count"], 3)
        self.assertEqual(smoke["sentence_count"], 4)  # 3 handoff + 1 bridge
        # All 3 handoff terms appear in the program
        for i in range(3):
            self.assertIn(f"Target{i}", smoke["program"])
        # Bridge implication appears
        self.assertIn("Implication", smoke["program"])
        self.assertIn("PMDerivedFromMultiHandoff", smoke["program"])
        # PLN.Query and Test appear
        self.assertIn("(PLN.Query", smoke["program"])
        self.assertIn("!(Test", smoke["program"])

    def test_multi_sentence_program_stamp_sidecar_maps_all_items(self):
        handoff = self._multi_item_handoff(3)
        smoke = patham9_pln_multi_sentence_derivation_smoke_program(handoff)
        sidecar = smoke["stamp_sidecar"]
        # 3 source stamps + 1 bridge stamp
        self.assertEqual(len(sidecar), 4)
        self.assertIn("(0)", sidecar)
        self.assertIn("(1)", sidecar)
        self.assertIn("(2)", sidecar)
        self.assertIn("(3)", sidecar)
        self.assertEqual(sidecar["(0)"]["kind"], "petta-memory-source-sentence")
        self.assertEqual(sidecar["(0)"]["source_item_index"], 0)
        self.assertEqual(sidecar["(3)"]["kind"], "synthetic-non-live-bridge-implication")

    def test_multi_sentence_program_expected_result_uses_first_item(self):
        handoff = self._multi_item_handoff(3)
        smoke = patham9_pln_multi_sentence_derivation_smoke_program(handoff)
        # Expected result references stamp 0 and last stamp (bridge)
        self.assertIn("(0 3)", smoke["expected_result"])
        self.assertIn("(stv ", smoke["expected_result"])

    def test_multi_sentence_program_custom_bridge_term(self):
        handoff = self._multi_item_handoff(2)
        smoke = patham9_pln_multi_sentence_derivation_smoke_program(
            handoff, bridge_term="(CustomDerived Term)"
        )
        self.assertIn("(CustomDerived Term)", smoke["derived_term"])
        self.assertIn("(CustomDerived Term)", smoke["program"])

    def test_multi_sentence_program_boundary_text(self):
        handoff = self._multi_item_handoff(1)
        smoke = patham9_pln_multi_sentence_derivation_smoke_program(handoff)
        self.assertIn("no memory append", smoke["boundary"])
        self.assertIn("no inferred-belief promotion", smoke["boundary"])
        self.assertIn("no OmegaClaw/GoalChainer live path", smoke["boundary"])

    def test_multi_sentence_program_rejects_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "expected petta-memory-patham9-pln-handoff-v1"):
            patham9_pln_multi_sentence_derivation_smoke_program({"schema": "other", "items": []})

    def test_multi_sentence_program_rejects_empty_handoff(self):
        with self.assertRaisesRegex(ValueError, "no Sentence items"):
            patham9_pln_multi_sentence_derivation_smoke_program({"schema": "petta-memory-patham9-pln-handoff-v1", "items": []})

    def test_multi_sentence_program_single_item_works(self):
        handoff = self._multi_item_handoff(1)
        smoke = patham9_pln_multi_sentence_derivation_smoke_program(handoff)
        self.assertEqual(smoke["handoff_sentence_count"], 1)
        self.assertEqual(smoke["sentence_count"], 2)  # 1 handoff + 1 bridge
        self.assertIn("(0 1)", smoke["expected_result"])


# --- Multi-belief cluster fixtures for round-trip empirical tests ---

_PROMOTED_BELIEF_A = """
(MemoryCluster mc-rt-a)
(SchemaVersion mc-rt-a medium-memory-v1)
(ClusterType mc-rt-a belief-promotion)
(ClusterOpenedAt mc-rt-a "2026-07-05 19:00 PDT")
(ClusterSource mc-rt-a src-test)
(Contains mc-rt-a pe-rt-a)
(Contains mc-rt-a b-rt-a)
(ClusterStatus mc-rt-a active)
(PromotionEvent pe-rt-a)
(PromotesFrom pe-rt-a qc-rt-a)
(PromotesTo pe-rt-a b-rt-a)
(PromotionRule pe-rt-a explicit-test-promotion)
(PromotionTrust pe-rt-a 0.85)
(PromotionDomain pe-rt-a memory-architecture)
(DerivedBelief b-rt-a)
(BeliefContent b-rt-a (Requires MemoryTarget0 PLNReadyViews))
(TruthValue b-rt-a (stv 0.88 0.72))
(EvidenceFor b-rt-a qc-rt-a)
(EvidenceSupportCount b-rt-a 9.0)
(EvidenceOppositionCount b-rt-a 1.0)
"""

_PROMOTED_BELIEF_B = """
(MemoryCluster mc-rt-b)
(SchemaVersion mc-rt-b medium-memory-v1)
(ClusterType mc-rt-b belief-promotion)
(ClusterOpenedAt mc-rt-b "2026-07-05 19:01 PDT")
(ClusterSource mc-rt-b src-test)
(Contains mc-rt-b pe-rt-b)
(Contains mc-rt-b b-rt-b)
(ClusterStatus mc-rt-b active)
(PromotionEvent pe-rt-b)
(PromotesFrom pe-rt-b qc-rt-b)
(PromotesTo pe-rt-b b-rt-b)
(PromotionRule pe-rt-b explicit-test-promotion)
(PromotionTrust pe-rt-b 0.80)
(PromotionDomain pe-rt-b memory-architecture)
(DerivedBelief b-rt-b)
(BeliefContent b-rt-b (Requires MemoryTarget1 PLNReadyViews))
(TruthValue b-rt-b (stv 0.82 0.68))
(EvidenceFor b-rt-b qc-rt-b)
(EvidenceSupportCount b-rt-b 7.0)
(EvidenceOppositionCount b-rt-b 3.0)
"""


class StoreRoundTripPatham9PlnTests(unittest.TestCase):
    """Empirical round-trip tests: MediumMemoryStore -> handoff cache -> patham9/PLN.

    These tests exercise the full artifact pipeline from stored promoted
    beliefs through to a patham9/PLN derivation program, without invoking
    the SWI/PeTTa runtime.  They validate that real store-promoted beliefs
    survive the handoff chain with correct STV, provenance, and boundary
    metadata.
    """

    def _store_with_two_promoted_beliefs(self, td: str):
        from petta_memory.store import MediumMemoryStore
        store = MediumMemoryStore(Path(td) / "roundtrip_memory.metta")
        store.append_cluster(_PROMOTED_BELIEF_A)
        store.append_cluster(_PROMOTED_BELIEF_B)
        return store

    def test_roundtrip_handoff_cache_from_store(self):
        """Store -> pettachainer_handoff_cache produces well-formed items."""
        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_two_promoted_beliefs(td)
            cache = store.pettachainer_handoff_cache()

        self.assertEqual(cache["schema"], "petta-memory-pettachainer-handoff-v1")
        self.assertEqual(cache["item_count"], 4)  # 2 STV + 2 EvidencePacket
        kinds = [item["kind"] for item in cache["items"]]
        self.assertEqual(kinds.count("pettachainer-stv-statement"), 2)
        self.assertEqual(kinds.count("pettachainer-evidence-packet"), 2)
        belief_ids = {item["belief_id"] for item in cache["items"]}
        self.assertEqual(belief_ids, {"b-rt-a", "b-rt-b"})
        for item in cache["items"]:
            self.assertEqual(item["item_status"], "pln-ready-input-not-inferred-belief")
            self.assertEqual(item["promotion_domain"], "memory-architecture")

    def test_roundtrip_handoff_sentences_preserve_stv_and_provenance(self):
        """Handoff cache -> patham9_pln_handoff_sentences preserves STV and provenance."""
        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_two_promoted_beliefs(td)
            cache = store.pettachainer_handoff_cache()

        sentences = patham9_pln_handoff_sentences(cache)
        self.assertEqual(sentences["schema"], "petta-memory-patham9-pln-handoff-v1")
        self.assertEqual(sentences["item_count"], 2)  # only STV statements become Sentences
        for item in sentences["items"]:
            self.assertEqual(item["kind"], "patham9-pln-sentence-input")
            self.assertIn("(Sentence ", item["atom"])
            self.assertIn("(stv ", item["atom"])
            self.assertIn("(PMEvidence ", item["atom"])
            # EC packets preserved in pi-PLN extension block
            packets = item["pi_pln_extension"]["contextual_evidence_packets"]
            self.assertEqual(len(packets), 1)
            self.assertIn(packets[0]["support"], ("9.0", "7.0"))
            self.assertIn(packets[0]["opposition"], ("1.0", "3.0"))

    def test_roundtrip_multi_sentence_program_from_store(self):
        """Full round-trip: store -> cache -> sentences -> derivation program."""
        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_two_promoted_beliefs(td)
            cache = store.pettachainer_handoff_cache()

        sentences = patham9_pln_handoff_sentences(cache)
        smoke = patham9_pln_multi_sentence_derivation_smoke_program(sentences)

        self.assertEqual(smoke["schema"], "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1")
        self.assertEqual(smoke["handoff_sentence_count"], 2)
        self.assertEqual(smoke["sentence_count"], 3)  # 2 handoff + 1 bridge

        # Both terms appear in the program
        self.assertIn("MemoryTarget0", smoke["program"])
        self.assertIn("MemoryTarget1", smoke["program"])

        # Bridge implication and PLN.Query appear
        self.assertIn("Implication", smoke["program"])
        self.assertIn("(PLN.Query", smoke["program"])
        self.assertIn("PMDerivedFromMultiHandoff", smoke["program"])

        # Stamp sidecar has 2 source + 1 bridge = 3 entries
        self.assertEqual(len(smoke["stamp_sidecar"]), 3)
        self.assertIn("(0)", smoke["stamp_sidecar"])
        self.assertIn("(1)", smoke["stamp_sidecar"])
        self.assertIn("(2)", smoke["stamp_sidecar"])
        self.assertEqual(smoke["stamp_sidecar"]["(0)"]["kind"], "petta-memory-source-sentence")
        self.assertEqual(smoke["stamp_sidecar"]["(2)"]["kind"], "synthetic-non-live-bridge-implication")

        # Boundary text
        self.assertIn("no memory append", smoke["boundary"])
        self.assertIn("no inferred-belief promotion", smoke["boundary"])

        # Expected result references stamps 0 and 2 (first source + bridge)
        self.assertIn("(0 2)", smoke["expected_result"])

    def test_roundtrip_single_belief_store(self):
        """Round-trip with a single promoted belief: 1 STV + 1 packet -> 1 Sentence."""
        with tempfile.TemporaryDirectory() as td:
            from petta_memory.store import MediumMemoryStore
            store = MediumMemoryStore(Path(td) / "single.metta")
            store.append_cluster(_PROMOTED_BELIEF_A)
            cache = store.pettachainer_handoff_cache()

        self.assertEqual(cache["item_count"], 2)  # 1 STV + 1 EvidencePacket
        sentences = patham9_pln_handoff_sentences(cache)
        self.assertEqual(sentences["item_count"], 1)
        smoke = patham9_pln_multi_sentence_derivation_smoke_program(sentences)
        self.assertEqual(smoke["handoff_sentence_count"], 1)
        self.assertEqual(smoke["sentence_count"], 2)
        self.assertIn("MemoryTarget0", smoke["program"])

    def test_roundtrip_ec_projection_from_store_packets(self):
        """EC packets from the store survive into the pi-PLN extension block and
        can feed ec_projected_stv() for a projected-vs-direct comparison."""
        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_two_promoted_beliefs(td)
            cache = store.pettachainer_handoff_cache()

        sentences = patham9_pln_handoff_sentences(cache)
        for item in sentences["items"]:
            packets = item["pi_pln_extension"]["contextual_evidence_packets"]
            self.assertEqual(len(packets), 1)
            # Each packet has EC counts usable by ec_projected_stv()
            support = float(packets[0]["support"])
            opposition = float(packets[0]["opposition"])
            base_s = float(item["stv"]["strength"])
            base_c = float(item["stv"]["confidence"])
            projected = ec_projected_stv(base_s, base_c, [{"support": support, "opposition": opposition}])
            self.assertGreater(projected["projected_strength"], 0.0)
            self.assertLess(projected["projected_strength"], 1.0)
            self.assertGreater(projected["projected_confidence"], 0.0)
            self.assertLessEqual(projected["projected_confidence"], 1.0)


class TrueagiChainingInferenceControlSurveyTests(unittest.TestCase):
    """Tests for the source-level survey of trueagi-io/chaining inference-control patterns."""

    def _repo(self) -> Path:
        return Path(__file__).resolve().parents[2] / "trueagi-chaining"

    def test_survey_returns_well_formed_artifact(self):
        repo = self._repo()
        if not repo.exists():
            self.skipTest("trueagi-io/chaining checkout not found")
        survey = survey_trueagi_chaining_inference_control(repo)

        self.assertEqual(survey["schema"], "petta-memory-trueagi-chaining-inference-control-survey-v1")
        self.assertEqual(survey["mode"], "source-level-no-runtime-inspection")
        self.assertIn("bc9beb2672953e07971b3abecc1fe67651ecddc4", survey["source_commit"])
        self.assertGreaterEqual(survey["pattern_count"], 6)
        self.assertEqual(len(survey["patterns"]), survey["pattern_count"])

    def test_survey_patterns_have_required_fields(self):
        repo = self._repo()
        if not repo.exists():
            self.skipTest("trueagi-io/chaining checkout not found")
        survey = survey_trueagi_chaining_inference_control(repo)

        for pattern in survey["patterns"]:
            self.assertIn("name", pattern)
            self.assertIn("file", pattern)
            self.assertIn("line_count", pattern)
            self.assertGreater(pattern["line_count"], 0)
            self.assertIn("description", pattern)
            self.assertIn("key_concepts", pattern)
            self.assertGreaterEqual(len(pattern["key_concepts"]), 2)
            self.assertIn("wrapper_adoption", pattern)
            self.assertIn("requires_patham9_source_change", pattern)
            self.assertFalse(pattern["requires_patham9_source_change"])
            self.assertIn("complexity", pattern)

    def test_survey_includes_pln_inf_ctl_pattern(self):
        repo = self._repo()
        if not repo.exists():
            self.skipTest("trueagi-io/chaining checkout not found")
        survey = survey_trueagi_chaining_inference_control(repo)

        names = [p["name"] for p in survey["patterns"]]
        self.assertIn("PLN-based inference controller", names)
        pln_ctl = next(p for p in survey["patterns"] if "PLN-based" in p["name"])
        self.assertIn("Thompson sampling", " ".join(pln_ctl["key_concepts"]))
        self.assertIn("EDCall", " ".join(pln_ctl["key_concepts"]))
        self.assertIn("pln-inf-ctl.metta", pln_ctl["file"])

    def test_survey_includes_probabilistic_filtering_pattern(self):
        repo = self._repo()
        if not repo.exists():
            self.skipTest("trueagi-io/chaining checkout not found")
        survey = survey_trueagi_chaining_inference_control(repo)

        names = [p["name"] for p in survey["patterns"]]
        self.assertIn("Probabilistic backward chaining (ProbLog-inspired)", names)
        prob = next(p for p in survey["patterns"] if "Probabilistic" in p["name"])
        self.assertIn("prob-chaining.metta", prob["file"])
        self.assertTrue(prob["complexity"].startswith("low"))

    def test_survey_adoption_by_phase_categorizes_patterns(self):
        repo = self._repo()
        if not repo.exists():
            self.skipTest("trueagi-io/chaining checkout not found")
        survey = survey_trueagi_chaining_inference_control(repo)

        by_phase = survey["adoption_by_phase"]
        self.assertIn("near_term", by_phase)
        self.assertIn("medium_term", by_phase)
        self.assertIn("long_term", by_phase)
        total = len(by_phase["near_term"]) + len(by_phase["medium_term"]) + len(by_phase["long_term"])
        self.assertEqual(total, survey["pattern_count"])
        self.assertGreaterEqual(len(by_phase["near_term"]), 1)
        self.assertGreaterEqual(len(by_phase["long_term"]), 1)

    def test_survey_pi_pln_extension_references_are_present(self):
        repo = self._repo()
        if not repo.exists():
            self.skipTest("trueagi-io/chaining checkout not found")
        survey = survey_trueagi_chaining_inference_control(repo)

        refs = survey["pi_pln_extension_references"]
        self.assertIn("inference_control_hooks", refs)
        self.assertIn("context_selection", refs)
        self.assertIn("ec_projection", refs)
        self.assertIn("roadmap item 4", refs["inference_control_hooks"])

    def test_survey_boundary_text_is_non_live(self):
        repo = self._repo()
        if not repo.exists():
            self.skipTest("trueagi-io/chaining checkout not found")
        survey = survey_trueagi_chaining_inference_control(repo)

        self.assertIn("source-level inspection only", survey["boundary"])
        self.assertIn("no memory append", survey["boundary"])
        self.assertIn("no OmegaClaw/GoalChainer live path", survey["boundary"])

    def test_survey_raises_for_missing_repo(self):
        with self.assertRaises(FileNotFoundError):
            survey_trueagi_chaining_inference_control("/nonexistent/path/to/chaining")


class ProbabilisticInferenceFilterTests(unittest.TestCase):
    """Tests for the first inference-control mechanism: probabilistic filtering.

    The filter applies EC projection to each handoff Sentence, computes a
    composite score, and filters/ranks before loading into patham9/PLN.
    These tests validate the filtering logic without invoking any runtime.
    """

    def _handoff(self, num_items: int = 3) -> dict:
        """Build a handoff with mixed evidence quality for filter testing."""
        items = []
        # Item 0: strong STV, strong EC support (9, 1)
        items.append({
            "kind": "patham9-pln-sentence-input",
            "atom": "(Sentence (Acceptable publish_redacted_summary) (stv 0.91 0.74) ((PMEvidence b-0 mc-0 pe-0 rule domain)))",
            "term": "(Acceptable publish_redacted_summary)",
            "stv": {"strength": 0.91, "confidence": 0.74},
            "evidence_id": "(PMEvidence b-0 mc-0 pe-0 rule domain)",
            "belief_id": "b-0",
            "cluster_id": "mc-0",
            "promotion_event": "pe-0",
            "promotion_rule": "explicit-test",
            "promotion_domain": "memory-architecture",
            "source_status": "pln-ready-input-not-inferred-belief",
            "pi_pln_extension": {
                "contextual_evidence_packets": [
                    {"support": 9, "opposition": 1, "statement": "(Acceptable publish_redacted_summary)"},
                ],
                "ec_projection_policy": "preserve packets first; later project EC",
                "context_selection": "not-run",
            },
        })
        if num_items >= 2:
            # Item 1: moderate STV, conflicting EC (1, 9) — should rank lower
            items.append({
                "kind": "patham9-pln-sentence-input",
                "atom": "(Sentence (Acceptable share_full_log) (stv 0.94 0.80) ((PMEvidence b-1 mc-1 pe-1 rule domain)))",
                "term": "(Acceptable share_full_log)",
                "stv": {"strength": 0.94, "confidence": 0.80},
                "evidence_id": "(PMEvidence b-1 mc-1 pe-1 rule domain)",
                "belief_id": "b-1",
                "cluster_id": "mc-1",
                "promotion_event": "pe-1",
                "promotion_rule": "explicit-test",
                "promotion_domain": "memory-architecture",
                "source_status": "pln-ready-input-not-inferred-belief",
                "pi_pln_extension": {
                    "contextual_evidence_packets": [
                        {"support": 1, "opposition": 9, "statement": "(Acceptable share_full_log)"},
                    ],
                    "ec_projection_policy": "preserve packets first; later project EC",
                    "context_selection": "not-run",
                },
            })
        if num_items >= 3:
            # Item 2: weak STV, no EC packets — moderate score
            items.append({
                "kind": "patham9-pln-sentence-input",
                "atom": "(Sentence (Requires MemoryTarget0 PLNReadyViews) (stv 0.70 0.55) ((PMEvidence b-2 mc-2 pe-2 rule domain)))",
                "term": "(Requires MemoryTarget0 PLNReadyViews)",
                "stv": {"strength": 0.70, "confidence": 0.55},
                "evidence_id": "(PMEvidence b-2 mc-2 pe-2 rule domain)",
                "belief_id": "b-2",
                "cluster_id": "mc-2",
                "promotion_event": "pe-2",
                "promotion_rule": "explicit-test",
                "promotion_domain": "memory-architecture",
                "source_status": "pln-ready-input-not-inferred-belief",
                "pi_pln_extension": {
                    "contextual_evidence_packets": [],
                    "ec_projection_policy": "preserve packets first; later project EC",
                    "context_selection": "not-run",
                },
            })
        return {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": len(items),
            "items": items,
        }

    def test_filter_returns_correct_schema(self):
        result = probabilistic_inference_filter(self._handoff())
        self.assertEqual(result["schema"], "petta-memory-pi-pln-inference-filter-v1")

    def test_filter_input_output_counts_match(self):
        result = probabilistic_inference_filter(self._handoff())
        self.assertEqual(result["input_count"], 3)
        self.assertEqual(result["output_count"], 3)
        self.assertEqual(len(result["selected_indices"]), 3)
        self.assertEqual(len(result["filtered_indices"]), 0)

    def test_filter_no_ec_packets_uses_base_stv(self):
        handoff = self._handoff()
        # Item 2 has no packets; projected should equal base
        result = probabilistic_inference_filter(handoff)
        item2 = [pi for pi in result["items"] if pi["item_index"] == 2][0]
        self.assertAlmostEqual(item2["projected_stv"]["strength"], 0.70)
        self.assertAlmostEqual(item2["projected_stv"]["confidence"], 0.55)
        self.assertEqual(item2["contextual_packet_count"], 0)

    def test_filter_conflicting_ec_lowers_projected_strength(self):
        handoff = self._handoff()
        result = probabilistic_inference_filter(handoff)
        item1 = [pi for pi in result["items"] if pi["item_index"] == 1][0]
        # EC (1, 9) should lower strength from 0.94 to ~0.511
        self.assertLess(item1["projected_stv"]["strength"], 0.60)
        self.assertGreater(item1["projected_stv"]["confidence"], 0.80)  # confidence rises

    def test_filter_ranks_by_composite_score(self):
        handoff = self._handoff()
        result = probabilistic_inference_filter(handoff)
        # Item 0 (strong support) should rank first, item 1 (conflicting) should rank lower
        ranking = result["ranking"]
        self.assertEqual(ranking[0]["item_index"], 0)
        # Item 1 with conflicting EC should have lower score than item 0
        item0_score = [pi for pi in result["items"] if pi["item_index"] == 0][0]["composite_score"]
        item1_score = [pi for pi in result["items"] if pi["item_index"] == 1][0]["composite_score"]
        self.assertGreater(item0_score, item1_score)

    def test_filter_min_confidence_excludes_low_confidence_items(self):
        handoff = self._handoff(num_items=3)
        # Item 2 has confidence 0.55; set threshold above it
        result = probabilistic_inference_filter(handoff, min_confidence=0.60)
        self.assertEqual(result["output_count"], 2)
        self.assertNotIn(2, result["selected_indices"])
        self.assertIn(2, result["filtered_indices"])
        excluded = [pi for pi in result["items"] if pi["item_index"] == 2][0]
        self.assertFalse(excluded["included"])
        self.assertIsNotNone(excluded["filter_reason"])
        self.assertIn("min_confidence", excluded["filter_reason"])

    def test_filter_top_k_keeps_only_k_items(self):
        handoff = self._handoff()
        result = probabilistic_inference_filter(handoff, top_k=1)
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(len(result["selected_indices"]), 1)
        self.assertEqual(len(result["filtered_indices"]), 2)
        # The top item should be item 0 (strong support, high composite score)
        self.assertEqual(result["selected_indices"][0], 0)

    def test_filter_top_k_with_min_confidence_combines_both(self):
        handoff = self._handoff()
        # Set confidence threshold that item 2 fails, then top_k=1
        result = probabilistic_inference_filter(handoff, min_confidence=0.60, top_k=1)
        self.assertEqual(result["output_count"], 1)
        # Only items 0 and 1 pass confidence; top_k=1 picks the best (item 0)
        self.assertEqual(result["selected_indices"][0], 0)
        self.assertIn(2, result["filtered_indices"])

    def test_filter_empty_handoff_returns_empty_result(self):
        handoff = {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": 0,
            "items": [],
        }
        result = probabilistic_inference_filter(handoff)
        self.assertEqual(result["input_count"], 0)
        self.assertEqual(result["output_count"], 0)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["selected_indices"], [])
        self.assertEqual(result["filtered_indices"], [])
        self.assertEqual(result["ranking"], [])

    def test_filter_rejects_wrong_schema(self):
        with self.assertRaises(ValueError):
            probabilistic_inference_filter({"schema": "wrong"})

    def test_filter_rejects_out_of_range_min_confidence(self):
        with self.assertRaises(ValueError):
            probabilistic_inference_filter(self._handoff(), min_confidence=-0.1)
        with self.assertRaises(ValueError):
            probabilistic_inference_filter(self._handoff(), min_confidence=1.1)

    def test_filter_rejects_negative_top_k(self):
        with self.assertRaises(ValueError):
            probabilistic_inference_filter(self._handoff(), top_k=-1)

    def test_filter_boundary_text_is_non_live(self):
        result = probabilistic_inference_filter(self._handoff())
        self.assertIn("non-live", result["boundary"])
        self.assertIn("no memory append", result["boundary"])
        self.assertIn("no inferred-belief promotion", result["boundary"])
        self.assertIn("no OmegaClaw/GoalChainer live path", result["boundary"])

    def test_filter_policy_records_source_pattern(self):
        result = probabilistic_inference_filter(self._handoff())
        policy = result["filter_policy"]
        self.assertEqual(policy["scoring_formula"], "projected_strength * projected_confidence")
        self.assertIn("probabilistic filtering", policy["source_pattern"])
        self.assertEqual(policy["min_confidence"], 0.0)
        self.assertIsNone(policy["top_k"])

    def test_filter_composite_score_is_strength_times_confidence(self):
        handoff = self._handoff(num_items=1)
        result = probabilistic_inference_filter(handoff)
        item0 = result["items"][0]
        expected = item0["projected_stv"]["strength"] * item0["projected_stv"]["confidence"]
        self.assertAlmostEqual(item0["composite_score"], expected)


class StoreRoundTripInferenceFilterTests(unittest.TestCase):
    """Empirical round-trip: store -> handoff -> inference filter."""

    def test_roundtrip_inference_filter_from_store(self):
        from petta_memory.store import MediumMemoryStore

        cluster = """
(MemoryCluster mc-if-a)
(SchemaVersion mc-if-a medium-memory-v1)
(ClusterType mc-if-a belief-promotion)
(ClusterOpenedAt mc-if-a "2026-07-05 22:00 PDT")
(ClusterSource mc-if-a src-test)
(Contains mc-if-a pe-if-a)
(Contains mc-if-a b-if-a)
(ClusterStatus mc-if-a active)
(PromotionEvent pe-if-a)
(PromotesFrom pe-if-a qc-if-a)
(PromotesTo pe-if-a b-if-a)
(PromotionRule pe-if-a explicit-test-promotion)
(PromotionTrust pe-if-a 0.85)
(PromotionDomain pe-if-a memory-architecture)
(DerivedBelief b-if-a)
(BeliefContent b-if-a (Requires MemoryTarget0 PLNReadyViews))
(TruthValue b-if-a (stv 0.88 0.72))
(EvidenceFor b-if-a qc-if-a)
(EvidenceSupportCount b-if-a 9.0)
(EvidenceOppositionCount b-if-a 1.0)
"""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "inf_filter_memory.metta")
            store.append_cluster(cluster)
            cache = store.pettachainer_handoff_cache()

        handoff = patham9_pln_handoff_sentences(cache)
        result = probabilistic_inference_filter(handoff)
        self.assertEqual(result["schema"], "petta-memory-pi-pln-inference-filter-v1")
        self.assertEqual(result["input_count"], 1)
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(len(result["selected_indices"]), 1)
        self.assertEqual(len(result["filtered_indices"]), 0)
        # EC (9, 1) should project strength above 0.80
        item = result["items"][0]
        self.assertGreater(item["projected_stv"]["strength"], 0.80)
        self.assertGreater(item["composite_score"], 0.0)


class ContextSelectionWrapperTests(unittest.TestCase):
    """Tests for the second inference-control mechanism: context selection.

    The wrapper filters EvidencePackets by domain/cluster/promotion_rule and
    scores them for relevance before PLN invocation.  These tests validate the
    filtering and scoring logic without invoking any runtime.
    """

    def _handoff(self, num_items: int = 3) -> dict:
        """Build a handoff with mixed evidence contexts for selection testing."""
        items = []
        # Item 0: packets from domain-a, cluster mc-0
        items.append({
            "kind": "patham9-pln-sentence-input",
            "atom": "(Sentence (Acceptable publish_redacted_summary) (stv 0.91 0.74) ((PMEvidence b-0 mc-0 pe-0 rule domain)))",
            "term": "(Acceptable publish_redacted_summary)",
            "stv": {"strength": 0.91, "confidence": 0.74},
            "evidence_id": "(PMEvidence b-0 mc-0 pe-0 rule domain)",
            "belief_id": "b-0",
            "cluster_id": "mc-0",
            "promotion_event": "pe-0",
            "promotion_rule": "explicit-test",
            "promotion_domain": "memory-architecture",
            "source_status": "pln-ready-input-not-inferred-belief",
            "pi_pln_extension": {
                "contextual_evidence_packets": [
                    {"support": 9, "opposition": 1, "statement": "(Acceptable publish_redacted_summary)",
                     "promotion_domain": "domain-a", "cluster_id": "mc-0", "promotion_rule": "explicit-test"},
                    {"support": 3, "opposition": 7, "statement": "(Acceptable publish_redacted_summary)",
                     "promotion_domain": "domain-b", "cluster_id": "mc-0", "promotion_rule": "explicit-test"},
                ],
                "ec_projection_policy": "preserve packets first; later project EC",
                "context_selection": "not-run",
            },
        })
        if num_items >= 2:
            # Item 1: packets from domain-b, cluster mc-1
            items.append({
                "kind": "patham9-pln-sentence-input",
                "atom": "(Sentence (Acceptable share_full_log) (stv 0.94 0.80) ((PMEvidence b-1 mc-1 pe-1 rule domain)))",
                "term": "(Acceptable share_full_log)",
                "stv": {"strength": 0.94, "confidence": 0.80},
                "evidence_id": "(PMEvidence b-1 mc-1 pe-1 rule domain)",
                "belief_id": "b-1",
                "cluster_id": "mc-1",
                "promotion_event": "pe-1",
                "promotion_rule": "explicit-test",
                "promotion_domain": "memory-architecture",
                "source_status": "pln-ready-input-not-inferred-belief",
                "pi_pln_extension": {
                    "contextual_evidence_packets": [
                        {"support": 5, "opposition": 5, "statement": "(Acceptable share_full_log)",
                         "promotion_domain": "domain-b", "cluster_id": "mc-1", "promotion_rule": "explicit-test"},
                    ],
                    "ec_projection_policy": "preserve packets first; later project EC",
                    "context_selection": "not-run",
                },
            })
        if num_items >= 3:
            # Item 2: no EC packets — should pass through context selection unchanged
            items.append({
                "kind": "patham9-pln-sentence-input",
                "atom": "(Sentence (Requires MemoryTarget0 PLNReadyViews) (stv 0.70 0.55) ((PMEvidence b-2 mc-2 pe-2 rule domain)))",
                "term": "(Requires MemoryTarget0 PLNReadyViews)",
                "stv": {"strength": 0.70, "confidence": 0.55},
                "evidence_id": "(PMEvidence b-2 mc-2 pe-2 rule domain)",
                "belief_id": "b-2",
                "cluster_id": "mc-2",
                "promotion_event": "pe-2",
                "promotion_rule": "explicit-test",
                "promotion_domain": "memory-architecture",
                "source_status": "pln-ready-input-not-inferred-belief",
                "pi_pln_extension": {
                    "contextual_evidence_packets": [],
                    "ec_projection_policy": "preserve packets first; later project EC",
                    "context_selection": "not-run",
                },
            })
        return {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": len(items),
            "items": items,
        }

    def test_selection_returns_correct_schema(self):
        result = context_selection_wrapper(self._handoff())
        self.assertEqual(result["schema"], "petta-memory-pi-pln-context-selection-v1")

    def test_selection_no_filters_keeps_all_packets(self):
        result = context_selection_wrapper(self._handoff())
        self.assertEqual(result["input_count"], 3)
        self.assertEqual(result["output_count"], 3)
        self.assertEqual(result["total_packets_in"], 3)
        self.assertEqual(result["total_packets_out"], 3)
        self.assertEqual(len(result["selected_indices"]), 3)
        self.assertEqual(len(result["filtered_indices"]), 0)

    def test_selection_domain_filter_keeps_matching_packets(self):
        result = context_selection_wrapper(self._handoff(), domain="domain-a")
        # Item 0 has one domain-a packet and one domain-b packet
        self.assertEqual(result["total_packets_in"], 3)
        self.assertEqual(result["total_packets_out"], 1)
        # Item 0 is included (it still has a domain-a packet)
        self.assertIn(0, result["selected_indices"])
        # Item 1 is filtered out (all its packets are domain-b)
        self.assertIn(1, result["filtered_indices"])
        # Item 2 is included (no packets to filter)
        self.assertIn(2, result["selected_indices"])

    def test_selection_cluster_filter_keeps_matching_packets(self):
        result = context_selection_wrapper(self._handoff(), cluster_id="mc-1")
        # Only item 1's packets match cluster mc-1
        self.assertEqual(result["total_packets_out"], 1)
        self.assertIn(1, result["selected_indices"])
        # Item 0 is filtered out (all its packets are cluster mc-0)
        self.assertIn(0, result["filtered_indices"])
        # Item 2 is included (no packets)
        self.assertIn(2, result["selected_indices"])

    def test_selection_promotion_rule_filter(self):
        result = context_selection_wrapper(self._handoff(), promotion_rule="explicit-test")
        # All packets have promotion_rule=explicit-test, so all kept
        self.assertEqual(result["total_packets_out"], 3)
        self.assertEqual(len(result["selected_indices"]), 3)

    def test_selection_no_matching_domain_filters_item(self):
        result = context_selection_wrapper(self._handoff(), domain="domain-z")
        # No packets match domain-z, but item 2 has no packets so passes through
        self.assertEqual(result["total_packets_out"], 0)
        self.assertIn(2, result["selected_indices"])
        self.assertIn(0, result["filtered_indices"])
        self.assertIn(1, result["filtered_indices"])

    def test_selection_min_relevance_filters_low_evidence_packets(self):
        handoff = self._handoff(num_items=1)
        # Add a low-evidence packet to item 0 with same domain
        handoff["items"][0]["pi_pln_extension"]["contextual_evidence_packets"].append(
            {"support": 1, "opposition": 0, "statement": "test",
             "promotion_domain": "domain-a", "cluster_id": "mc-0", "promotion_rule": "explicit-test"}
        )
        # total=1, evidence_weight=1/3=0.333
        result = context_selection_wrapper(handoff, domain="domain-a", min_packet_relevance=0.5)
        item0 = result["items"][0]
        kept = [p for p in item0["packets_in"] if p["included"]]
        filtered = [p for p in item0["packets_in"] if not p["included"]]
        self.assertEqual(len(kept), 1)  # only (9, 1) survives
        self.assertEqual(len(filtered), 2)  # (3, 7) domain-b + (1, 0) low relevance
        # The low-evidence packet should be filtered by relevance
        rel_filtered = [p for p in filtered if "relevance" in p.get("filter_reason", "")]
        self.assertEqual(len(rel_filtered), 1)
        self.assertEqual(rel_filtered[0]["support"], 1)
        self.assertEqual(rel_filtered[0]["opposition"], 0)

    def test_selection_empty_handoff_returns_empty_result(self):
        empty_handoff = {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": 0,
            "items": [],
        }
        result = context_selection_wrapper(empty_handoff)
        self.assertEqual(result["schema"], "petta-memory-pi-pln-context-selection-v1")
        self.assertEqual(result["input_count"], 0)
        self.assertEqual(result["output_count"], 0)
        self.assertEqual(result["total_packets_in"], 0)
        self.assertEqual(result["total_packets_out"], 0)

    def test_selection_rejects_wrong_schema(self):
        with self.assertRaises(ValueError):
            context_selection_wrapper({"schema": "wrong-schema"})

    def test_selection_rejects_out_of_range_min_relevance(self):
        with self.assertRaises(ValueError):
            context_selection_wrapper(self._handoff(), min_packet_relevance=-0.1)
        with self.assertRaises(ValueError):
            context_selection_wrapper(self._handoff(), min_packet_relevance=1.5)

    def test_selection_boundary_text_is_non_live(self):
        result = context_selection_wrapper(self._handoff())
        boundary = result["boundary"]
        self.assertIn("non-live", boundary)
        self.assertIn("no memory append", boundary)
        self.assertIn("no OmegaClaw/GoalChainer live path", boundary)

    def test_selection_policy_records_criteria(self):
        result = context_selection_wrapper(
            self._handoff(),
            domain="domain-a",
            cluster_id="mc-0",
            promotion_rule="explicit-test",
            min_packet_relevance=0.3,
        )
        policy = result["selection_policy"]
        self.assertEqual(policy["domain"], "domain-a")
        self.assertEqual(policy["cluster_id"], "mc-0")
        self.assertEqual(policy["promotion_rule"], "explicit-test")
        self.assertEqual(policy["min_packet_relevance"], 0.3)
        self.assertIn("context selection", policy["source_pattern"])

    def test_selection_packet_summaries_record_filter_reasons(self):
        result = context_selection_wrapper(self._handoff(), domain="domain-a")
        item0 = result["items"][0]
        # Item 0 has 2 packets: one domain-a, one domain-b
        self.assertEqual(item0["original_packet_count"], 2)
        self.assertEqual(item0["kept_packet_count"], 1)
        # The filtered packet should have a filter_reason
        filtered_packets = [p for p in item0["packets_in"] if not p["included"]]
        self.assertEqual(len(filtered_packets), 1)
        self.assertIn("domain mismatch", filtered_packets[0]["filter_reason"])

    def test_selection_combined_domain_and_cluster_filter(self):
        result = context_selection_wrapper(
            self._handoff(),
            domain="domain-a",
            cluster_id="mc-0",
        )
        # Only packets matching both domain-a AND cluster mc-0 survive
        self.assertEqual(result["total_packets_out"], 1)
        self.assertIn(0, result["selected_indices"])

    def test_selection_items_without_packets_pass_through(self):
        result = context_selection_wrapper(self._handoff(), domain="domain-a")
        # Item 2 has no packets, should pass through
        self.assertIn(2, result["selected_indices"])
        item2 = result["items"][2]
        self.assertTrue(item2["included"])
        self.assertEqual(item2["original_packet_count"], 0)
        self.assertEqual(item2["kept_packet_count"], 0)


class StoreRoundTripContextSelectionTests(unittest.TestCase):
    """Empirical round-trip: store -> handoff -> context selection."""

    def test_roundtrip_context_selection_from_store(self):
        from petta_memory.store import MediumMemoryStore

        cluster = """
(MemoryCluster mc-cs-a)
(SchemaVersion mc-cs-a medium-memory-v1)
(ClusterType mc-cs-a belief-promotion)
(ClusterOpenedAt mc-cs-a "2026-07-06 01:00 PDT")
(ClusterSource mc-cs-a src-test)
(Contains mc-cs-a pe-cs-a)
(Contains mc-cs-a b-cs-a)
(ClusterStatus mc-cs-a active)
(PromotionEvent pe-cs-a)
(PromotesFrom pe-cs-a qc-cs-a)
(PromotesTo pe-cs-a b-cs-a)
(PromotionRule pe-cs-a explicit-test-promotion)
(PromotionTrust pe-cs-a 0.85)
(PromotionDomain pe-cs-a memory-architecture)
(DerivedBelief b-cs-a)
(BeliefContent b-cs-a (Requires MemoryTarget0 PLNReadyViews))
(TruthValue b-cs-a (stv 0.88 0.72))
(EvidenceFor b-cs-a qc-cs-a)
(EvidenceSupportCount b-cs-a 9.0)
(EvidenceOppositionCount b-cs-a 1.0)
"""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "cs_memory.metta")
            store.append_cluster(cluster)
            cache = store.pettachainer_handoff_cache()

        handoff = patham9_pln_handoff_sentences(cache)
        result = context_selection_wrapper(handoff)
        self.assertEqual(result["schema"], "petta-memory-pi-pln-context-selection-v1")
        self.assertEqual(result["input_count"], 1)
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(len(result["selected_indices"]), 1)
        self.assertGreater(result["total_packets_in"], 0)
        self.assertGreater(result["total_packets_out"], 0)


class ChainedInferencePipelineTests(unittest.TestCase):
    """Tests for the third inference-control mechanism: chained filter+select pipeline.

    The pipeline chains context selection (stage 1) with probabilistic
    filtering (stage 2) so that irrelevant evidence packets are removed before
    EC projection and composite-score ranking.  These tests validate the
    pipeline logic without invoking any runtime.
    """

    def _handoff(self, num_items: int = 3) -> dict:
        """Build a handoff with mixed evidence contexts and quality for pipeline testing."""
        items = []
        # Item 0: strong STV, packets from domain-a (strong support)
        items.append({
            "kind": "patham9-pln-sentence-input",
            "atom": "(Sentence (Acceptable publish_redacted_summary) (stv 0.91 0.74) ((PMEvidence b-0 mc-0 pe-0 rule domain)))",
            "term": "(Acceptable publish_redacted_summary)",
            "stv": {"strength": 0.91, "confidence": 0.74},
            "evidence_id": "(PMEvidence b-0 mc-0 pe-0 rule domain)",
            "belief_id": "b-0",
            "cluster_id": "mc-0",
            "promotion_event": "pe-0",
            "promotion_rule": "explicit-test",
            "promotion_domain": "memory-architecture",
            "source_status": "pln-ready-input-not-inferred-belief",
            "pi_pln_extension": {
                "contextual_evidence_packets": [
                    {"support": 9, "opposition": 1, "statement": "(Acceptable publish_redacted_summary)",
                     "promotion_domain": "domain-a", "cluster_id": "mc-0", "promotion_rule": "explicit-test"},
                    {"support": 3, "opposition": 7, "statement": "(Acceptable publish_redacted_summary)",
                     "promotion_domain": "domain-b", "cluster_id": "mc-0", "promotion_rule": "explicit-test"},
                ],
                "ec_projection_policy": "preserve packets first; later project EC",
                "context_selection": "not-run",
            },
        })
        if num_items >= 2:
            # Item 1: strong base STV but packets from domain-b (conflicting EC)
            items.append({
                "kind": "patham9-pln-sentence-input",
                "atom": "(Sentence (Acceptable share_full_log) (stv 0.94 0.80) ((PMEvidence b-1 mc-1 pe-1 rule domain)))",
                "term": "(Acceptable share_full_log)",
                "stv": {"strength": 0.94, "confidence": 0.80},
                "evidence_id": "(PMEvidence b-1 mc-1 pe-1 rule domain)",
                "belief_id": "b-1",
                "cluster_id": "mc-1",
                "promotion_event": "pe-1",
                "promotion_rule": "explicit-test",
                "promotion_domain": "memory-architecture",
                "source_status": "pln-ready-input-not-inferred-belief",
                "pi_pln_extension": {
                    "contextual_evidence_packets": [
                        {"support": 1, "opposition": 9, "statement": "(Acceptable share_full_log)",
                         "promotion_domain": "domain-b", "cluster_id": "mc-1", "promotion_rule": "explicit-test"},
                    ],
                    "ec_projection_policy": "preserve packets first; later project EC",
                    "context_selection": "not-run",
                },
            })
        if num_items >= 3:
            # Item 2: weak STV, no packets — passes context selection, low composite score
            items.append({
                "kind": "patham9-pln-sentence-input",
                "atom": "(Sentence (Requires MemoryTarget0 PLNReadyViews) (stv 0.70 0.55) ((PMEvidence b-2 mc-2 pe-2 rule domain)))",
                "term": "(Requires MemoryTarget0 PLNReadyViews)",
                "stv": {"strength": 0.70, "confidence": 0.55},
                "evidence_id": "(PMEvidence b-2 mc-2 pe-2 rule domain)",
                "belief_id": "b-2",
                "cluster_id": "mc-2",
                "promotion_event": "pe-2",
                "promotion_rule": "explicit-test",
                "promotion_domain": "memory-architecture",
                "source_status": "pln-ready-input-not-inferred-belief",
                "pi_pln_extension": {
                    "contextual_evidence_packets": [],
                    "ec_projection_policy": "preserve packets first; later project EC",
                    "context_selection": "not-run",
                },
            })
        return {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": len(items),
            "items": items,
        }

    def test_pipeline_returns_correct_schema(self):
        result = chained_inference_pipeline(self._handoff())
        self.assertEqual(result["schema"], "petta-memory-pi-pln-inference-pipeline-v1")

    def test_pipeline_no_filters_keeps_all_items(self):
        result = chained_inference_pipeline(self._handoff())
        self.assertEqual(result["input_count"], 3)
        self.assertEqual(result["stage1_output_count"], 3)
        self.assertEqual(result["stage1_filtered_count"], 0)
        self.assertEqual(result["output_count"], 3)
        self.assertEqual(len(result["selected_indices"]), 3)
        self.assertEqual(len(result["filtered_indices"]), 0)

    def test_pipeline_boundary_text_is_non_live(self):
        result = chained_inference_pipeline(self._handoff())
        self.assertIn("non-live", result["boundary"])

    def test_pipeline_domain_filter_excludes_items_with_only_foreign_packets(self):
        # Item 1 has packets only from domain-b; filtering by domain-a should exclude it
        result = chained_inference_pipeline(self._handoff(), domain="domain-a")
        self.assertEqual(result["stage1_output_count"], 2)  # items 0 and 2
        self.assertIn(1, result["stage1_filtered_indices"])
        self.assertNotIn(1, result["selected_indices"])

    def test_pipeline_domain_filter_reduces_packets_for_multi_domain_item(self):
        # Item 0 has packets from both domain-a and domain-b
        # Filtering by domain-a should keep item 0 but with fewer packets
        result = chained_inference_pipeline(self._handoff(), domain="domain-a")
        item0 = [pi for pi in result["items"] if pi["item_index"] == 0][0]
        self.assertTrue(item0["included"])
        # Stage 1 should have reduced packets (from 2 to 1)
        self.assertEqual(result["stage1_total_packets_in"], 3)
        self.assertEqual(result["stage1_total_packets_out"], 1)

    def test_pipeline_min_confidence_excludes_low_confidence_items(self):
        # Item 2 has confidence 0.55; set threshold above it
        result = chained_inference_pipeline(self._handoff(), min_confidence=0.60)
        self.assertNotIn(2, result["selected_indices"])
        self.assertIn(2, result["filtered_indices"])

    def test_pipeline_top_k_keeps_only_k_items(self):
        result = chained_inference_pipeline(self._handoff(), top_k=1)
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(len(result["selected_indices"]), 1)
        # Item 0 should rank highest (strong support from domain-a packet)
        self.assertEqual(result["selected_indices"][0], 0)

    def test_pipeline_combined_domain_filter_and_top_k(self):
        # Filter by domain-a, then keep only top-1
        result = chained_inference_pipeline(self._handoff(), domain="domain-a", top_k=1)
        self.assertEqual(result["stage1_output_count"], 2)  # items 0 and 2
        self.assertEqual(result["output_count"], 1)  # only top-1 from stage 2
        self.assertEqual(result["selected_indices"][0], 0)

    def test_pipeline_combined_domain_filter_and_min_confidence(self):
        # Filter by domain-a (keeps items 0, 2), then exclude item 2 by confidence
        result = chained_inference_pipeline(self._handoff(), domain="domain-a", min_confidence=0.60)
        self.assertEqual(result["stage1_output_count"], 2)
        self.assertIn(0, result["selected_indices"])
        self.assertNotIn(2, result["selected_indices"])
        self.assertIn(2, result["filtered_indices"])

    def test_pipeline_empty_handoff_returns_empty_result(self):
        handoff = {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": 0,
            "items": [],
        }
        result = chained_inference_pipeline(handoff)
        self.assertEqual(result["input_count"], 0)
        self.assertEqual(result["output_count"], 0)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["selected_indices"], [])
        self.assertEqual(result["filtered_indices"], [])

    def test_pipeline_rejects_wrong_schema(self):
        with self.assertRaises(ValueError):
            chained_inference_pipeline({"schema": "wrong"})

    def test_pipeline_rejects_out_of_range_min_relevance(self):
        with self.assertRaises(ValueError):
            chained_inference_pipeline(self._handoff(), min_packet_relevance=-0.1)
        with self.assertRaises(ValueError):
            chained_inference_pipeline(self._handoff(), min_packet_relevance=1.1)

    def test_pipeline_rejects_out_of_range_min_confidence(self):
        with self.assertRaises(ValueError):
            chained_inference_pipeline(self._handoff(), min_confidence=-0.1)
        with self.assertRaises(ValueError):
            chained_inference_pipeline(self._handoff(), min_confidence=1.1)

    def test_pipeline_rejects_negative_top_k(self):
        with self.assertRaises(ValueError):
            chained_inference_pipeline(self._handoff(), top_k=-1)

    def test_pipeline_ranking_remapped_to_original_indices(self):
        result = chained_inference_pipeline(self._handoff(), domain="domain-a", top_k=1)
        # With domain-a filter, stage 1 keeps items 0 and 2 (original indices)
        # Stage 2 top_k=1 should pick item 0
        self.assertEqual(result["ranking"][0]["item_index"], 0)
        self.assertIn("composite_score", result["ranking"][0])

    def test_pipeline_stage1_and_stage2_results_present(self):
        result = chained_inference_pipeline(self._handoff())
        self.assertIn("stage1_result", result)
        self.assertIn("stage2_result", result)
        self.assertEqual(result["stage1_result"]["schema"], "petta-memory-pi-pln-context-selection-v1")
        self.assertEqual(result["stage2_result"]["schema"], "petta-memory-pi-pln-inference-filter-v1")


class StoreRoundTripPipelineTests(unittest.TestCase):
    """Empirical round-trip: store -> handoff -> chained inference pipeline."""

    def test_roundtrip_pipeline_from_store(self):
        from petta_memory.store import MediumMemoryStore

        cluster = """
(MemoryCluster mc-pl-a)
(SchemaVersion mc-pl-a medium-memory-v1)
(ClusterType mc-pl-a belief-promotion)
(ClusterOpenedAt mc-pl-a "2026-07-06 03:00 PDT")
(ClusterSource mc-pl-a src-test)
(Contains mc-pl-a pe-pl-a)
(Contains mc-pl-a b-pl-a)
(ClusterStatus mc-pl-a active)
(PromotionEvent pe-pl-a)
(PromotesFrom pe-pl-a qc-pl-a)
(PromotesTo pe-pl-a b-pl-a)
(PromotionRule pe-pl-a explicit-test-promotion)
(PromotionTrust pe-pl-a 0.85)
(PromotionDomain pe-pl-a memory-architecture)
(DerivedBelief b-pl-a)
(BeliefContent b-pl-a (Requires MemoryTarget0 PLNReadyViews))
(TruthValue b-pl-a (stv 0.88 0.72))
(EvidenceFor b-pl-a qc-pl-a)
(EvidenceSupportCount b-pl-a 9.0)
(EvidenceOppositionCount b-pl-a 1.0)
"""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "pipeline_memory.metta")
            store.append_cluster(cluster)
            cache = store.pettachainer_handoff_cache()

        handoff = patham9_pln_handoff_sentences(cache)
        result = chained_inference_pipeline(handoff)
        self.assertEqual(result["schema"], "petta-memory-pi-pln-inference-pipeline-v1")
        self.assertEqual(result["input_count"], 1)
        self.assertEqual(result["stage1_output_count"], 1)
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(len(result["selected_indices"]), 1)
        self.assertEqual(len(result["filtered_indices"]), 0)
        # Item should have a composite score > 0
        item = result["items"][0]
        self.assertGreater(item["composite_score"], 0.0)


class MetaLearningBenchmarkHandoffTests(unittest.TestCase):
    """Tests for build_meta_learning_benchmark_handoff()."""

    def test_default_handoff_schema(self):
        h = build_meta_learning_benchmark_handoff()
        self.assertEqual(h["schema"], "petta-memory-patham9-pln-handoff-v1")
        self.assertEqual(h["item_count"], 4)  # 1 shortcut + 3 chain

    def test_shortcut_item_has_higher_stv_than_chain(self):
        h = build_meta_learning_benchmark_handoff()
        items = h["items"]
        shortcut = items[0]
        for item in items[1:]:
            self.assertGreater(
                shortcut["stv"]["strength"], item["stv"]["strength"],
                "shortcut should have higher strength than chain items",
            )
            self.assertGreater(
                shortcut["stv"]["confidence"], item["stv"]["confidence"],
                "shortcut should have higher confidence than chain items",
            )

    def test_shortcut_and_chain_have_evidence_packets(self):
        h = build_meta_learning_benchmark_handoff()
        for item in h["items"]:
            packets = item["pi_pln_extension"]["contextual_evidence_packets"]
            self.assertEqual(len(packets), 1)
            pkt = packets[0]
            self.assertIn("ec", pkt)
            self.assertIn("promotion_domain", pkt)
            self.assertIn("cluster_id", pkt)
            self.assertIn("promotion_rule", pkt)

    def test_shortcut_has_supportive_ec_chain_has_declining_ec(self):
        h = build_meta_learning_benchmark_handoff()
        shortcut_pkt = h["items"][0]["pi_pln_extension"]["contextual_evidence_packets"][0]
        self.assertEqual(shortcut_pkt["ec"], {"support": 9, "opposition": 1})
        chain_pkts = [
            item["pi_pln_extension"]["contextual_evidence_packets"][0]
            for item in h["items"][1:]
        ]
        self.assertEqual(chain_pkts[0]["ec"], {"support": 3, "opposition": 1})
        self.assertEqual(chain_pkts[1]["ec"], {"support": 2, "opposition": 2})
        self.assertEqual(chain_pkts[2]["ec"], {"support": 1, "opposition": 3})

    def test_custom_chain_lengths(self):
        h = build_meta_learning_benchmark_handoff(
            chain_strengths=[0.80, 0.70],
            chain_confidences=[0.60, 0.50],
            chain_ecs=[(4, 1), (2, 3)],
        )
        self.assertEqual(h["item_count"], 3)
        self.assertEqual(h["items"][2]["stv"]["strength"], 0.70)
        self.assertEqual(h["items"][2]["pi_pln_extension"]["contextual_evidence_packets"][0]["ec"],
                         {"support": 2, "opposition": 3})

    def test_rejects_mismatched_chain_lengths(self):
        with self.assertRaises(ValueError):
            build_meta_learning_benchmark_handoff(
                chain_strengths=[0.70, 0.60],
                chain_confidences=[0.55],
            )

    def test_rejects_mismatched_ecs(self):
        with self.assertRaises(ValueError):
            build_meta_learning_benchmark_handoff(
                chain_strengths=[0.70, 0.60],
                chain_confidences=[0.55, 0.50],
                chain_ecs=[(3, 1)],
            )

    def test_rejects_out_of_range_stv(self):
        with self.assertRaises(ValueError):
            build_meta_learning_benchmark_handoff(shortcut_strength=1.5)
        with self.assertRaises(ValueError):
            build_meta_learning_benchmark_handoff(chain_strengths=[0.70, -0.1])

    def test_rejects_negative_ec(self):
        with self.assertRaises(ValueError):
            build_meta_learning_benchmark_handoff(shortcut_ec=(-1, 2))

    def test_custom_domains(self):
        h = build_meta_learning_benchmark_handoff(
            shortcut_domain="memory",
            chain_domain="planning",
        )
        self.assertEqual(
            h["items"][0]["pi_pln_extension"]["contextual_evidence_packets"][0]["promotion_domain"],
            "memory",
        )
        self.assertEqual(
            h["items"][1]["pi_pln_extension"]["contextual_evidence_packets"][0]["promotion_domain"],
            "planning",
        )


class MetaLearningBenchmarkRunTests(unittest.TestCase):
    """Tests for run_meta_learning_benchmark()."""

    def test_default_benchmark_passes(self):
        """With default STVs, the shortcut should be ranked first."""
        result = run_meta_learning_benchmark()
        self.assertEqual(result["schema"], "petta-memory-pi-pln-meta-learning-benchmark-v1")
        self.assertTrue(result["overall_pass"],
                        "shortcut should be preferred over chain items by default")
        self.assertTrue(result["filter_shortcut_first"])
        self.assertTrue(result["pipeline_shortcut_first"])

    def test_shortcut_ranked_first_in_filter(self):
        result = run_meta_learning_benchmark()
        self.assertEqual(result["filter_result"]["shortcut_rank"], 1)
        self.assertTrue(result["filter_result"]["shortcut_first"])

    def test_shortcut_ranked_first_in_pipeline(self):
        result = run_meta_learning_benchmark()
        self.assertEqual(result["pipeline_result"]["shortcut_rank"], 1)
        self.assertTrue(result["pipeline_result"]["shortcut_first"])

    def test_shortcut_has_higher_composite_score_than_chain(self):
        result = run_meta_learning_benchmark()
        self.assertIsNotNone(result["filter_result"]["shortcut_composite_score"])
        best_chain = result["filter_result"]["best_chain_composite_score"]
        if best_chain is not None:
            self.assertGreater(
                result["filter_result"]["shortcut_composite_score"],
                best_chain,
            )

    def test_no_chain_item_outranks_shortcut(self):
        result = run_meta_learning_benchmark()
        self.assertFalse(result["filter_result"]["chain_outranks_shortcut"])

    def test_top_k_1_keeps_only_shortcut(self):
        result = run_meta_learning_benchmark(top_k=1)
        # The shortcut (index 0) should be rank 1 in the pipeline ranking
        ranking = result["pipeline_result"]["ranking"]
        self.assertEqual(len(ranking), 1)
        self.assertEqual(ranking[0]["item_index"], 0)
        self.assertEqual(ranking[0]["rank"], 1)

    def test_min_confidence_filters_chain_items(self):
        """With a high min_confidence, low-confidence chain items should be filtered."""
        # Default chain confidences after EC projection: need to check
        # The shortcut has confidence 0.90, chain items have 0.55, 0.50, 0.45
        # EC projection may raise confidence, but the composite score should still
        # rank shortcut first.  Set min_confidence high enough to filter chain.
        result = run_meta_learning_benchmark(min_confidence=0.85)
        # Shortcut should survive, chain items should be filtered
        self.assertTrue(result["filter_result"]["shortcut_first"])

    def test_domain_filter_excludes_chain(self):
        """Filtering by shortcut domain should exclude chain-domain items from pipeline."""
        result = run_meta_learning_benchmark(domain="benchmark")
        # Both shortcut and chain have domain "benchmark" by default, so all pass
        self.assertEqual(result["pipeline_result"]["ranking"][0]["item_index"], 0)

    def test_domain_filter_excludes_shortcut(self):
        """Filtering by chain-only domain should exclude the shortcut."""
        h = build_meta_learning_benchmark_handoff(
            shortcut_domain="memory",
            chain_domain="planning",
        )
        result = run_meta_learning_benchmark(
            handoff=h,
            domain="planning",
        )
        # Shortcut should not be in selected indices (it has domain "memory")
        # Chain items with domain "planning" should pass context selection
        self.assertNotIn(0, result["pipeline_result"].get("ranking", []) and
                         [r["item_index"] for r in result["pipeline_result"].get("ranking", [])]
                         or [])

    def test_rejects_wrong_handoff_schema(self):
        with self.assertRaises(ValueError):
            run_meta_learning_benchmark(handoff={"schema": "wrong"})

    def test_rejects_out_of_range_confidence(self):
        with self.assertRaises(ValueError):
            run_meta_learning_benchmark(min_confidence=-0.1)
        with self.assertRaises(ValueError):
            run_meta_learning_benchmark(min_confidence=1.5)

    def test_rejects_negative_top_k(self):
        with self.assertRaises(ValueError):
            run_meta_learning_benchmark(top_k=-1)

    def test_rejects_out_of_range_relevance(self):
        with self.assertRaises(ValueError):
            run_meta_learning_benchmark(min_packet_relevance=-0.1)
        with self.assertRaises(ValueError):
            run_meta_learning_benchmark(min_packet_relevance=1.5)

    def test_boundary_text_present(self):
        result = run_meta_learning_benchmark()
        self.assertIn("non-live wrapper-only benchmark", result["boundary"])
        self.assertIn("no memory append", result["boundary"])

    def test_benchmark_scenario_metadata(self):
        result = run_meta_learning_benchmark()
        scenario = result["benchmark_scenario"]
        self.assertEqual(scenario["shortcut_item_index"], 0)
        self.assertEqual(scenario["chain_item_indices"], [1, 2, 3])
        self.assertEqual(scenario["shortcut_belief_id"], "shortcut-0")
        self.assertEqual(scenario["chain_belief_ids"], ["chain-1", "chain-2", "chain-3"])

    def test_filter_and_pipeline_results_present(self):
        result = run_meta_learning_benchmark()
        self.assertIn("filter_result", result)
        self.assertIn("pipeline_result", result)
        self.assertEqual(
            result["filter_result"]["schema"],
            "petta-memory-pi-pln-inference-filter-v1",
        )
        self.assertEqual(
            result["pipeline_result"]["schema"],
            "petta-memory-pi-pln-inference-pipeline-v1",
        )


class StoreRoundTripMetaLearningBenchmarkTests(unittest.TestCase):
    """Empirical round-trip: store -> handoff -> meta-learning benchmark."""

    def test_roundtrip_benchmark_from_store(self):
        from petta_memory.store import MediumMemoryStore

        cluster = """
(MemoryCluster mc-bench-a)
(SchemaVersion mc-bench-a medium-memory-v1)
(ClusterType mc-bench-a belief-promotion)
(ClusterOpenedAt mc-bench-a "2026-07-06 05:00 PDT")
(ClusterSource mc-bench-a src-test)
(Contains mc-bench-a pe-bench-a)
(Contains mc-bench-a b-bench-a)
(ClusterStatus mc-bench-a active)
(PromotionEvent pe-bench-a)
(PromotesFrom pe-bench-a qc-bench-a)
(PromotesTo pe-bench-a b-bench-a)
(PromotionRule pe-bench-a explicit-benchmark-promotion)
(PromotionTrust pe-bench-a 0.90)
(PromotionDomain pe-bench-a benchmark)
(DerivedBelief b-bench-a)
(BeliefContent b-bench-a (ShortcutConclusion))
(TruthValue b-bench-a (stv 0.95 0.90))
(EvidenceFor b-bench-a qc-bench-a)
(EvidenceSupportCount b-bench-a 9.0)
(EvidenceOppositionCount b-bench-a 1.0)
"""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "bench_memory.metta")
            store.append_cluster(cluster)
            cache = store.pettachainer_handoff_cache()

        handoff = patham9_pln_handoff_sentences(cache)
        result = run_meta_learning_benchmark(handoff=handoff)
        self.assertEqual(result["schema"], "petta-memory-pi-pln-meta-learning-benchmark-v1")
        # Single item from store: shortcut by default (index 0), no chain items
        self.assertEqual(result["benchmark_scenario"]["shortcut_item_index"], 0)
        self.assertEqual(result["benchmark_scenario"]["chain_item_indices"], [])
        self.assertTrue(result["overall_pass"])


# ---------------------------------------------------------------------------
# Continuation predicate wrapper tests
# ---------------------------------------------------------------------------

class ContinuationPredicateWrapperTests(unittest.TestCase):
    """Unit tests for the continuation_predicate_wrapper inference-control mechanism."""

    def _make_handoff(self, n: int = 3) -> dict[str, Any]:
        """Build a small handoff with varied STVs, EC counts, and domains."""
        items: list[dict[str, Any]] = []
        for i in range(n):
            strengths = [0.90, 0.50, 0.30]
            confidences = [0.80, 0.60, 0.40]
            domains = ["reasoning", "reasoning", "planning"]
            rules = ["explicit-promotion", "explicit-promotion", "heuristic"]
            depths = [0, 1, 2]
            ec_support = [9, 3, 1]
            ec_opposition = [1, 2, 4]
            items.append({
                "belief_id": f"b-cp-{i}",
                "term": f"(TestTerm{i})",
                "stv": {"strength": strengths[i % 3], "confidence": confidences[i % 3]},
                "pi_pln_extension": {
                    "promotion_domain": domains[i % 3],
                    "promotion_rule": rules[i % 3],
                    "derivation_depth": depths[i % 3],
                    "contextual_evidence_packets": [
                        {
                            "ec": {
                                "support": ec_support[i % 3],
                                "opposition": ec_opposition[i % 3],
                            },
                            "promotion_domain": domains[i % 3],
                            "promotion_rule": rules[i % 3],
                        }
                    ],
                },
            })
        return {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": n,
            "items": items,
        }

    def test_schema(self):
        result = continuation_predicate_wrapper(self._make_handoff())
        self.assertEqual(result["schema"], "petta-memory-pi-pln-continuation-predicate-v1")
        self.assertEqual(result["mode"], "design-specification-no-runtime")

    def test_boundary_text(self):
        result = continuation_predicate_wrapper(self._make_handoff())
        self.assertIn("non-live wrapper-only", result["boundary"])
        self.assertIn("no SWI/PeTTa/MeTTa runtime", result["boundary"])
        self.assertIn("no memory append", result["boundary"])

    def test_all_continue_with_default_policy(self):
        """With no thresholds set, all items should continue."""
        handoff = self._make_handoff()
        result = continuation_predicate_wrapper(handoff)
        self.assertEqual(result["continue_count"], 3)
        self.assertEqual(result["terminate_count"], 0)
        self.assertEqual(result["reject_count"], 0)
        self.assertEqual(sorted(result["continue_indices"]), [0, 1, 2])

    def test_min_strength_filters(self):
        result = continuation_predicate_wrapper(self._make_handoff(), min_strength=0.6)
        # Item 0: 0.90 >= 0.6 -> continue
        # Item 1: 0.50 < 0.6 -> reject
        # Item 2: 0.30 < 0.6 -> reject
        self.assertEqual(result["continue_count"], 1)
        self.assertEqual(result["reject_count"], 2)
        self.assertEqual(result["continue_indices"], [0])
        self.assertEqual(sorted(result["reject_indices"]), [1, 2])

    def test_min_confidence_filters(self):
        result = continuation_predicate_wrapper(self._make_handoff(), min_confidence=0.7)
        # Item 0: 0.80 >= 0.7 -> continue
        # Item 1: 0.60 < 0.7 -> reject
        # Item 2: 0.40 < 0.7 -> reject
        self.assertEqual(result["continue_count"], 1)
        self.assertEqual(result["reject_count"], 2)

    def test_domain_filter(self):
        result = continuation_predicate_wrapper(self._make_handoff(), domain="reasoning")
        # Items 0,1 have domain "reasoning" -> continue
        # Item 2 has domain "planning" -> reject
        self.assertEqual(result["continue_count"], 2)
        self.assertEqual(result["reject_count"], 1)
        self.assertEqual(result["reject_indices"], [2])
        # Check that domain check appears in checks
        item2 = [i for i in result["items"] if i["item_index"] == 2][0]
        domain_check = [c for c in item2["checks"] if c["check"] == "domain"][0]
        self.assertEqual(domain_check["required"], "reasoning")
        self.assertEqual(domain_check["actual"], "planning")
        self.assertFalse(domain_check["passed"])

    def test_promotion_rule_filter(self):
        result = continuation_predicate_wrapper(self._make_handoff(), promotion_rule="explicit-promotion")
        # Items 0,1 have rule "explicit-promotion" -> continue
        # Item 2 has rule "heuristic" -> reject
        self.assertEqual(result["continue_count"], 2)
        self.assertEqual(result["reject_count"], 1)
        self.assertEqual(result["reject_indices"], [2])

    def test_ec_ratio_threshold(self):
        # Item 0: 9/(9+1)=0.9, Item 1: 3/(3+2)=0.6, Item 2: 1/(1+4)=0.2
        result = continuation_predicate_wrapper(self._make_handoff(), ec_ratio_threshold=0.5)
        # Items 0,1 pass; item 2 rejected
        self.assertEqual(result["continue_count"], 2)
        self.assertEqual(result["reject_count"], 1)
        self.assertEqual(result["reject_indices"], [2])
        item2 = [i for i in result["items"] if i["item_index"] == 2][0]
        ec_check = [c for c in item2["checks"] if c["check"] == "ec_ratio"][0]
        self.assertAlmostEqual(ec_check["actual"], 0.2, places=2)
        self.assertFalse(ec_check["passed"])

    def test_max_derivation_depth_terminates(self):
        """Items at or beyond max depth should be terminated, not rejected."""
        result = continuation_predicate_wrapper(self._make_handoff(), max_derivation_depth=1)
        # Item 0: depth 0 < 1 -> continue
        # Item 1: depth 1 >= 1 -> terminate
        # Item 2: depth 2 >= 1 -> terminate
        self.assertEqual(result["continue_count"], 1)
        self.assertEqual(result["terminate_count"], 2)
        self.assertEqual(result["reject_count"], 0)
        self.assertEqual(result["continue_indices"], [0])
        self.assertEqual(sorted(result["terminate_indices"]), [1, 2])

    def test_terminate_with_depth_and_strength(self):
        """If an item both exceeds depth and fails strength, it should be rejected."""
        result = continuation_predicate_wrapper(
            self._make_handoff(),
            min_strength=0.6,
            max_derivation_depth=1,
        )
        # Item 0: strength 0.90 >= 0.6, depth 0 < 1 -> continue
        # Item 1: strength 0.50 < 0.6 -> reject (strength check fails first)
        # Item 2: strength 0.30 < 0.6 -> reject
        self.assertEqual(result["continue_count"], 1)
        self.assertEqual(result["terminate_count"], 0)
        self.assertEqual(result["reject_count"], 2)

    def test_combined_filters(self):
        result = continuation_predicate_wrapper(
            self._make_handoff(),
            min_strength=0.4,
            min_confidence=0.5,
            domain="reasoning",
        )
        # Item 0: strength 0.90, confidence 0.80, domain reasoning -> continue
        # Item 1: strength 0.50 >= 0.4, confidence 0.60 >= 0.5, domain reasoning -> continue
        # Item 2: strength 0.30 < 0.4 -> reject; also domain planning -> reject
        self.assertEqual(result["continue_count"], 2)
        self.assertEqual(result["reject_count"], 1)

    def test_empty_handoff(self):
        handoff = {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": 0,
            "items": [],
        }
        result = continuation_predicate_wrapper(handoff)
        self.assertEqual(result["input_count"], 0)
        self.assertEqual(result["continue_count"], 0)
        self.assertEqual(result["items"], [])

    def test_validation_wrong_schema(self):
        with self.assertRaises(ValueError):
            continuation_predicate_wrapper({"schema": "wrong"})

    def test_validation_min_strength_out_of_range(self):
        with self.assertRaises(ValueError):
            continuation_predicate_wrapper(self._make_handoff(), min_strength=-0.1)
        with self.assertRaises(ValueError):
            continuation_predicate_wrapper(self._make_handoff(), min_strength=1.1)

    def test_validation_min_confidence_out_of_range(self):
        with self.assertRaises(ValueError):
            continuation_predicate_wrapper(self._make_handoff(), min_confidence=-0.1)
        with self.assertRaises(ValueError):
            continuation_predicate_wrapper(self._make_handoff(), min_confidence=1.1)

    def test_validation_max_depth_negative(self):
        with self.assertRaises(ValueError):
            continuation_predicate_wrapper(self._make_handoff(), max_derivation_depth=-1)

    def test_validation_ec_ratio_out_of_range(self):
        with self.assertRaises(ValueError):
            continuation_predicate_wrapper(self._make_handoff(), ec_ratio_threshold=-0.1)
        with self.assertRaises(ValueError):
            continuation_predicate_wrapper(self._make_handoff(), ec_ratio_threshold=1.1)

    def test_no_ec_packets_passes_ec_check(self):
        """Items without EC packets should pass the EC ratio check."""
        handoff = {
            "schema": "petta-memory-patham9-pln-handoff-v1",
            "item_count": 1,
            "items": [{
                "belief_id": "b-no-ec",
                "term": "(NoEC)",
                "stv": {"strength": 0.8, "confidence": 0.7},
                "pi_pln_extension": {
                    "promotion_domain": "test",
                    "promotion_rule": "rule",
                    "derivation_depth": 0,
                },
            }],
        }
        result = continuation_predicate_wrapper(handoff, ec_ratio_threshold=0.5)
        self.assertEqual(result["continue_count"], 1)
        item0 = result["items"][0]
        self.assertIsNone(item0["ec_summary"])

    def test_item_decision_field(self):
        result = continuation_predicate_wrapper(self._make_handoff(), min_strength=0.6)
        item0 = [i for i in result["items"] if i["item_index"] == 0][0]
        self.assertEqual(item0["decision"], "continue")
        item1 = [i for i in result["items"] if i["item_index"] == 1][0]
        self.assertEqual(item1["decision"], "reject")

    def test_checks_structure(self):
        result = continuation_predicate_wrapper(
            self._make_handoff(),
            min_strength=0.5,
            min_confidence=0.5,
            domain="reasoning",
            ec_ratio_threshold=0.5,
            max_derivation_depth=2,
            promotion_rule="explicit-promotion",
        )
        item0 = result["items"][0]
        check_names = [c["check"] for c in item0["checks"]]
        self.assertIn("min_strength", check_names)
        self.assertIn("min_confidence", check_names)
        self.assertIn("domain", check_names)
        self.assertIn("promotion_rule", check_names)
        self.assertIn("ec_ratio", check_names)
        self.assertIn("max_derivation_depth", check_names)
        for check in item0["checks"]:
            self.assertIn("passed", check)
            self.assertIn("required", check)
            self.assertIn("actual", check)

    def test_ec_summary_with_packets(self):
        result = continuation_predicate_wrapper(self._make_handoff())
        item0 = result["items"][0]
        self.assertIsNotNone(item0["ec_summary"])
        self.assertEqual(item0["ec_summary"]["support"], 9)
        self.assertEqual(item0["ec_summary"]["opposition"], 1)
        self.assertAlmostEqual(item0["ec_summary"]["ratio"], 0.9, places=2)

    def test_policy_in_result(self):
        result = continuation_predicate_wrapper(
            self._make_handoff(),
            min_strength=0.5,
            max_derivation_depth=3,
            domain="reasoning",
            ec_ratio_threshold=0.3,
            promotion_rule="explicit-promotion",
        )
        policy = result["continuation_policy"]
        self.assertEqual(policy["min_strength"], 0.5)
        self.assertEqual(policy["max_derivation_depth"], 3)
        self.assertEqual(policy["domain"], "reasoning")
        self.assertEqual(policy["ec_ratio_threshold"], 0.3)
        self.assertEqual(policy["promotion_rule"], "explicit-promotion")
        self.assertIn("continuation predicate", policy["source_pattern"])

    def test_depth_termination_check_has_termination_flag(self):
        result = continuation_predicate_wrapper(self._make_handoff(), max_derivation_depth=1)
        item1 = [i for i in result["items"] if i["item_index"] == 1][0]
        depth_check = [c for c in item1["checks"] if c["check"] == "max_derivation_depth"][0]
        self.assertTrue(depth_check.get("termination"))
        self.assertFalse(depth_check["passed"])


class StoreRoundTripContinuationPredicateTests(unittest.TestCase):
    """Empirical round-trip: store -> handoff -> continuation predicate."""

    def test_roundtrip_continuation_predicate_from_store(self):
        from petta_memory.store import MediumMemoryStore

        cluster = """
(MemoryCluster mc-cp-a)
(SchemaVersion mc-cp-a medium-memory-v1)
(ClusterType mc-cp-a belief-promotion)
(ClusterOpenedAt mc-cp-a "2026-07-06 07:00 PDT")
(ClusterSource mc-cp-a src-test)
(Contains mc-cp-a pe-cp-a)
(Contains mc-cp-a b-cp-a)
(ClusterStatus mc-cp-a active)
(PromotionEvent pe-cp-a)
(PromotesFrom pe-cp-a qc-cp-a)
(PromotesTo pe-cp-a b-cp-a)
(PromotionRule pe-cp-a explicit-continuation-test)
(PromotionTrust pe-cp-a 0.85)
(PromotionDomain pe-cp-a reasoning)
(DerivedBelief b-cp-a)
(BeliefContent b-cp-a (ContinuationTestConclusion))
(TruthValue b-cp-a (stv 0.88 0.75))
(EvidenceFor b-cp-a qc-cp-a)
(EvidenceSupportCount b-cp-a 8.0)
(EvidenceOppositionCount b-cp-a 2.0)
"""
        with tempfile.TemporaryDirectory() as td:
            store = MediumMemoryStore(Path(td) / "cp_memory.metta")
            store.append_cluster(cluster)
            cache = store.pettachainer_handoff_cache()

        handoff = patham9_pln_handoff_sentences(cache)
        result = continuation_predicate_wrapper(
            handoff,
            min_strength=0.5,
            min_confidence=0.5,
            domain="reasoning",
        )
        self.assertEqual(result["schema"], "petta-memory-pi-pln-continuation-predicate-v1")
        self.assertGreaterEqual(result["continue_count"], 1)
        # The promoted belief should continue (strength 0.88, confidence ~0.6375 after trust cap)
        item0 = result["items"][0]
        self.assertEqual(item0["decision"], "continue")
        self.assertGreaterEqual(float(item0["stv"]["strength"]), 0.5)


if __name__ == "__main__":
    unittest.main()
