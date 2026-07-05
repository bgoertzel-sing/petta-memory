import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.patham9_pln import (
    classify_smoke_result,
    classify_smoke_result_with_retry,
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


if __name__ == "__main__":
    unittest.main()
