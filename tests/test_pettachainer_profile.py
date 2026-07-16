import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import petta_memory.pettachainer_profile as profile
from petta_memory.pipln_models import PeTTaChainerEpisodeContract, PeTTaChainerInputStatement
from petta_memory.pettachainer_profile import _run_isolated_stage, build_profile_store, build_promoted_cluster


def _slow_profile_stage() -> dict[str, object]:
    time.sleep(5)
    return {"result": "too late"}


def _echo_profile_stage(value: str) -> dict[str, object]:
    print("noisy runtime output")
    return {"result": value}


class PeTTaChainerProfileWorkloadTests(unittest.TestCase):
    @staticmethod
    def episode_contract():
        digest = "a" * 64
        statement = PeTTaChainerInputStatement(
            atom=f"(: pm-{digest} (S a) (STV 0.75 0.8))",
            proof_id=f"pm-{digest}",
            sentence_digest=digest,
            canonical_term="(S a)",
            strength=0.75,
            confidence=0.8,
            stamp_ints=(0,),
            evidence_basis_ids=("basis-1",),
        )
        return PeTTaChainerEpisodeContract(
            episode_id="episode-probe",
            chart_fingerprint="b" * 64,
            statements=(statement,),
            query_term="(S a)",
            query_atom="(: $prf (S a) $tv)",
        )

    def test_episode_contract_probe_runs_runtime_only_after_exact_validation(self):
        events = [
            {"label": "validate_episode_contract", "status": "ok", "statement_results": [1.0], "query_result": 1.0},
            {"label": "compileadd_and_query_episode_contract", "status": "timeout", "timeout_sec": 2.0},
        ]
        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=events) as isolated,
        ):
            result = profile.probe_pettachainer_episode_contract(
                self.episode_contract(), project_root=Path("/unused"), stage_timeout_sec=2.0,
            )

        self.assertTrue(result["validators_admitted"])
        self.assertFalse(result["runtime_admitted"])
        self.assertEqual(isolated.call_count, 2)
        self.assertTrue(result["boundaries"]["no_result_claim_unless_runtime_admitted"])

    def test_episode_contract_probe_fails_closed_before_runtime_on_validator_drift(self):
        validation = {
            "label": "validate_episode_contract", "status": "ok",
            "statement_results": [True], "query_result": 1.0,
        }
        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", return_value=validation) as isolated,
        ):
            result = profile.probe_pettachainer_episode_contract(
                self.episode_contract(), project_root=Path("/unused"),
            )

        self.assertFalse(result["validators_admitted"])
        self.assertFalse(result["runtime_admitted"])
        self.assertEqual(isolated.call_count, 1)

    def test_episode_contract_probe_does_not_admit_empty_query_result(self):
        events = [
            {"status": "ok", "statement_results": [1.0], "query_result": 1.0},
            {
                "status": "ok",
                "stages": [
                    {"status": "ok", "result": "added"},
                    {"status": "ok", "result": []},
                ],
            },
        ]
        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=events),
        ):
            result = profile.probe_pettachainer_episode_contract(
                self.episode_contract(), project_root=Path("/unused"),
            )

        self.assertTrue(result["validators_admitted"])
        self.assertFalse(result["runtime_admitted"])

    def test_build_promoted_cluster_exports_statement_and_packet(self):
        with tempfile.TemporaryDirectory() as td:
            store = build_profile_store(Path(td) / "medium_memory.metta", 2)
            statements = store.pettachainer_evidence_view().splitlines()
            packets = store.pettachainer_evidence_packet_view().splitlines()

        self.assertEqual(len(statements), 2)
        self.assertEqual(len(packets), 2)
        self.assertIn("(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))", statements)
        self.assertIn(
            "(EvidencePacket (Requires MemoryTarget0 PLNReadyViews) (EC 3.0 1.0) "
            "((domain omegaclaw-memory) (promotion-rule explicit-profile-workload)) pe-profile-000)",
            packets,
        )

    def test_negative_profile_workload_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                build_profile_store(Path(td) / "medium_memory.metta", -1)

    def test_generated_profile_cluster_uses_explicit_counts(self):
        cluster = build_promoted_cluster(4, support=10.0, opposition=2.0)
        self.assertIn("(EvidenceSupportCount b-profile-004 14.0)", cluster)
        self.assertIn("(EvidenceOppositionCount b-profile-004 3.0)", cluster)
        self.assertIn("(PromotionDomain pe-profile-004 omegaclaw-memory)", cluster)

    def test_isolated_stage_captures_output_and_result(self):
        event = _run_isolated_stage("echo", _echo_profile_stage, ("ok",), stage_timeout_sec=2.0)

        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["label"], "echo")
        self.assertEqual(event["result"], "ok")
        self.assertGreater(event["stdout_chars"], 0)

    def test_isolated_stage_timeout_returns_bounded_event(self):
        event = _run_isolated_stage("slow", _slow_profile_stage, (), stage_timeout_sec=0.05)

        self.assertEqual(event["status"], "timeout")
        self.assertEqual(event["label"], "slow")
        self.assertEqual(event["timeout_sec"], 0.05)

    def test_compileadd_probe_call_text_distinguishes_direct_from_eval_control(self):
        statement = "(: p (S x) (STV 1 0.9))"

        direct = profile._compileadd_probe_call_text(statement, "kb", "materialize_stmt_lambdas", "direct")
        eval_control = profile._compileadd_probe_call_text(statement, "kb", "materialize_stmt_lambdas", "eval")

        self.assertEqual(direct, "!(materialize-stmt-lambdas (: p (S x) (STV 1 0.9)))")
        self.assertEqual(eval_control, "!(eval (materialize-stmt-lambdas (: p (S x) (STV 1 0.9))))")

    def test_materialize_identity_match_allows_numeric_rendering_changes(self):
        self.assertTrue(profile._materialize_identity_matches("(STV 0.70 0.55)", ["(STV 0.7 0.55)"]))
        self.assertTrue(
            profile._materialize_identity_matches(
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
                ["(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.7 0.55))"],
            )
        )
        self.assertFalse(profile._materialize_identity_matches("(STV 0.70 0.55)", ["(STV 0.6 0.55)"]))

    def test_materialize_output_summary_bounds_duplicate_fanout(self):
        statement = "(: p (S x) (STV 1 0.9))"
        outputs = [statement] * 20

        summary = profile._summarize_materialize_outputs(
            statement,
            outputs,
            max_output_items=3,
        )

        self.assertTrue(summary["identity_output_present"])
        self.assertEqual(summary["output_count"], 20)
        self.assertEqual(summary["unique_output_count"], 1)
        self.assertEqual(summary["output_items"], [statement] * 3)
        self.assertTrue(summary["output_truncated"])

    def test_materialize_output_summary_rejects_nonpositive_bound(self):
        with self.assertRaises(ValueError):
            profile._summarize_materialize_outputs("(S x)", [], max_output_items=0)

    def test_inspect_pettachainer_add_api_reports_no_precompiled_api(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            package = repo / "pettachainer"
            metta_dir = package / "metta"
            metta_dir.mkdir(parents=True)
            (package / "pettachainer.py").write_text(
                "class PeTTaChainer:\n"
                "    def add_atom(self, atom):\n"
                "        return self.handler.process_metta_string(f'!(compileadd {atom})')\n"
                "    def add_atoms_no_check(self, atoms):\n"
                "        return self.handler.process_metta_string('!(superpose ((compileadd kb a)))')\n"
                "    def query(self, atom):\n"
                "        return []\n",
                encoding="utf-8",
            )
            (metta_dir / "petta_chainer.metta").write_text(
                "(= (compileadd $kb $stmt)\n"
                "  (let* (($stmt1 (materialize-stmt-lambdas $stmt))\n"
                "         ($atoms (collapse (mm2compile $kb $stmt1)))\n"
                "         ($_index (index-source-implication $kb $stmt1))\n"
                "         ($_ (maybe-process-on-add $kb $stmt1)))\n"
                "    $atoms))\n",
                encoding="utf-8",
            )

            summary = profile.inspect_pettachainer_add_api(repo)

        self.assertEqual(summary["public_add_methods"], ["add_atom", "add_atoms_no_check"])
        self.assertFalse(summary["exposes_precompiled_add_api"])
        self.assertEqual(summary["compileadd_definitions"], ["compileadd"])
        self.assertEqual(summary["add_method_compile_calls"]["add_atom"], ["compileadd"])
        self.assertIn("no public precompiled-add API found", summary["recommended_boundary"])

    def test_inspect_materialize_stmt_lambdas_for_statement_marks_lambda_free_identity(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            metta_dir = repo / "pettachainer" / "metta"
            metta_dir.mkdir(parents=True)
            (metta_dir / "petta_chainer.metta").write_text(
                "(= (materialize-stmt-lambdas $term)\n"
                "   (if (is-var $term) $term\n"
                "      (if (is-expr $term)\n"
                "         (if (== (car-atom $term) |->) (eval $term)\n"
                "            (cons (materialize-stmt-lambdas (car-atom $term))\n"
                "               (map-flat materialize-stmt-lambdas (cdr-atom $term))))\n"
                "         $term)))\n",
                encoding="utf-8",
            )

            summary = profile.inspect_materialize_stmt_lambdas_for_statement(
                repo,
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
            )

        self.assertTrue(summary["materialize_expected_identity"])
        self.assertEqual(summary["statement_stats"]["lambda_form_count"], 0)
        self.assertGreater(summary["statement_stats"]["expression_nodes"], 1)
        self.assertIn("eval", summary["definition"]["calls"])
        self.assertIn("runtime success", " ".join(summary["gates"]))
        self.assertEqual(
            summary["next_probe"]["kind"],
            "non-live materialize identity runtime gate",
        )

    def test_run_materialize_identity_gate_uses_source_check_and_isolated_stage(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_inspection(repo, checked_statement):
            captured["repo"] = repo
            captured["checked_statement"] = checked_statement
            return {
                "materialize_expected_identity": True,
                "statement_stats": {"lambda_form_count": 0},
            }

        def fake_isolated_stage(label, target, args, *, stage_timeout_sec):
            captured["label"] = label
            captured["target"] = target
            captured["args"] = args
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "label": label,
                "status": "ok",
                "identity_output_present": True,
                "expected_statement": statement,
            }

        with (
            patch.object(profile, "inspect_materialize_stmt_lambdas_for_statement", side_effect=fake_inspection),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_isolated_stage),
        ):
            result = profile.run_materialize_identity_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=4.0,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(captured["repo"], Path("/project/repos/PeTTaChainer"))
        self.assertEqual(captured["checked_statement"], statement)
        self.assertEqual(captured["label"], "materialize_stmt_lambdas_identity")
        self.assertEqual(captured["target"], profile._materialize_identity_stage)
        self.assertEqual(captured["args"], (statement,))
        self.assertEqual(captured["stage_timeout_sec"], 4.0)
        self.assertIn("no mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_run_materialize_identity_gate_skips_lambda_statements(self):
        with patch.object(
            profile,
            "inspect_materialize_stmt_lambdas_for_statement",
            return_value={"materialize_expected_identity": False},
        ), patch.object(profile, "_configure_local_runtime") as configure:
            result = profile.run_materialize_identity_gate(
                "(: p (|-> x x) (STV 1 0.9))",
                project_root=Path("/project"),
                stage_timeout_sec=4.0,
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()
        self.assertIn("statement contains |-> lambda forms", result["reason"])

    def test_run_materialize_identity_ladder_stops_at_first_blocked_rung(self):
        rungs = ["(Requires MemoryTarget0 PLNReadyViews)", "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"]
        seen = []

        def fake_inspection(repo, statement):
            return {"repo_path": str(repo), "statement": statement, "materialize_expected_identity": True}

        def fake_event(statement, *, stage_timeout_sec):
            seen.append((statement, stage_timeout_sec))
            if statement == rungs[0]:
                return {"label": "materialize_stmt_lambdas_identity", "status": "ok", "identity_output_present": True}
            return {"label": "materialize_stmt_lambdas_identity", "status": "timeout", "timeout_sec": stage_timeout_sec}

        with (
            patch.object(profile, "inspect_materialize_stmt_lambdas_for_statement", side_effect=fake_inspection),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_materialize_identity_event", side_effect=fake_event),
        ):
            result = profile.run_materialize_identity_ladder_gate(
                rungs,
                project_root=Path("/project"),
                stage_timeout_sec=2.5,
            )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["first_blocked_rung"], 1)
        self.assertEqual(result["rung_count_executed"], 2)
        self.assertEqual(seen, [(rungs[0], 2.5), (rungs[1], 2.5)])
        self.assertIn("Stop at the first blocked rung", " ".join(result["gates"]))

    def test_materialize_identity_ladder_skips_if_any_rung_has_lambda(self):
        with patch.object(
            profile,
            "inspect_materialize_stmt_lambdas_for_statement",
            side_effect=[
                {"statement": "(Requires MemoryTarget0 PLNReadyViews)", "materialize_expected_identity": True},
                {"statement": "(|-> x x)", "materialize_expected_identity": False},
            ],
        ), patch.object(profile, "_configure_local_runtime") as configure:
            result = profile.run_materialize_identity_ladder_gate(
                ["(Requires MemoryTarget0 PLNReadyViews)", "(|-> x x)"],
                project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()
        self.assertEqual(result["skipped_statements"], ["(|-> x x)"])

    def test_materialize_identity_ladder_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            profile.run_materialize_identity_ladder_gate([], project_root=Path("/project"))

    def test_materialize_proof_shape_rungs_include_prefixes_before_full_proof(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_identity_proof_shape_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(Requires MemoryTarget0 PLNReadyViews)",
                "(STV 0.70 0.55)",
                "(: b-profile-000)",
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews))",
                "(: b-profile-000 ProofShapeSentinel (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 1.0 1.0))",
                "(: b-profile-000 ProofShapeSentinel (STV 0.70 0.55))",
                statement,
            ],
        )

    def test_materialize_nested_type_proof_rungs_rebuild_type_under_sentinel_tv(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_nested_type_proof_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(: b-profile-000 Requires (STV 1.0 1.0))",
                "(: b-profile-000 (Requires) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 PLNReadyViews) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_nested_type_ladder_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 3,
                "rung_count_executed": 4,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_nested_type_ladder_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas nested-type proof ladder gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["nested_type_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_nested_type_rungs_reject_atom_type(self):
        with self.assertRaises(ValueError):
            profile.materialize_nested_type_proof_rungs("(: p PlainType (STV 1 1))")

    def test_materialize_nested_type_arity_matrix_rungs_test_sentinel_arity_before_original(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_nested_type_arity_matrix_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(: b-profile-000 (Requires) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 PLNReadyViews) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_nested_type_arity_matrix_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 2,
                "rung_count_executed": 3,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_nested_type_arity_matrix_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas nested-type arity matrix gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["nested_type_arity_matrix_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_nested_type_context_matrix_rungs_move_sentinel_type_through_contexts(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_nested_type_context_matrix_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(Requires TypeArgSentinel0 TypeArgSentinel1)",
                "(: b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(: b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_nested_type_context_matrix_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 3,
                "rung_count_executed": 4,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_nested_type_context_matrix_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas nested-type context matrix gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["nested_type_context_matrix_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_generic_four_field_context_arity_rungs_order_arity_before_tokens(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_generic_four_field_context_arity_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope b-profile-000 (Requires) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires MemoryTarget0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 PLNReadyViews) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_generic_four_field_context_arity_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 2,
                "rung_count_executed": 3,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_generic_four_field_context_arity_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas generic four-field context arity gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["generic_four_field_context_arity_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_four_field_nested_position_rungs_move_sentinel_type_by_slot(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_four_field_nested_position_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadA PayloadB)",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadB)",
                "(ProofEnvelope PayloadA PayloadB (Requires TypeArgSentinel0 TypeArgSentinel1))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_four_field_nested_position_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 0,
                "rung_count_executed": 1,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_four_field_nested_position_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas four-field nested-position gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["four_field_nested_position_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_four_field_neighbor_shape_rungs_add_neighbors_stepwise(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_four_field_neighbor_shape_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadB)",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadB)",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (TruthValuePayload 1.0 1.0))",
                "(ProofEnvelope b-profile-000 (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_four_field_neighbor_shape_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 4,
                "rung_count_executed": 5,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_four_field_neighbor_shape_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas four-field neighbor-shape gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["four_field_neighbor_shape_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_four_field_right_payload_arity_rungs_vary_generic_before_stv(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_four_field_right_payload_arity_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) PayloadB)",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (RightPayload))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (RightPayload 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (RightPayload 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_four_field_right_payload_arity_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 3,
                "rung_count_executed": 4,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_four_field_right_payload_arity_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas four-field right-payload arity gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["four_field_right_payload_arity_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_four_field_adjacent_nested_arity_rungs_vary_left_nested_type(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"

        rungs = profile.materialize_four_field_adjacent_nested_arity_rungs(statement)

        self.assertEqual(
            rungs,
            [
                "(ProofEnvelope PayloadA (Requires) (RightPayload 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0) (RightPayload 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (RightPayload 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires) (STV 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0) (STV 1.0 1.0))",
                "(ProofEnvelope PayloadA (Requires TypeArgSentinel0 TypeArgSentinel1) (STV 1.0 1.0))",
            ],
        )

    def test_materialize_four_field_adjacent_nested_arity_gate_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 2,
                "rung_count_executed": 3,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_four_field_adjacent_nested_arity_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas four-field adjacent-nested arity gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["four_field_adjacent_nested_arity_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_nested_type_arity_and_context_matrix_rungs_reject_atom_type(self):
        with self.assertRaises(ValueError):
            profile.materialize_nested_type_arity_matrix_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_nested_type_context_matrix_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_generic_four_field_context_arity_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_four_field_nested_position_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_four_field_neighbor_shape_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_four_field_right_payload_arity_rungs("(: p PlainType (STV 1 1))")
        with self.assertRaises(ValueError):
            profile.materialize_four_field_adjacent_nested_arity_rungs("(: p PlainType (STV 1 1))")

    def test_materialize_proof_shape_ladder_delegates_to_identity_ladder(self):
        statement = "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"
        captured = {}

        def fake_ladder(rungs, *, project_root, stage_timeout_sec):
            captured["rungs"] = rungs
            captured["project_root"] = project_root
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "source": "non-live materialize-stmt-lambdas identity ladder gate",
                "status": "blocked",
                "first_blocked_rung": 4,
                "rung_count_executed": 5,
                "gates": [],
            }

        with patch.object(profile, "run_materialize_identity_ladder_gate", side_effect=fake_ladder):
            result = profile.run_materialize_proof_shape_ladder_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
            )

        self.assertEqual(result["source"], "non-live materialize-stmt-lambdas proof-shape ladder gate")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["proof_shape_rungs"], captured["rungs"])
        self.assertEqual(captured["project_root"], Path("/project"))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertIn("No mm2compile, compileadd, query", " ".join(result["gates"]))

    def test_materialize_proof_shape_rungs_reject_non_proof_atom(self):
        with self.assertRaises(ValueError):
            profile.materialize_identity_proof_shape_rungs("(Requires MemoryTarget0 PLNReadyViews)")

    def test_inspect_compileadd_bottleneck_sources_records_target_definitions(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            metta_dir = repo / "pettachainer" / "metta"
            chainer_dir = metta_dir / "chainer"
            chainer_dir.mkdir(parents=True)
            (metta_dir / "petta_chainer.metta").write_text(
                "!(import! &self chainer/compile)\n"
                "!(import! &self chainer/mining)\n"
                "(= (materialize-stmt-lambdas $term)\n"
                "   (if (is-var $term) $term (materialize-stmt-lambdas (car-atom $term))))\n"
                "(= (compileadd $kb $stmt)\n"
                "   (let* (($stmt1 (materialize-stmt-lambdas $stmt))\n"
                "          ($atoms (collapse (mm2compile $kb $stmt1))))\n"
                "      $atoms))\n"
                "(= (compileadd-mine $kb $stmt) (compileadd $kb $stmt))\n",
                encoding="utf-8",
            )
            (chainer_dir / "compile.metta").write_text(
                "(= (index-source-implication $kb $stmt) ())\n"
                "(= (compile $kb $stmt) (((() |- ($stmt)) ())))\n"
                "(= (mm2compile $kb $stmt)\n"
                "   (progn (remove-all-atoms ctx) (superpose ((mm2stmt (compile $kb $stmt)) (get-atoms ctx)))))\n",
                encoding="utf-8",
            )
            (chainer_dir / "mining.metta").write_text(
                "(= (maybe-process-on-add $kb $stmt) ())\n",
                encoding="utf-8",
            )

            summary = profile.inspect_compileadd_bottleneck_sources(repo)

        self.assertIn("chainer/compile", summary["root_imports"])
        self.assertEqual(
            summary["definitions"]["materialize-stmt-lambdas"]["file"],
            "pettachainer/metta/petta_chainer.metta",
        )
        self.assertTrue(summary["definitions"]["materialize-stmt-lambdas"]["recursive"])
        self.assertIn("compile", summary["definitions"]["mm2compile"]["calls"])
        self.assertEqual(
            [target["symbol"] for target in summary["next_instrumentation_targets"]],
            ["materialize-stmt-lambdas", "mm2compile", "compile_"],
        )
        self.assertIn("no compileadd/query/runtime execution", " ".join(summary["gates"]))

    def test_inspect_compile_dispatch_for_statement_maps_promoted_belief_to_fact_branch(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            chainer_dir = repo / "pettachainer" / "metta" / "chainer"
            chainer_dir.mkdir(parents=True)
            (chainer_dir / "compile.metta").write_text(
                "(= (compile_ $kb (@ $stmt (: $prf $Type $tv)))\n"
                "   (if (is-var $Type) (empty)\n"
                "      (if (= $Type (Implication (cons Premises $premises) (cons Conclusions $conclusions)))\n"
                "         (compile-implication-forward-rules $kb $prf $premises $conclusions)\n"
                "         (if (bidirectional-implication-type? $Type)\n"
                "            (compile_ $kb (: (bi-forward $prf) (Implication $left $right) $tv))\n"
                "            (let $fact-kb (compile-fact-kb $kb)\n"
                "               (superpose ((() |- ((: $fact-kb $prf $Type $tv)))\n"
                "                           (compile-outputs (: $fact-kb $prf $Type $tv)))))))))\n",
                encoding="utf-8",
            )
            (chainer_dir / "logic_config.metta").write_text(
                "!(set-bidirectional-implication-form BiImplication)\n",
                encoding="utf-8",
            )

            summary = profile.inspect_compile_dispatch_for_statement(
                repo,
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
            )

        self.assertEqual(summary["parsed_statement"]["type_head"], "Requires")
        self.assertEqual(summary["selected_compile_branch"], "fact-assertion")
        self.assertIn("concrete non-Implication", summary["reason"])
        self.assertIn("compile-outputs", summary["compile_definition"]["calls"])
        self.assertEqual(summary["configured_bidirectional_heads"], ["BiImplication"])
        self.assertIn("no PeTTaChainer runtime", " ".join(summary["gates"]))

    def test_inspect_compile_dispatch_for_statement_rejects_non_proof_atom(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            chainer_dir = repo / "pettachainer" / "metta" / "chainer"
            chainer_dir.mkdir(parents=True)
            (chainer_dir / "compile.metta").write_text("(= (compile_ $kb $stmt) ())\n", encoding="utf-8")
            (chainer_dir / "logic_config.metta").write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                profile.inspect_compile_dispatch_for_statement(repo, "(Requires MemoryTarget0 PLNReadyViews)")

    def test_run_compile_dispatch_gate_isolates_fact_branch(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(
                label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec,
            )
            return {"label": label, "status": "ok", "output_count": 1, "unique_output_count": 1}

        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "fact-assertion"},
            ),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_dispatch_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_fact_dispatch")
        self.assertEqual(captured["target"], profile._compile_dispatch_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertTrue(result["boundaries"]["no_mm2compile_or_compileadd"])

    def test_run_compile_dispatch_gate_skips_non_fact_branch(self):
        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "implication-rule"},
            ),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_dispatch_gate(
                "(: p (Implication (cons Premises ()) (cons Conclusions ())) (STV 1 0.9))",
                project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_compile_dispatch_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_compile_dispatch_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), max_output_items=0,
            )

    def test_inspect_compile_fact_branch_shape_closes_copied_source(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (compile_ $kb (@ $stmt (: $prf $Type $tv)))\n"
                "   (let $fact-kb (compile-fact-kb $kb)\n"
                "      (superpose ((() |- ((: $fact-kb $prf $Type $tv)))\n"
                "                  (compile-outputs (: $fact-kb $prf $Type $tv))))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_compile_fact_branch_shape(repo)

        self.assertTrue(result["shape_confirmed"])
        self.assertIn("superposes", result["interpretation"])

    def test_inspect_compile_fact_branch_shape_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (compile_ $kb (@ $stmt (: $prf $Type $tv))) (compile-fact-kb $kb))\n",
                encoding="utf-8",
            )

            result = profile.inspect_compile_fact_branch_shape(repo)

        self.assertFalse(result["shape_confirmed"])
        self.assertIn("do not run", result["interpretation"])

    def test_run_compile_fact_branch_component_gate_isolates_components(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            return {"status": "ok", "components_admitted": True}

        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_branch_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_fact_branch_component_gate(
                statement, project_root=Path("/project"), stage_timeout_sec=3.0, max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_fact_branch_components")
        self.assertEqual(captured["target"], profile._compile_fact_branch_component_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertTrue(result["boundaries"]["no_compile_or_mm2compile_or_compileadd"])

    def test_run_compile_fact_branch_component_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_branch_shape", return_value={"shape_confirmed": False}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_fact_branch_component_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_compile_fact_literal_kb_gate_measures_three_rungs(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            rung = {"output_count": 32, "unique_output_count": 1}
            return {
                "status": "ok",
                "base_clause_superpose": rung,
                "literal_kb_with_empty_arm": rung,
                "literal_kb_fact_branch": rung,
            }

        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_branch_shape", return_value={"shape_confirmed": True}),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_compile_fact_literal_kb_gate(
                statement, project_root=Path("/project"), stage_timeout_sec=3.0, max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "compile_fact_literal_kb_ladder")
        self.assertEqual(captured["target"], profile._compile_fact_literal_kb_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertTrue(result["boundaries"]["literal_kb_substitution_only"])
        self.assertTrue(result["boundaries"]["no_compile_or_mm2compile_or_compileadd"])

    def test_run_compile_fact_literal_kb_gate_skips_source_drift(self):
        with (
            patch.object(profile, "inspect_compile_dispatch_for_statement", return_value={"selected_compile_branch": "fact-assertion"}),
            patch.object(profile, "inspect_compile_fact_branch_shape", return_value={"shape_confirmed": False}),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_compile_fact_literal_kb_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_compile_fact_literal_kb_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_compile_fact_literal_kb_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), stage_timeout_sec=0,
            )

    def test_fact_compiled_clause_builds_one_canonical_base_fact(self):
        clause = profile._fact_compiled_clause(
            "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))", "kb-test",
        )

        self.assertEqual(
            clause,
            "(() |- ((: (kb-test MAIN Nil) p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))))",
        )
        with self.assertRaises(ValueError):
            profile._fact_compiled_clause("(Requires A B)", "kb-test")
        with self.assertRaises(ValueError):
            profile._fact_compiled_clause("(: p (S x) (STV 1 0.9))", "bad kb")

    def test_inspect_mm2stmt_fact_case_overlap_closes_exact_source_cause(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (mm2stmt $stmt) (case $stmt (((() |- ($ccl)) $ccl) "
                "(($prms |- ($ccl)) (rules ($prms |- $ccl))))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_mm2stmt_fact_case_overlap(repo)

        self.assertTrue(result["fact_arm_present"])
        self.assertTrue(result["general_arm_present"])
        self.assertTrue(result["overlap_confirmed"])
        self.assertIn("matches both", result["interpretation"])
        self.assertTrue(result["boundaries"]["source_inspection_only"])

    def test_inspect_mm2stmt_fact_case_overlap_fails_closed_on_source_drift(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (mm2stmt $stmt) (case $stmt (((() |- ($ccl)) $ccl))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_mm2stmt_fact_case_overlap(repo)

        self.assertTrue(result["fact_arm_present"])
        self.assertFalse(result["general_arm_present"])
        self.assertFalse(result["overlap_confirmed"])
        self.assertIn("do not attribute", result["interpretation"])

    def test_inspect_mm2compile_collection_shape_closes_copied_source(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (mm2compile $kb $stmt)\n"
                "   (progn (remove-all-atoms ctx) "
                "(superpose ((mm2stmt (compile $kb $stmt)) (get-atoms ctx)))))\n",
                encoding="utf-8",
            )

            result = profile.inspect_mm2compile_collection_shape(repo)

        self.assertTrue(result["shape_confirmed"])
        self.assertIn("superposes", result["interpretation"])
        self.assertTrue(result["boundaries"]["source_inspection_only"])

    def test_inspect_mm2compile_collection_shape_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
            compile_path.parent.mkdir(parents=True)
            compile_path.write_text(
                "(= (mm2compile $kb $stmt) (mm2stmt (compile $kb $stmt)))\n",
                encoding="utf-8",
            )

            result = profile.inspect_mm2compile_collection_shape(repo)

        self.assertFalse(result["shape_confirmed"])
        self.assertIn("do not run", result["interpretation"])

    def test_run_mm2stmt_deduplicated_fact_gate_isolates_one_clause(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            return {
                "label": label,
                "status": "ok",
                "expected_fact_present": True,
                "converted_count": 2,
                "converted_unique_count": 1,
                "ctx_atom_count": 0,
            }

        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "fact-assertion"},
            ),
            patch.object(
                profile,
                "inspect_mm2stmt_fact_case_overlap",
                return_value={"overlap_confirmed": True},
            ),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_mm2stmt_deduplicated_fact_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "mm2stmt_deduplicated_fact")
        self.assertEqual(captured["target"], profile._mm2stmt_deduplicated_fact_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertTrue(result["mm2stmt_inspection"]["overlap_confirmed"])
        self.assertTrue(result["boundaries"]["no_compile_or_mm2compile_or_compileadd"])

    def test_run_mm2stmt_deduplicated_fact_gate_skips_non_fact_branch(self):
        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "implication-rule"},
            ),
            patch.object(
                profile,
                "inspect_mm2stmt_fact_case_overlap",
                return_value={"overlap_confirmed": True},
            ),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_mm2stmt_deduplicated_fact_gate(
                "(: p (Implication (cons Premises ()) (cons Conclusions ())) (STV 1 0.9))",
                project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        configure.assert_not_called()

    def test_run_mm2stmt_deduplicated_fact_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_mm2stmt_deduplicated_fact_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), stage_timeout_sec=0,
            )

    def test_run_mm2compile_deduplicated_fact_gate_isolates_collection(self):
        statement = "(: p (Requires MemoryTarget0 PLNReadyViews) (STV 1 0.9))"
        captured = {}

        def fake_stage(label, target, args, *, stage_timeout_sec):
            captured.update(label=label, target=target, args=args, stage_timeout_sec=stage_timeout_sec)
            return {
                "label": label,
                "status": "ok",
                "expected_fact_present": True,
                "output_count": 2,
                "unique_output_count": 1,
            }

        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "fact-assertion"},
            ),
            patch.object(
                profile,
                "inspect_mm2compile_collection_shape",
                return_value={"shape_confirmed": True},
            ),
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_stage),
        ):
            result = profile.run_mm2compile_deduplicated_fact_gate(
                statement,
                project_root=Path("/project"),
                stage_timeout_sec=3.0,
                max_output_items=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(captured["label"], "mm2compile_deduplicated_fact_collection")
        self.assertEqual(captured["target"], profile._mm2compile_deduplicated_fact_stage)
        self.assertEqual(captured["args"], (statement, 4))
        self.assertEqual(captured["stage_timeout_sec"], 3.0)
        self.assertTrue(result["boundaries"]["no_compile_or_compileadd"])

    def test_run_mm2compile_deduplicated_fact_gate_skips_source_drift(self):
        with (
            patch.object(
                profile,
                "inspect_compile_dispatch_for_statement",
                return_value={"selected_compile_branch": "fact-assertion"},
            ),
            patch.object(
                profile,
                "inspect_mm2compile_collection_shape",
                return_value={"shape_confirmed": False},
            ),
            patch.object(profile, "_configure_local_runtime") as configure,
        ):
            result = profile.run_mm2compile_deduplicated_fact_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"),
            )

        self.assertEqual(result["status"], "skipped")
        self.assertIn("source shape", result["reason"])
        configure.assert_not_called()

    def test_run_mm2compile_deduplicated_fact_gate_rejects_nonpositive_bounds(self):
        with self.assertRaises(ValueError):
            profile.run_mm2compile_deduplicated_fact_gate(
                "(: p (S x) (STV 1 0.9))", project_root=Path("/project"), max_output_items=0,
            )

    def test_inspect_petta_static_import_source_flags_current_export_as_unsafe(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            lib_dir = repo / "lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / "lib_import.pl").write_text(
                "metta_file_to_prolog(Input, Space, Output) :- true.\n"
                "convert_stream(In, Out, Space) :- read_line_to_string(In, Line), convert_line(Line, Space, Out).\n"
                "convert_line(Line0, Space, Out) :- sub_string(Line0, 1, _, 1, Inner0), replace_all(\"(\", \"[\", Inner0, Inner1).\n"
                "'static-import!'(Space, File, true) :- metta_file_to_prolog(MettaFile, Space, PlFile), qcompile(PlFile), consult(QlfFile).\n"
                ":- multifile '~w'/3.\n",
                encoding="utf-8",
            )

            summary = profile.inspect_petta_static_import_source(
                repo,
                ["(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"],
            )

        self.assertTrue(summary["source_features"]["defines_static_import"])
        self.assertTrue(summary["source_features"]["uses_qcompile"])
        self.assertFalse(summary["sample_atoms_safe_for_current_converter"])
        converted = summary["sample_conversions"][0]["converted_prolog_fact"]
        self.assertEqual(converted, "'gckb'(:,b-profile-000,[Requires,MemoryTarget0,PLNReadyViews],[STV,0.70,0.55]).")
        self.assertIn("MemoryTarget0", " ".join(summary["sample_conversions"][0]["warnings"]))
        self.assertIn("Do not use static-import! directly", summary["recommendation"])
        self.assertIn("no SWI qcompile", " ".join(summary["gates"]))

    def test_design_static_import_microbenchmark_atoms_uses_safe_three_argument_facts(self):
        summary = profile.design_static_import_microbenchmark_atoms(
            [
                "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
                "(EvidencePacket (Requires MemoryTarget0 PLNReadyViews) (EC 3.0 1.0) "
                "((domain omegaclaw-memory) (promotion-rule explicit-profile-workload)) pe-profile-000)",
            ]
        )

        self.assertTrue(summary["all_records_safe_for_current_converter"])
        self.assertEqual(
            summary["records"][0]["normalized_atom"],
            "(pm_stv_statement b_profile_000 (pm_stv_payload requires_memorytarget0_plnreadyviews 0.70 0.55))",
        )
        self.assertEqual(
            summary["records"][0]["converted_prolog_fact"],
            "'gckb'(pm_stv_statement,b_profile_000,[pm_stv_payload,requires_memorytarget0_plnreadyviews,0.70,0.55]).",
        )
        self.assertEqual(
            summary["records"][1]["normalized_atom"],
            "(pm_evidence_packet requires_memorytarget0_plnreadyviews (pm_ec_payload 3.0 1.0 pe_profile_000))",
        )
        self.assertIn("temporary scratch", " ".join(summary["benchmark_gate"]))

    def test_compileadd_strategy_summary_recommends_precompiled_cache_gate(self):
        sample_profile = {
            "results": [
                {
                    "events": [
                        {"label": "check_stmt_all", "status": "ok"},
                        {"label": "pettachainer_init_only", "status": "ok"},
                        {"label": "compileadd_probe_materialize_direct", "status": "timeout"},
                        {"label": "compileadd_probe_materialize_eval_control", "status": "timeout"},
                        {"label": "compileadd_probe_mm2compile_direct", "status": "timeout"},
                        {"label": "compileadd_probe_mm2compile_eval_control", "status": "timeout"},
                        {"label": "compileadd_probe_index_source_direct", "status": "ok"},
                        {"label": "compileadd_probe_maybe_process_on_add_direct", "status": "ok"},
                        {"label": "proof_runtime_add_only", "status": "timeout"},
                    ]
                }
            ]
        }

        summary = profile.summarize_compileadd_strategy(sample_profile)

        self.assertEqual(summary["recommended_next_add_path"], "precompiled_statement_cache_gate")
        self.assertEqual(
            summary["fast_later_probes"],
            ["compileadd_probe_index_source_direct", "compileadd_probe_maybe_process_on_add_direct"],
        )
        self.assertIn("checked handoff inputs only", " ".join(summary["gates"]))

    def test_contextual_profile_schedules_add_only_bottleneck_stages(self):
        def fake_isolated_stage(label, _target, _args, *, stage_timeout_sec):
            return {"label": label, "status": "ok", "timeout_sec": stage_timeout_sec}

        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(
                profile,
                "_build_export_payload",
                return_value={
                    "statements": ["(: p (S x) (STV 1 0.9))"],
                    "packets": ["(EvidencePacket (S x) (EC 1 0) () pe)"],
                },
            ),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_isolated_stage),
        ):
            result = profile.profile_sizes(
                [1],
                steps=1,
                timeout_sec=1.0,
                project_root=Path("/unused"),
                stage_timeout_sec=2.0,
                include_runtime_add=True,
                include_contextual=True,
            )

        labels = [event["label"] for event in result["results"][0]["events"]]
        self.assertEqual(
            labels,
            [
                "build_store_and_exports",
                "check_stmt_all",
                "pettachainer_init_only",
                "compileadd_probe_materialize_direct",
                "compileadd_probe_materialize_eval_control",
                "compileadd_probe_mm2compile_direct",
                "compileadd_probe_mm2compile_eval_control",
                "compileadd_probe_internalize_direct",
                "compileadd_probe_externalize_direct",
                "compileadd_probe_index_source_direct",
                "compileadd_probe_add_internalized_direct",
                "compileadd_probe_maybe_process_on_add_direct",
                "proof_runtime_add_only",
                "proof_runtime_add_and_query",
                "contextual_packet_add_only",
                "contextual_runtime_add_and_query",
            ],
        )


    def test_static_import_fact_goal_strips_clause_full_stop(self):
        fact = "'pmbench'(pm_stv_statement,b_profile_000,[pm_stv_payload,key,0.70,0.55])."

        self.assertEqual(
            profile._petta_static_import_fact_goal(fact),
            "'pmbench'(pm_stv_statement,b_profile_000,[pm_stv_payload,key,0.70,0.55])",
        )
        with self.assertRaises(ValueError):
            profile._petta_static_import_fact_goal("'pmbench'(a,b,c)")

    def test_run_static_import_microbenchmark_skips_unsafe_atoms(self):
        """Microbenchmark should skip when atoms are not converter-safe."""
        result = profile.run_static_import_microbenchmark(
            ["(unsupported_atom_only)"],
            project_root=Path("/unused"),
            stage_timeout_sec=1.0,
        )
        self.assertEqual(result["status"], "skipped")
        self.assertIn("not all normalized atoms are safe", result["reason"])
        self.assertFalse(result["design"]["all_records_safe_for_current_converter"])

    def test_run_static_import_microbenchmark_rejects_unsafe_space_name(self):
        sample = ["(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))"]

        with self.assertRaises(ValueError):
            profile.run_static_import_microbenchmark(
                sample,
                project_root=Path("/unused"),
                stage_timeout_sec=5.0,
                space="bad-space",
            )

    def test_run_static_import_microbenchmark_uses_isolated_stage(self):
        """Microbenchmark should delegate to _run_isolated_stage with expected args."""
        sample = [
            "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
            "(EvidencePacket (Requires MemoryTarget0 PLNReadyViews) (EC 3.0 1.0) "
            "((domain omegaclaw-memory) (promotion-rule explicit-profile-workload)) pe-profile-000)",
        ]
        captured = {}

        def fake_isolated_stage(label, target, args, *, stage_timeout_sec):
            captured["label"] = label
            captured["target"] = target
            captured["args"] = args
            captured["stage_timeout_sec"] = stage_timeout_sec
            return {
                "label": label,
                "status": "ok",
                "seconds": 0.1,
                "result": "loaded",
                "loaded_fact_count": 2,
                "expected_fact_count": 2,
                "facts_match": True,
            }

        with (
            patch.object(profile, "_configure_local_runtime", return_value=None),
            patch.object(profile, "_run_isolated_stage", side_effect=fake_isolated_stage),
        ):
            result = profile.run_static_import_microbenchmark(
                sample,
                project_root=Path("/unused"),
                stage_timeout_sec=5.0,
                space="pmbench",
            )

        self.assertEqual(result["source"], "non-live static-import microbenchmark")
        self.assertEqual(captured["label"], "static_import_load_and_query")
        self.assertEqual(captured["stage_timeout_sec"], 5.0)
        self.assertTrue(result["design"]["all_records_safe_for_current_converter"])
        self.assertEqual(result["runtime_event"]["status"], "ok")
        self.assertTrue(result["runtime_event"]["facts_match"])
        # Verify normalized atoms were passed to the stage
        normalized = captured["args"][0]
        self.assertTrue(all("(" in atom and ")" in atom for atom in normalized))
        # Verify expected facts were passed
        expected = captured["args"][1]
        self.assertTrue(all("pmbench" in fact for fact in expected))
        self.assertEqual(captured["args"][2], "pmbench")
        # Gates
        gates = " ".join(result["gates"])
        self.assertIn("no petta-memory journal writes", gates)
        self.assertIn("not PeTTaChainer compileadd/query success", gates)


if __name__ == "__main__":
    unittest.main()
