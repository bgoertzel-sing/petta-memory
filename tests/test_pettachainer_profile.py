import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import petta_memory.pettachainer_profile as profile
from petta_memory.pettachainer_profile import _run_isolated_stage, build_profile_store, build_promoted_cluster


def _slow_profile_stage() -> dict[str, object]:
    time.sleep(5)
    return {"result": "too late"}


def _echo_profile_stage(value: str) -> dict[str, object]:
    print("noisy runtime output")
    return {"result": value}


class PeTTaChainerProfileWorkloadTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
