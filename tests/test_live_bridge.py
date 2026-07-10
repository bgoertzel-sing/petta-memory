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

    def test_live_bridge_admits_only_query_relevant_threadkeeper_branch_before_goalchainer(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "threadkeeper_canary_decision.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        patham9_calls = []
        goalchainer_calls = []

        def fake_patham9_runner(handoff, *, pln_repo, timeout_sec):
            patham9_calls.append((handoff, pln_repo, timeout_sec))
            return {
                "schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-result-v1",
                "status": "passed",
                "returncode": 0,
                "semantic_markers": {"passed_true_count": 1, "passed_false_count": 0, "semantic_passed": True},
                "program": {"schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1"},
            }

        def fake_goalchainer_runner(cache, **kwargs):
            goalchainer_calls.append((cache, kwargs))
            return {
                "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                "decision_payload": {
                    "decisions": [
                        {
                            "action_id": "install_threadkeeper_canary_on_protomegabot",
                            "status": "recommended",
                        }
                    ]
                },
                "checks": {
                    "no_memory_write": True,
                    "no_task_or_directive_claim": True,
                    "no_live_directive_or_task_claim": True,
                },
                "boundary": "fake read-only GoalChainer boundary",
            }

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            bridge = run_petta_memory_goalchainer_live_bridge(
                journal,
                goalchainer_repo=repo,
                cache_id="bridge-threadkeeper-canary-test",
                query_target="(Acceptable install_threadkeeper_canary_on_protomegabot)",
                require_query_relevance=True,
                include_patham9_runtime=True,
                pln_repo="/tmp/patham9-pln",
                patham9_timeout_sec=7.0,
                patham9_runner=fake_patham9_runner,
                goalchainer_runner=fake_goalchainer_runner,
            )

        self.assertEqual(len(patham9_calls), 1)
        admitted_handoff = patham9_calls[0][0]
        self.assertEqual(admitted_handoff["item_count"], 1)
        self.assertEqual(admitted_handoff["items"][0]["belief_id"], "b-tk-canary-approved")
        self.assertEqual(
            admitted_handoff["items"][0]["term"],
            "(Acceptable install_threadkeeper_canary_on_protomegabot)",
        )
        self.assertEqual(len(goalchainer_calls), 1)
        self.assertEqual(goalchainer_calls[0][0]["item_count"], 14)
        self.assertEqual(bridge["input_counts"]["admitted_items"], 1)
        self.assertEqual(bridge["pi_pln_gate"]["recommended_count"], 1)
        self.assertEqual(
            bridge["goalchainer_gate"]["recommended_action"],
            "install_threadkeeper_canary_on_protomegabot",
        )
        self.assertTrue(bridge["checks"]["patham9_runtime_passed_or_skipped"])
        self.assertTrue(bridge["checks"]["no_memory_write"])

    def test_live_bridge_feeds_patham9_admitted_threadkeeper_branch_into_goalchainer(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "threadkeeper_canary_decision.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        def fake_patham9_runner(handoff, *, pln_repo, timeout_sec):
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
                cache_id="bridge-threadkeeper-feedback-test",
                query_target="(Acceptable install_threadkeeper_canary_on_protomegabot)",
                require_query_relevance=True,
                include_heuristic_memory_probe=False,
                include_patham9_runtime=True,
                pln_repo="/tmp/patham9-pln",
                patham9_timeout_sec=7.0,
                patham9_runner=fake_patham9_runner,
            )

        self.assertEqual(bridge["input_counts"]["admitted_items"], 1)
        self.assertEqual(bridge["goalchainer_gate"]["recommended_action"], "reconcile_threadkeeper_pr")
        decisions = {item["action_id"]: item for item in bridge["goalchainer_gate"]["decisions"]}
        self.assertEqual(decisions["reconcile_threadkeeper_pr"]["status"], "recommended")
        self.assertEqual(decisions["install_threadkeeper_canary_on_protomegabot"]["status"], "candidate")
        self.assertTrue(
            any(
                "patham9/PLN admitted" in proof
                for proof in decisions["install_threadkeeper_canary_on_protomegabot"]["evidence"]["proofs"]
            )
        )
        self.assertIn("rollback path", " ".join(bridge["goalchainer_gate"]["notes"]))

    def test_live_bridge_fails_closed_before_goalchainer_when_patham9_runtime_fails(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        goalchainer_calls = []

        def fake_patham9_runner(handoff, *, pln_repo, timeout_sec):
            return {
                "schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-result-v1",
                "status": "failed",
                "returncode": 1,
                "semantic_markers": {"passed_true_count": 0, "passed_false_count": 1, "semantic_passed": False},
                "program": {"schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1"},
            }

        def fake_goalchainer_runner(*args, **kwargs):
            goalchainer_calls.append((args, kwargs))
            raise AssertionError("GoalChainer must not run after a failed patham9 runtime gate")

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "patham9 runtime gate did not pass"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-patham9-fail-closed-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    include_patham9_runtime=True,
                    pln_repo="/tmp/patham9-pln",
                    patham9_timeout_sec=7.0,
                    patham9_runner=fake_patham9_runner,
                    goalchainer_runner=fake_goalchainer_runner,
                )

        self.assertEqual(goalchainer_calls, [])

    def test_live_bridge_rejects_non_object_patham9_runtime_result_before_goalchainer(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        goalchainer_calls = []

        def fake_patham9_runner(handoff, *, pln_repo, timeout_sec):
            return ["not", "an", "object"]

        def fake_goalchainer_runner(*args, **kwargs):
            goalchainer_calls.append((args, kwargs))
            raise AssertionError("GoalChainer must not run after a malformed patham9 runtime result")

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "non-object result"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-patham9-non-object-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    include_patham9_runtime=True,
                    pln_repo="/tmp/patham9-pln",
                    patham9_timeout_sec=7.0,
                    patham9_runner=fake_patham9_runner,
                    goalchainer_runner=fake_goalchainer_runner,
                )

        self.assertEqual(goalchainer_calls, [])

    def test_live_bridge_rejects_malformed_patham9_runtime_top_level_audit_fields_before_goalchainer(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        cases = [
            ("schema", "", 0, "non-string schema"),
            ("returncode-string", "petta-memory-patham9-pln-multi-sentence-derivation-smoke-result-v1", "0", "nonzero or non-integer returncode"),
            ("returncode-bool", "petta-memory-patham9-pln-multi-sentence-derivation-smoke-result-v1", True, "nonzero or non-integer returncode"),
            ("returncode-nonzero", "petta-memory-patham9-pln-multi-sentence-derivation-smoke-result-v1", 1, "nonzero or non-integer returncode"),
        ]
        for name, schema, returncode, expected_error in cases:
            with self.subTest(name=name):
                goalchainer_calls = []

                def fake_patham9_runner(handoff, *, pln_repo, timeout_sec):
                    return {
                        "schema": schema,
                        "status": "passed",
                        "returncode": returncode,
                        "semantic_markers": {"passed_true_count": 1, "passed_false_count": 0, "semantic_passed": True},
                        "program": {"schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1"},
                    }

                def fake_goalchainer_runner(*args, **kwargs):
                    goalchainer_calls.append((args, kwargs))
                    raise AssertionError("GoalChainer must not run after malformed patham9 runtime top-level audit fields")

                with tempfile.TemporaryDirectory() as td:
                    journal = Path(td) / "journal.metta"
                    store = MediumMemoryStore(journal)
                    store.append_cluster(fixture.read_text(encoding="utf-8"))

                    with self.assertRaisesRegex(ValidationError, expected_error):
                        run_petta_memory_goalchainer_live_bridge(
                            journal,
                            goalchainer_repo=repo,
                            cache_id=f"bridge-patham9-top-{name}-test",
                            query_target="(Acceptable publish_redacted_summary)",
                            require_query_relevance=True,
                            include_patham9_runtime=True,
                            pln_repo="/tmp/patham9-pln",
                            patham9_timeout_sec=7.0,
                            patham9_runner=fake_patham9_runner,
                            goalchainer_runner=fake_goalchainer_runner,
                        )

                self.assertEqual(goalchainer_calls, [])

    def test_live_bridge_rejects_malformed_patham9_runtime_audit_fields_before_goalchainer(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        cases = [
            (
                "semantic_markers",
                ["not", "an", "object"],
                {"schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1"},
                "non-object semantic_markers",
            ),
            (
                "semantic_markers",
                {"passed_true_count": 0, "passed_false_count": 0, "semantic_passed": False},
                {"schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1"},
                "semantic markers did not pass",
            ),
            (
                "program",
                {"passed_true_count": 1, "passed_false_count": 0, "semantic_passed": True},
                ["not", "an", "object"],
                "non-object program artifact",
            ),
            (
                "program_schema",
                {"passed_true_count": 1, "passed_false_count": 0, "semantic_passed": True},
                {"schema": ""},
                "non-string program schema",
            ),
        ]
        for name, semantic_markers, program, expected_error in cases:
            with self.subTest(name=name):
                goalchainer_calls = []

                def fake_patham9_runner(handoff, *, pln_repo, timeout_sec):
                    return {
                        "schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-result-v1",
                        "status": "passed",
                        "returncode": 0,
                        "semantic_markers": semantic_markers,
                        "program": program,
                    }

                def fake_goalchainer_runner(*args, **kwargs):
                    goalchainer_calls.append((args, kwargs))
                    raise AssertionError("GoalChainer must not run after malformed patham9 runtime audit fields")

                with tempfile.TemporaryDirectory() as td:
                    journal = Path(td) / "journal.metta"
                    store = MediumMemoryStore(journal)
                    store.append_cluster(fixture.read_text(encoding="utf-8"))

                    with self.assertRaisesRegex(ValidationError, expected_error):
                        run_petta_memory_goalchainer_live_bridge(
                            journal,
                            goalchainer_repo=repo,
                            cache_id=f"bridge-patham9-{name}-test",
                            query_target="(Acceptable publish_redacted_summary)",
                            require_query_relevance=True,
                            include_patham9_runtime=True,
                            pln_repo="/tmp/patham9-pln",
                            patham9_timeout_sec=7.0,
                            patham9_runner=fake_patham9_runner,
                            goalchainer_runner=fake_goalchainer_runner,
                        )

                self.assertEqual(goalchainer_calls, [])

    def test_live_bridge_rejects_non_object_goalchainer_result(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        def fake_goalchainer_runner(*args, **kwargs):
            return ["not", "an", "object"]

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "GoalChainer gate returned a non-object result"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-goalchainer-non-object-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    goalchainer_runner=fake_goalchainer_runner,
                )

    def test_live_bridge_rejects_non_object_goalchainer_decision_payload(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        def fake_goalchainer_runner(*args, **kwargs):
            return {
                "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                "decision_payload": ["not", "an", "object"],
                "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                "boundary": "fake boundary",
            }

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "non-object decision_payload"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-goalchainer-payload-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    goalchainer_runner=fake_goalchainer_runner,
                )

    def test_live_bridge_rejects_non_object_goalchainer_checks(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        def fake_goalchainer_runner(*args, **kwargs):
            return {
                "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                "decision_payload": {"decisions": []},
                "checks": ["not", "an", "object"],
                "boundary": "fake boundary",
            }

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "non-object checks"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-goalchainer-checks-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    goalchainer_runner=fake_goalchainer_runner,
                )

    def test_live_bridge_rejects_goalchainer_boundary_check_drift(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        cases = [
            ({"no_live_directive_or_task_claim": True}, "no_memory_write"),
            ({"no_memory_write": False, "no_live_directive_or_task_claim": True}, "no_memory_write"),
            ({"no_memory_write": True}, "no_live_directive_or_task_claim"),
            ({"no_memory_write": True, "no_live_directive_or_task_claim": False}, "no_live_directive_or_task_claim"),
        ]
        for checks, expected_error in cases:
            with self.subTest(checks=checks):
                def fake_goalchainer_runner(*args, **kwargs):
                    return {
                        "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                        "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                        "decision_payload": {"decisions": []},
                        "checks": checks,
                        "boundary": "fake boundary",
                    }

                with tempfile.TemporaryDirectory() as td:
                    journal = Path(td) / "journal.metta"
                    store = MediumMemoryStore(journal)
                    store.append_cluster(fixture.read_text(encoding="utf-8"))

                    with self.assertRaisesRegex(ValidationError, expected_error):
                        run_petta_memory_goalchainer_live_bridge(
                            journal,
                            goalchainer_repo=repo,
                            cache_id="bridge-goalchainer-boundary-checks-test",
                            query_target="(Acceptable publish_redacted_summary)",
                            require_query_relevance=True,
                            goalchainer_runner=fake_goalchainer_runner,
                        )

    def test_live_bridge_rejects_missing_goalchainer_scalar_metadata(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        cases = [
            ("schema", None, "non-string schema"),
            ("mode", "", "non-string mode"),
            ("boundary", ["not", "a", "string"], "non-string boundary"),
        ]
        for field, value, expected_error in cases:
            with self.subTest(field=field):
                def fake_goalchainer_runner(*args, **kwargs):
                    result = {
                        "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                        "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                        "decision_payload": {"decisions": []},
                        "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                        "boundary": "fake boundary",
                    }
                    result[field] = value
                    return result

                with tempfile.TemporaryDirectory() as td:
                    journal = Path(td) / "journal.metta"
                    store = MediumMemoryStore(journal)
                    store.append_cluster(fixture.read_text(encoding="utf-8"))

                    with self.assertRaisesRegex(ValidationError, expected_error):
                        run_petta_memory_goalchainer_live_bridge(
                            journal,
                            goalchainer_repo=repo,
                            cache_id=f"bridge-goalchainer-{field}-metadata-test",
                            query_target="(Acceptable publish_redacted_summary)",
                            require_query_relevance=True,
                            goalchainer_runner=fake_goalchainer_runner,
                        )

    def test_live_bridge_rejects_malformed_goalchainer_heuristic_memory_probe(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        base_probe = {
            "schema": "petta-memory-goalchainer-heuristic-memory-probe-v1",
            "mode": "non-live-goalchainer-solve-incident-memory-items",
            "memory_proof_present": True,
            "leak_check_safe": True,
            "boundary": "fake non-live heuristic probe boundary",
        }
        cases = [
            ("non-object", "not-an-object", "non-object heuristic_memory_probe"),
            ("bad-schema", {**base_probe, "schema": ""}, "non-string metadata"),
            ("missing-memory-proof", {**base_probe, "memory_proof_present": False}, "did not confirm memory proof"),
            ("unsafe-leak-check", {**base_probe, "leak_check_safe": False}, "did not confirm leak_check_safe"),
        ]
        for name, probe, expected_error in cases:
            with self.subTest(name=name):
                def fake_goalchainer_runner(*args, **kwargs):
                    return {
                        "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                        "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                        "decision_payload": {"decisions": []},
                        "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                        "boundary": "fake boundary",
                        "heuristic_memory_probe": probe,
                    }

                with tempfile.TemporaryDirectory() as td:
                    journal = Path(td) / "journal.metta"
                    store = MediumMemoryStore(journal)
                    store.append_cluster(fixture.read_text(encoding="utf-8"))

                    with self.assertRaisesRegex(ValidationError, expected_error):
                        run_petta_memory_goalchainer_live_bridge(
                            journal,
                            goalchainer_repo=repo,
                            cache_id=f"bridge-goalchainer-heuristic-probe-{name}-test",
                            query_target="(Acceptable publish_redacted_summary)",
                            require_query_relevance=True,
                            goalchainer_runner=fake_goalchainer_runner,
                        )

    def test_live_bridge_rejects_non_list_goalchainer_decisions(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        def fake_goalchainer_runner(*args, **kwargs):
            return {
                "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                "decision_payload": {"decisions": "not-a-list"},
                "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                "boundary": "fake boundary",
            }

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "non-list decisions"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-goalchainer-decisions-list-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    goalchainer_runner=fake_goalchainer_runner,
                )

    def test_live_bridge_rejects_non_object_goalchainer_decision_entries(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        def fake_goalchainer_runner(*args, **kwargs):
            return {
                "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                "decision_payload": {"decisions": [{"status": "held"}, ["not", "an", "object"]]},
                "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                "boundary": "fake boundary",
            }

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "non-object decision entries"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-goalchainer-decision-entry-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    goalchainer_runner=fake_goalchainer_runner,
                )

    def test_live_bridge_rejects_malformed_goalchainer_notes(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        def fake_goalchainer_runner(*args, **kwargs):
            return {
                "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                "decision_payload": {"decisions": [], "notes": "not-a-list"},
                "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                "boundary": "fake boundary",
            }

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "non-list notes"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-goalchainer-notes-list-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    goalchainer_runner=fake_goalchainer_runner,
                )

    def test_live_bridge_rejects_malformed_goalchainer_decision_statuses(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        cases = [
            ("missing", [{"action_id": "publish_redacted_summary"}], "malformed status"),
            ("nonstring", [{"status": ["recommended"], "action_id": "publish_redacted_summary"}], "malformed status"),
            ("unknown", [{"status": "approved", "action_id": "publish_redacted_summary"}], "unknown status"),
        ]
        for name, decisions, expected_error in cases:
            with self.subTest(name=name):
                def fake_goalchainer_runner(*args, **kwargs):
                    return {
                        "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                        "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                        "decision_payload": {"decisions": decisions},
                        "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                        "boundary": "fake boundary",
                    }

                with tempfile.TemporaryDirectory() as td:
                    journal = Path(td) / "journal.metta"
                    store = MediumMemoryStore(journal)
                    store.append_cluster(fixture.read_text(encoding="utf-8"))

                    with self.assertRaisesRegex(ValidationError, expected_error):
                        run_petta_memory_goalchainer_live_bridge(
                            journal,
                            goalchainer_repo=repo,
                            cache_id=f"bridge-goalchainer-status-{name}-test",
                            query_target="(Acceptable publish_redacted_summary)",
                            require_query_relevance=True,
                            goalchainer_runner=fake_goalchainer_runner,
                        )

    def test_live_bridge_rejects_malformed_goalchainer_decision_action_ids(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        cases = [
            (
                "recommended-empty",
                [{"status": "recommended", "action_id": ""}],
                "malformed action_id",
            ),
            (
                "nonrecommended-nonstring",
                [{"status": "candidate", "action_id": ["not", "a", "string"]}],
                "malformed action_id",
            ),
            (
                "duplicate-action-id",
                [
                    {"status": "recommended", "action_id": "publish_redacted_summary"},
                    {"status": "candidate", "action_id": "publish_redacted_summary"},
                ],
                "duplicate decision action_id",
            ),
        ]
        for name, decisions, expected_error in cases:
            with self.subTest(name=name):
                def fake_goalchainer_runner(*args, **kwargs):
                    return {
                        "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                        "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                        "decision_payload": {"decisions": decisions},
                        "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                        "boundary": "fake boundary",
                    }

                with tempfile.TemporaryDirectory() as td:
                    journal = Path(td) / "journal.metta"
                    store = MediumMemoryStore(journal)
                    store.append_cluster(fixture.read_text(encoding="utf-8"))

                    with self.assertRaisesRegex(ValidationError, expected_error):
                        run_petta_memory_goalchainer_live_bridge(
                            journal,
                            goalchainer_repo=repo,
                            cache_id=f"bridge-goalchainer-action-id-{name}-test",
                            query_target="(Acceptable publish_redacted_summary)",
                            require_query_relevance=True,
                            goalchainer_runner=fake_goalchainer_runner,
                        )

    def test_live_bridge_rejects_malformed_goalchainer_decision_evidence(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        cases = [
            (
                "evidence-non-object",
                [{"status": "recommended", "action_id": "publish_redacted_summary", "evidence": "memory proof"}],
                "non-object evidence",
            ),
            (
                "proofs-non-list",
                [
                    {
                        "status": "recommended",
                        "action_id": "publish_redacted_summary",
                        "evidence": {"proofs": "proof-a"},
                    }
                ],
                "non-list proofs",
            ),
            (
                "candidate-without-action-id-evidence-non-object",
                [{"status": "candidate", "evidence": "memory proof"}],
                "non-object evidence",
            ),
        ]
        for name, decisions, expected_error in cases:
            with self.subTest(name=name):
                def fake_goalchainer_runner(*args, **kwargs):
                    return {
                        "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                        "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                        "decision_payload": {"decisions": decisions},
                        "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                        "boundary": "fake boundary",
                    }

                with tempfile.TemporaryDirectory() as td:
                    journal = Path(td) / "journal.metta"
                    store = MediumMemoryStore(journal)
                    store.append_cluster(fixture.read_text(encoding="utf-8"))

                    with self.assertRaisesRegex(ValidationError, expected_error):
                        run_petta_memory_goalchainer_live_bridge(
                            journal,
                            goalchainer_repo=repo,
                            cache_id=f"bridge-goalchainer-evidence-{name}-test",
                            query_target="(Acceptable publish_redacted_summary)",
                            require_query_relevance=True,
                            goalchainer_runner=fake_goalchainer_runner,
                        )

    def test_live_bridge_rejects_multiple_recommended_goalchainer_decisions(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "goalchainer_handoff_smoke.metta"
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"

        def fake_goalchainer_runner(*args, **kwargs):
            return {
                "schema": "petta-memory-goalchainer-precompiled-smoke-result-v1",
                "mode": "non-live-goalchainer-precompiled-handoff-smoke",
                "decision_payload": {
                    "decisions": [
                        {"status": "recommended", "action_id": "publish_redacted_summary"},
                        {"status": "recommended", "action_id": "defer_for_operator_review"},
                    ]
                },
                "checks": {"no_memory_write": True, "no_live_directive_or_task_claim": True},
                "boundary": "fake boundary",
            }

        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.metta"
            store = MediumMemoryStore(journal)
            store.append_cluster(fixture.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValidationError, "multiple recommended decisions"):
                run_petta_memory_goalchainer_live_bridge(
                    journal,
                    goalchainer_repo=repo,
                    cache_id="bridge-goalchainer-multiple-recommended-test",
                    query_target="(Acceptable publish_redacted_summary)",
                    require_query_relevance=True,
                    goalchainer_runner=fake_goalchainer_runner,
                )

    def test_live_bridge_rejects_empty_journal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValidationError, "non-empty journal"):
                run_petta_memory_goalchainer_live_bridge(Path(td) / "empty.metta")


if __name__ == "__main__":
    unittest.main()
