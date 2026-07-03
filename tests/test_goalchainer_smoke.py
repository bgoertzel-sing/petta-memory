import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petta_memory.goalchainer_smoke import run_goalchainer_handoff_smoke, run_goalchainer_precompiled_handoff_smoke
from petta_memory.store import ValidationError


def _cache():
    return {
        "schema": "petta-memory-goalchainer-handoff-v1",
        "cache_id": "cache-test",
        "items": [
            {
                "goalchainer_slot": "acceptability-belief-evidence",
                "belief_id": "b-redacted-summary",
                "cluster_id": "mc-goal-smoke",
                "promotion_event": "pe-redacted-summary",
                "promotion_domain": "incident-response",
                "source_kind": "pettachainer-stv-statement",
                "atom": "(: b-redacted-summary (Acceptable publish_redacted_summary) (STV 0.91 0.74))",
                "boundary": "read-only evidence for appraisal; not a directive, task claim, or inferred belief",
            },
            {
                "goalchainer_slot": "contextual-appraisal-evidence",
                "belief_id": "b-redacted-summary",
                "cluster_id": "mc-goal-smoke",
                "promotion_event": "pe-redacted-summary",
                "promotion_domain": "incident-response",
                "source_kind": "pettachainer-evidence-packet",
                "atom": "(EvidencePacket (Acceptable publish_redacted_summary) (EC 9 1) ((domain incident-response) (promotion-rule explicit-smoke)) pe-redacted-summary)",
                "boundary": "read-only evidence for appraisal; not a directive, task claim, or inferred belief",
            },
        ],
    }


def _payload():
    return {
        "scenario": "Incident response",
        "runtime": {"reasoner": "offline-keyword"},
        "decisions": [
            {"action_id": "publish_redacted_summary", "status": "recommended", "score": 0.98},
            {"action_id": "publish_raw_log", "status": "blocked", "score": -1.0},
        ],
        "explanation": ["Recommended: Publish redacted summary."],
        "notes": ["The raw log is blocked by privacy."],
        "motivation": None,
    }


class GoalChainerSmokeTests(unittest.TestCase):
    def test_precompiled_handoff_smoke_consumes_cache_without_compileadd(self):
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        smoke = run_goalchainer_precompiled_handoff_smoke(_cache(), goalchainer_repo=repo)

        self.assertEqual(smoke["schema"], "petta-memory-goalchainer-precompiled-smoke-v1")
        self.assertEqual(smoke["mode"], "non-live-precompiled-cache-decision-payload-only")
        self.assertIn("compileadd/query not invoked", smoke["boundary"])
        payload = smoke["decision_payload"]
        self.assertEqual(payload["runtime"]["reasoner"], "petta-memory-precompiled-handoff-cache")
        self.assertEqual(payload["decisions"][0]["action_id"], "publish_redacted_summary")
        redacted = next(item for item in payload["decisions"] if item["action_id"] == "publish_redacted_summary")
        self.assertEqual(redacted["evidence"]["strength"], 0.91)
        self.assertEqual(redacted["evidence"]["confidence"], 0.74)
        self.assertTrue(smoke["checks"]["compileadd_not_invoked"])
        self.assertTrue(smoke["checks"]["no_live_directive_or_task_claim"])

    def test_precompiled_handoff_rejects_missing_stv_items(self):
        cache = _cache()
        cache["items"] = [cache["items"][1]]
        repo = Path(__file__).resolve().parents[4] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
        with self.assertRaisesRegex(ValidationError, "no Acceptable STV"):
            run_goalchainer_precompiled_handoff_smoke(cache, goalchainer_repo=repo)

    def test_non_live_goalchainer_smoke_wraps_decision_payload_with_provenance(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(_payload()), stderr="")

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "OmegaClaw-GoalChainer"
            (repo / "src" / "goal_chainer").mkdir(parents=True)
            (repo / "src" / "goal_chainer" / "cli.py").write_text("", encoding="utf-8")
            smoke = run_goalchainer_handoff_smoke(_cache(), goalchainer_repo=repo, timeout_sec=3, runner=runner)

        self.assertEqual(smoke["schema"], "petta-memory-goalchainer-smoke-v1")
        self.assertEqual(smoke["mode"], "non-live-decision-payload-only")
        self.assertIn("no OmegaClaw skill loaded", smoke["boundary"])
        self.assertEqual(smoke["input_cache_id"], "cache-test")
        self.assertEqual(len(smoke["selected_handoff_items"]), 2)
        self.assertEqual(smoke["selected_handoff_items"][0]["belief_id"], "b-redacted-summary")
        self.assertTrue(smoke["checks"]["ranked_actions"])
        self.assertTrue(smoke["checks"]["recommended_action_present"])
        self.assertTrue(smoke["checks"]["no_live_directive_or_task_claim"])
        command, kwargs = calls[0]
        self.assertEqual(command[2:5], ["goal_chainer.cli", "demo", "--json"])
        self.assertNotIn("directive", command)
        self.assertIn("PYTHONPATH", kwargs["env"])
        self.assertEqual(kwargs["timeout"], 3)

    def test_smoke_rejects_directive_payload(self):
        def runner(command, **kwargs):
            bad = _payload()
            bad["claim"] = {"task": "do-not-claim"}
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(bad), stderr="")

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "OmegaClaw-GoalChainer"
            (repo / "src" / "goal_chainer").mkdir(parents=True)
            (repo / "src" / "goal_chainer" / "cli.py").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "directive/execution"):
                run_goalchainer_handoff_smoke(_cache(), goalchainer_repo=repo, runner=runner)

    def test_smoke_requires_handoff_items(self):
        with self.assertRaisesRegex(ValidationError, "at least one handoff item"):
            run_goalchainer_handoff_smoke({"items": []}, runner=lambda *a, **k: None)


if __name__ == "__main__":
    unittest.main()
