import tempfile
import time
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


if __name__ == "__main__":
    unittest.main()
