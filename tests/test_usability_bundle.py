import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from petta_memory.usability_bundle import (
    ARTIFACT_NAMES,
    SCHEMA_VERSION,
    validate_provider_free_usability_bundle,
)


class UsabilityBundleTests(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        bundle = root / "bundle"
        bundle.mkdir()
        for name in ARTIFACT_NAMES[:-1]:
            (bundle / name).write_bytes((name + "\n").encode())
        (bundle / "inference.json").write_text(json.dumps({
            "schema": "petta-memory-patham9-pln-derivation-smoke-result-v1",
            "status": "passed",
            "returncode": 0,
            "classification": {
                "log": None,
                "reasons": [],
                "status": "passed",
                "returncode": 0,
                "passed_true_count": 1,
                "passed_false_count": 0,
                "error_markers": 0,
                "test": "patham9-pln-handoff-derivation-smoke",
            },
            "program": {},
            "semantic_markers": {
                "diagnostic_lines": [],
                "semantic_passed": True,
                "passed_true_count": 1,
                "passed_false_count": 0,
                "error_markers": 0,
            },
            "stderr_tail": "",
            "stdout_tail": "",
        }), encoding="utf-8")
        journal_digest = hashlib.sha256(
            (bundle / "journal.metta").read_bytes()
        ).hexdigest()
        sidecar = f"{journal_digest}  {bundle / 'journal.metta'}\n".encode()
        (bundle / "journal.after-ingest.sha256").write_bytes(sidecar)
        (bundle / "journal.after-canary.sha256").write_bytes(sidecar)
        summary = {
            name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            for name in ARTIFACT_NAMES[:-1]
        }
        summary.update(
            schema_version=SCHEMA_VERSION,
            artifact_names=list(ARTIFACT_NAMES),
            inference_status="passed",
            retrieval_restart_byte_identical=True,
            journal_unchanged_by_canary=True,
            canary_mode="read-only",
            autonomous_writes_enabled=False,
            promotion_authorized=False,
        )
        summary["retrieval.after-restart.metta"] = summary["retrieval.metta"]
        (bundle / "retrieval.after-restart.metta").write_bytes(
            (bundle / "retrieval.metta").read_bytes()
        )
        (bundle / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return bundle

    def test_valid_bundle_is_admitted_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            before = {
                path.name: path.read_bytes() for path in bundle.iterdir()
            }

            summary = validate_provider_free_usability_bundle(bundle)

            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                before, {path.name: path.read_bytes() for path in bundle.iterdir()}
            )

    def test_tampered_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            (bundle / "journal.metta").write_text("tampered\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                validate_provider_free_usability_bundle(bundle)

    def test_live_authority_claim_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["promotion_authorized"] = True
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "promotion_authorized"):
                validate_provider_free_usability_bundle(bundle)

    def test_unexpected_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            (bundle / "undeclared.txt").write_text("extra", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inventory"):
                validate_provider_free_usability_bundle(bundle)

    def test_undeclared_summary_member_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["runtime_invocation_authorized"] = True
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "summary members"):
                validate_provider_free_usability_bundle(bundle)

    def test_rehashed_false_journal_checksum_sidecars_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            false_sidecar = f"{'0' * 64}  {bundle / 'journal.metta'}\n".encode()
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for name in (
                "journal.after-ingest.sha256",
                "journal.after-canary.sha256",
            ):
                (bundle / name).write_bytes(false_sidecar)
                summary[name] = hashlib.sha256(false_sidecar).hexdigest()
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match journal"):
                validate_provider_free_usability_bundle(bundle)

    def test_rehashed_failed_inference_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            inference = json.dumps({"status": "failed"}).encode()
            (bundle / "inference.json").write_bytes(inference)
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["inference.json"] = hashlib.sha256(inference).hexdigest()
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "status does not match summary"):
                validate_provider_free_usability_bundle(bundle)

    def test_rehashed_bare_passed_inference_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            inference = json.dumps({"status": "passed"}).encode()
            (bundle / "inference.json").write_bytes(inference)
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["inference.json"] = hashlib.sha256(inference).hexdigest()
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not prove"):
                validate_provider_free_usability_bundle(bundle)

    def test_rehashed_boolean_semantic_marker_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            inference_path = bundle / "inference.json"
            inference = json.loads(inference_path.read_text(encoding="utf-8"))
            inference["semantic_markers"]["passed_true_count"] = True
            inference_bytes = json.dumps(inference).encode()
            inference_path.write_bytes(inference_bytes)
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["inference.json"] = hashlib.sha256(inference_bytes).hexdigest()
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not prove"):
                validate_provider_free_usability_bundle(bundle)

    def test_rehashed_undeclared_inference_authority_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            inference_path = bundle / "inference.json"
            inference = json.loads(inference_path.read_text(encoding="utf-8"))
            inference["promotion_authorized"] = True
            inference_bytes = json.dumps(inference).encode()
            inference_path.write_bytes(inference_bytes)
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["inference.json"] = hashlib.sha256(inference_bytes).hexdigest()
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not prove"):
                validate_provider_free_usability_bundle(bundle)

    def test_rehashed_undeclared_nested_inference_authority_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            inference_path = bundle / "inference.json"
            inference = json.loads(inference_path.read_text(encoding="utf-8"))
            inference["classification"]["live_integration_authorized"] = True
            inference_bytes = json.dumps(inference).encode()
            inference_path.write_bytes(inference_bytes)
            summary_path = bundle / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["inference.json"] = hashlib.sha256(inference_bytes).hexdigest()
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not prove"):
                validate_provider_free_usability_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
