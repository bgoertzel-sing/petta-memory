"""Tests for TraceAttribution: persisted proof-trace attribution with stable reload identity."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import petta_memory.pipln_models as pipln_models
from petta_memory.pipln_models import (
    PeTTaChainerInputStatement,
    PeTTaChainerEpisodeContract,
    PeTTaChainerStageCapture,
    TraceAttribution,
    build_trace_attribution,
    build_pettachainer_derived_result_capture,
    build_pettachainer_rule_attribution,
    read_trace_attribution,
    write_trace_attribution,
)


class TraceAttributionTests(unittest.TestCase):
    """Prove that TraceAttribution survives store/reload with identity intact."""

    @staticmethod
    def _derived_capture():
        """Build a minimal PeTTaChainerDerivedResultCapture for testing."""
        fact_digest = "a" * 64
        rule_digest = "c" * 64
        fact = PeTTaChainerInputStatement(
            atom=f"(: pm-{fact_digest} (S a) (STV 0.8 0.6))",
            proof_id=f"pm-{fact_digest}", sentence_digest=fact_digest,
            canonical_term="(S a)", strength=0.8, confidence=0.6,
            stamp_ints=(0,), evidence_basis_ids=("basis-fact",),
        )
        rule = PeTTaChainerInputStatement(
            atom=f"(: pm-{rule_digest} (Implication (Premises (S $x)) (Conclusions (T $x))) (STV 0.9 0.8))",
            proof_id=f"pm-{rule_digest}", sentence_digest=rule_digest,
            canonical_term="(Implication (Premises (S $x)) (Conclusions (T $x)))",
            strength=0.9, confidence=0.8,
            stamp_ints=(1,), evidence_basis_ids=("basis-rule",),
        )
        validator = pipln_models.build_pettachainer_stage_capture(
            label="validate_repaired_one_rule_derivation", elapsed_seconds=0.1,
            stdout_bytes=1, stdout_sha256="d" * 64,
            stderr_bytes=0, stderr_sha256="e" * 64,
        )
        runtime = pipln_models.build_pettachainer_stage_capture(
            label="repaired_one_rule_derivation", elapsed_seconds=0.2,
            stdout_bytes=1, stdout_sha256="f" * 64,
            stderr_bytes=0, stderr_sha256="0" * 64,
        )
        proof = f"(rule-proof {rule.proof_id} {fact.proof_id})"
        return build_pettachainer_derived_result_capture(
            episode_id="episode-trace-test",
            chart_fingerprint="b" * 64,
            fact=fact, rule=rule, query_term="(T a)",
            derived_atom=f"(: {proof} (T a) (STV 0.7 0.5))",
            derived_proof=proof, strength=0.7, confidence=0.5,
            validator_capture=validator, runtime_capture=runtime,
        )

    # --- Construction ---

    def test_construction_requires_all_mandatory_fields(self):
        capture = self._derived_capture()
        proof_trace = "(trace (rule-proof pm-ccc pm-aaa) (T a) (STV 0.7 0.5))"
        attribution = build_trace_attribution(
            result=capture, proof_trace=proof_trace,
        )
        self.assertEqual(attribution.result_digest, capture.result_digest)
        self.assertEqual(attribution.inference_rule, "TotalMP")
        self.assertEqual(attribution.rule_sentence_digest, capture.rule_sentence_digest)
        self.assertEqual(attribution.rule_proof_id, capture.rule_proof_id)
        self.assertEqual(attribution.proof_trace, proof_trace)
        self.assertRegex(attribution.trace_digest, r"^[0-9a-f]{64}$")

    def test_construction_rejects_non_typed_result(self):
        with self.assertRaisesRegex(ValueError, "typed PeTTaChainer derived result"):
            build_trace_attribution(result=None, proof_trace="trace")

    def test_construction_rejects_empty_proof_trace(self):
        capture = self._derived_capture()
        with self.assertRaisesRegex(ValueError, "proof_trace must be a non-empty string"):
            build_trace_attribution(result=capture, proof_trace="")

    def test_construction_rejects_non_string_proof_trace(self):
        capture = self._derived_capture()
        with self.assertRaisesRegex(ValueError, "proof_trace must be a non-empty string"):
            build_trace_attribution(result=capture, proof_trace=None)

    def test_construction_rejects_oversized_proof_trace(self):
        capture = self._derived_capture()
        oversized = "x" * (pipln_models.DEFAULT_MAX_COMPILED_ATOM_CHARS + 1)
        with self.assertRaisesRegex(ValueError, "exceeds max_atom_chars"):
            build_trace_attribution(result=capture, proof_trace=oversized)

    def test_construction_rejects_mismatched_rule_proof_id(self):
        capture = self._derived_capture()
        values = {
            "result_digest": capture.result_digest,
            "inference_rule": "TotalMP",
            "rule_sentence_digest": capture.rule_sentence_digest,
            "rule_proof_id": "pm-wrong",
            "proof_trace": "trace text",
        }
        with self.assertRaisesRegex(ValueError, "rule proof id does not match"):
            TraceAttribution(**values, trace_digest=pipln_models._canonical_hash(values))

    def test_construction_rejects_empty_inference_rule(self):
        capture = self._derived_capture()
        values = {
            "result_digest": capture.result_digest,
            "inference_rule": "",
            "rule_sentence_digest": capture.rule_sentence_digest,
            "rule_proof_id": capture.rule_proof_id,
            "proof_trace": "trace text",
        }
        with self.assertRaisesRegex(ValueError, "inference_rule must be a non-empty string"):
            TraceAttribution(**values, trace_digest=pipln_models._canonical_hash(values))

    def test_construction_rejects_invalid_result_digest(self):
        capture = self._derived_capture()
        values = {
            "result_digest": "not-a-sha256",
            "inference_rule": "TotalMP",
            "rule_sentence_digest": capture.rule_sentence_digest,
            "rule_proof_id": capture.rule_proof_id,
            "proof_trace": "trace text",
        }
        with self.assertRaisesRegex(ValueError, "result_digest"):
            TraceAttribution(**values, trace_digest=pipln_models._canonical_hash(values))

    def test_construction_rejects_invalid_trace_digest(self):
        capture = self._derived_capture()
        values = {
            "result_digest": capture.result_digest,
            "inference_rule": "TotalMP",
            "rule_sentence_digest": capture.rule_sentence_digest,
            "rule_proof_id": capture.rule_proof_id,
            "proof_trace": "trace text",
        }
        with self.assertRaisesRegex(ValueError, "trace attribution digest does not match"):
            TraceAttribution(**values, trace_digest="0" * 64)

    def test_construction_rejects_tampered_proof_trace(self):
        """Recomputing the digest with a different proof_trace must fail."""
        capture = self._derived_capture()
        values = {
            "result_digest": capture.result_digest,
            "inference_rule": "TotalMP",
            "rule_sentence_digest": capture.rule_sentence_digest,
            "rule_proof_id": capture.rule_proof_id,
            "proof_trace": "original trace",
        }
        wrong = pipln_models._canonical_hash(values | {"proof_trace": "tampered trace"})
        with self.assertRaisesRegex(ValueError, "trace attribution digest does not match"):
            TraceAttribution(**values, trace_digest=wrong)

    # --- Store/reload identity ---

    def test_store_reload_preserves_identity(self):
        capture = self._derived_capture()
        proof_trace = "(trace (rule-proof pm-ccc pm-aaa) (T a) (STV 0.7 0.5))"
        attribution = build_trace_attribution(result=capture, proof_trace=proof_trace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_attribution.json"
            write_trace_attribution(path, attribution)
            reloaded = read_trace_attribution(path, result=capture)
            self.assertEqual(reloaded, attribution)
            self.assertEqual(reloaded.trace_digest, attribution.trace_digest)
            self.assertEqual(reloaded.proof_trace, attribution.proof_trace)

    def test_store_is_create_once(self):
        capture = self._derived_capture()
        proof_trace = "trace"
        attribution = build_trace_attribution(result=capture, proof_trace=proof_trace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_attribution.json"
            write_trace_attribution(path, attribution)
            with self.assertRaises(FileExistsError):
                write_trace_attribution(path, attribution)

    def test_reload_rejects_wrong_result(self):
        """Reloading against a different derived result must fail closed."""
        capture = self._derived_capture()
        proof_trace = "trace"
        attribution = build_trace_attribution(result=capture, proof_trace=proof_trace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_attribution.json"
            write_trace_attribution(path, attribution)
            # Build a different capture by changing the STV
            fact = PeTTaChainerInputStatement(
                atom=f"(: pm-{'a' * 64} (S a) (STV 0.8 0.6))",
                proof_id=f"pm-{'a' * 64}", sentence_digest="a" * 64,
                canonical_term="(S a)", strength=0.8, confidence=0.6,
                stamp_ints=(0,), evidence_basis_ids=("basis-fact",),
            )
            rule = PeTTaChainerInputStatement(
                atom=f"(: pm-{'c' * 64} (Implication (Premises (S $x)) (Conclusions (T $x))) (STV 0.9 0.8))",
                proof_id=f"pm-{'c' * 64}", sentence_digest="c" * 64,
                canonical_term="(Implication (Premises (S $x)) (Conclusions (T $x)))",
                strength=0.9, confidence=0.8,
                stamp_ints=(1,), evidence_basis_ids=("basis-rule",),
            )
            validator = pipln_models.build_pettachainer_stage_capture(
                label="validate_repaired_one_rule_derivation", elapsed_seconds=0.1,
                stdout_bytes=1, stdout_sha256="d" * 64,
                stderr_bytes=0, stderr_sha256="e" * 64,
            )
            runtime = pipln_models.build_pettachainer_stage_capture(
                label="repaired_one_rule_derivation", elapsed_seconds=0.2,
                stdout_bytes=1, stdout_sha256="f" * 64,
                stderr_bytes=0, stderr_sha256="0" * 64,
            )
            proof = f"(rule-proof {rule.proof_id} {fact.proof_id})"
            different_capture = build_pettachainer_derived_result_capture(
                episode_id="episode-trace-test",
                chart_fingerprint="b" * 64,
                fact=fact, rule=rule, query_term="(T a)",
                derived_atom=f"(: {proof} (T a) (STV 0.8 0.5))",
                derived_proof=proof, strength=0.8, confidence=0.5,
                validator_capture=validator, runtime_capture=runtime,
            )
            with self.assertRaisesRegex(ValueError, "does not match derived result"):
                read_trace_attribution(path, result=different_capture)

    def test_reload_rejects_tampered_result_digest(self):
        """Tampering with the persisted result_digest must fail closed."""
        capture = self._derived_capture()
        proof_trace = "original trace"
        attribution = build_trace_attribution(result=capture, proof_trace=proof_trace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_attribution.json"
            write_trace_attribution(path, attribution)
            # Tamper with result_digest and recompute both digests
            original_text = path.read_text(encoding="utf-8")
            document = json.loads(original_text)
            payload = document["payload"]
            payload["result_digest"] = "f" * 64
            digest_payload = {
                key: value for key, value in payload.items()
                if key != "trace_digest"
            }
            payload["trace_digest"] = pipln_models._canonical_hash(digest_payload)
            document["document_digest"] = pipln_models._canonical_hash(payload)
            path.write_text(
                json.dumps(document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "result_digest does not match derived result"
            ):
                read_trace_attribution(path, result=capture)

    def test_reload_rejects_malformed_schema(self):
        capture = self._derived_capture()
        proof_trace = "trace"
        attribution = build_trace_attribution(result=capture, proof_trace=proof_trace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_attribution.json"
            write_trace_attribution(path, attribution)
            # Tamper with the schema
            original_text = path.read_text(encoding="utf-8")
            document = json.loads(original_text)
            document["schema"] = "petta-memory-wrong-schema-v1"
            path.write_text(
                json.dumps(document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid trace attribution document schema"):
                read_trace_attribution(path, result=capture)

    def test_reload_rejects_checksum_drift(self):
        capture = self._derived_capture()
        proof_trace = "trace"
        attribution = build_trace_attribution(result=capture, proof_trace=proof_trace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_attribution.json"
            write_trace_attribution(path, attribution)
            # Tamper with the document_digest
            original_text = path.read_text(encoding="utf-8")
            document = json.loads(original_text)
            document["document_digest"] = "0" * 64
            path.write_text(
                json.dumps(document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "document checksum mismatch"):
                read_trace_attribution(path, result=capture)

    def test_write_rejects_non_typed_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_attribution.json"
            with self.assertRaisesRegex(ValueError, "typed attribution"):
                write_trace_attribution(path, None)

    def test_read_rejects_non_typed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_attribution.json"
            with self.assertRaisesRegex(ValueError, "typed derived result"):
                read_trace_attribution(path, result=None)

    # --- Distinct from PeTTaChainerRuleAttribution ---

    def test_trace_attribution_is_distinct_from_rule_attribution(self):
        """TraceAttribution carries proof_trace; PeTTaChainerRuleAttribution does not."""
        capture = self._derived_capture()
        rule_attribution = build_pettachainer_rule_attribution(capture)
        trace_attribution = build_trace_attribution(
            result=capture, proof_trace="runtime trace text",
        )
        self.assertFalse(hasattr(rule_attribution, "proof_trace"))
        self.assertTrue(hasattr(trace_attribution, "proof_trace"))
        self.assertNotEqual(
            rule_attribution.attribution_digest,
            trace_attribution.trace_digest,
        )

    def test_trace_attribution_is_immutable(self):
        capture = self._derived_capture()
        attribution = build_trace_attribution(
            result=capture, proof_trace="trace",
        )
        with self.assertRaises(Exception):
            replace(attribution, proof_trace="changed")


if __name__ == "__main__":
    unittest.main()
