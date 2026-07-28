"""Fail-closed admission for provider-free usability evidence bundles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "petta-memory-provider-free-usability-summary-v2"
ARTIFACT_NAMES = (
    "journal.metta",
    "index.metta",
    "retrieval.metta",
    "retrieval.after-restart.metta",
    "inference.json",
    "read_only_canary.metta",
    "journal.metta.lock",
    "journal.after-ingest.sha256",
    "journal.after-canary.sha256",
    "summary.json",
)
_DIGEST_NAMES = ARTIFACT_NAMES[:-1]
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
_CLAIM_NAMES = (
    "inference_status",
    "retrieval_restart_byte_identical",
    "journal_unchanged_by_canary",
    "canary_mode",
    "autonomous_writes_enabled",
    "promotion_authorized",
)
_SUMMARY_NAMES = frozenset(
    (*_DIGEST_NAMES, "schema_version", "artifact_names", *_CLAIM_NAMES)
)
_SHA256_SIDECAR = re.compile(
    rb"^(?P<digest>[0-9a-f]{64})  (?P<path>(?:[^\r\n]*/)?)journal\.metta\n$"
)
_INFERENCE_SCHEMA = "petta-memory-patham9-pln-derivation-smoke-result-v1"
_INFERENCE_TEST = "patham9-pln-handoff-derivation-smoke"
_PROGRAM_SCHEMA = "petta-memory-patham9-pln-derivation-smoke-program-v1"
_PROGRAM_MODE = "read-only-two-premise-derivation-smoke"
_PROGRAM_BOUNDARY = (
    "loads one generated Sentence plus one synthetic bridge implication into local "
    "patham9/PLN for a bounded derivation smoke; no memory append, no inferred-belief "
    "promotion, no OmegaClaw/GoalChainer live path"
)
_RUNTIME_STAMP_POLICY = (
    "numeric patham9/PLN stamps used for chainer compatibility; source evidence and "
    "synthetic bridge provenance preserved in sidecar"
)
_INFERENCE_NAMES = frozenset(
    (
        "classification",
        "program",
        "returncode",
        "schema",
        "semantic_markers",
        "status",
        "stderr_tail",
        "stdout_tail",
    )
)
_CLASSIFICATION_NAMES = frozenset(
    (
        "error_markers",
        "log",
        "passed_false_count",
        "passed_true_count",
        "reasons",
        "returncode",
        "status",
        "test",
    )
)
_SEMANTIC_MARKER_NAMES = frozenset(
    (
        "diagnostic_lines",
        "error_markers",
        "passed_false_count",
        "passed_true_count",
        "semantic_passed",
    )
)
_PROGRAM_NAMES = frozenset(
    (
        "boundary",
        "derived_term",
        "expected_result",
        "mode",
        "program",
        "runtime_sentences",
        "runtime_stamp_policy",
        "schema",
        "source_term",
        "stamp_sidecar",
    )
)


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bundle artifact is not a regular non-symlink file: {path.name}")
    if path.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"bundle artifact exceeds byte limit: {path.name}")
    return path.read_bytes()


def _read_unambiguous_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular_file(path).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_members,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"bundle {label} is not unambiguous UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"bundle {label} must be a JSON object")
    return value


def validate_provider_free_usability_bundle(root: Path | str) -> dict[str, Any]:
    """Validate and return the frozen schema-v2 summary without writing."""

    bundle = Path(root)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("bundle root must be a non-symlink directory")
    actual_names = tuple(sorted(path.name for path in bundle.iterdir()))
    if actual_names != tuple(sorted(ARTIFACT_NAMES)):
        raise ValueError("bundle artifact inventory does not match schema v2")

    summary = _read_unambiguous_json_object(
        bundle / "summary.json", label="summary"
    )
    if summary.keys() != _SUMMARY_NAMES:
        raise ValueError("bundle summary members do not match schema v2")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported bundle summary schema")
    if summary.get("artifact_names") != list(ARTIFACT_NAMES):
        raise ValueError("declared bundle artifact inventory does not match schema v2")

    for name in _DIGEST_NAMES:
        expected = summary.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"missing or malformed artifact digest: {name}")
        actual = hashlib.sha256(_read_regular_file(bundle / name)).hexdigest()
        if expected != actual:
            raise ValueError(f"artifact digest mismatch: {name}")

    required_claims = {
        "inference_status": "passed",
        "retrieval_restart_byte_identical": True,
        "journal_unchanged_by_canary": True,
        "canary_mode": "read-only",
        "autonomous_writes_enabled": False,
        "promotion_authorized": False,
    }
    for key, value in required_claims.items():
        if summary.get(key) != value or type(summary.get(key)) is not type(value):
            raise ValueError(f"unsafe or unsupported bundle claim: {key}")
    inference = _read_unambiguous_json_object(
        bundle / "inference.json", label="inference result"
    )
    if inference.get("status") != summary["inference_status"]:
        raise ValueError("inference result status does not match summary")
    classification = inference.get("classification")
    program = inference.get("program")
    semantic_markers = inference.get("semantic_markers")
    if (
        inference.keys() != _INFERENCE_NAMES
        or inference.get("schema") != _INFERENCE_SCHEMA
        or type(inference.get("returncode")) is not int
        or inference["returncode"] != 0
        or not isinstance(program, dict)
        or program.keys() != _PROGRAM_NAMES
        or program.get("schema") != _PROGRAM_SCHEMA
        or program.get("mode") != _PROGRAM_MODE
        or program.get("boundary") != _PROGRAM_BOUNDARY
        or program.get("runtime_stamp_policy") != _RUNTIME_STAMP_POLICY
        or not isinstance(classification, dict)
        or classification.keys() != _CLASSIFICATION_NAMES
        or classification.get("test") != _INFERENCE_TEST
        or classification.get("status") != "passed"
        or classification.get("log") is not None
        or classification.get("reasons") != []
        or type(classification.get("returncode")) is not int
        or classification["returncode"] != 0
        or type(classification.get("passed_true_count")) is not int
        or classification["passed_true_count"] < 1
        or type(classification.get("passed_false_count")) is not int
        or classification["passed_false_count"] != 0
        or type(classification.get("error_markers")) is not int
        or classification["error_markers"] != 0
        or not isinstance(semantic_markers, dict)
        or semantic_markers.keys() != _SEMANTIC_MARKER_NAMES
        or semantic_markers.get("semantic_passed") is not True
        or type(semantic_markers.get("passed_true_count")) is not int
        or semantic_markers.get("passed_true_count")
        != classification["passed_true_count"]
        or type(semantic_markers.get("passed_false_count")) is not int
        or semantic_markers.get("passed_false_count") != 0
        or type(semantic_markers.get("error_markers")) is not int
        or semantic_markers.get("error_markers") != 0
    ):
        raise ValueError("inference result does not prove a passed derivation")
    if summary["retrieval.metta"] != summary["retrieval.after-restart.metta"]:
        raise ValueError("restart retrieval digests differ")
    journal_digest = summary["journal.metta"]
    for name in ("journal.after-ingest.sha256", "journal.after-canary.sha256"):
        match = _SHA256_SIDECAR.fullmatch(_read_regular_file(bundle / name))
        if match is None or match.group("digest").decode("ascii") != journal_digest:
            raise ValueError(f"journal checksum sidecar does not match journal: {name}")
    return summary


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result
