"""Fail-closed admission for provider-free usability evidence bundles."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .sexpr import SExpressionSyntaxError, parse_one_list, to_source


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
_MAX_RUNTIME_TAIL_CHARS = 4000
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
_SOURCE_STATUS = "pln-ready-input-not-inferred-belief"
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
_SOURCE_STAMP_NAMES = frozenset(("kind", "source_evidence_id", "source_item"))
_BRIDGE_STAMP_NAMES = frozenset(("kind", "source_item_index", "rule"))
_SOURCE_ITEM_NAMES = frozenset(
    (
        "atom",
        "belief_id",
        "cluster_id",
        "evidence_id",
        "kind",
        "pi_pln_extension",
        "promotion_domain",
        "promotion_event",
        "promotion_rule",
        "source_status",
        "stv",
        "term",
    )
)
_PI_PLN_EXTENSION = {
    "context_selection": "not-run; no generated contexts in this handoff gate",
    "contextual_evidence_packets": [],
    "ec_projection_policy": (
        "preserve packets first; later project EC support/opposition through "
        "reviewed pi-PLN truth-value formulas"
    ),
}


def _is_canonical_term(value: str) -> bool:
    try:
        wrapped = parse_one_list(f"({value})")
    except SExpressionSyntaxError:
        return False
    return len(wrapped) == 1 and to_source(wrapped[0]) == value


def _valid_derivation_provenance(program: dict[str, Any]) -> bool:
    source_term = program.get("source_term")
    derived_term = program.get("derived_term")
    sidecar = program.get("stamp_sidecar")
    if (
        not isinstance(source_term, str)
        or not source_term
        or not _is_canonical_term(source_term)
        or derived_term != f"(PMDerivedFromHandoff {source_term})"
        or not isinstance(sidecar, dict)
        or sidecar.keys() != {"(0)", "(1)"}
    ):
        return False
    source = sidecar["(0)"]
    bridge = sidecar["(1)"]
    if (
        not isinstance(source, dict)
        or source.keys() != _SOURCE_STAMP_NAMES
        or source.get("kind") != "petta-memory-source-sentence"
        or not isinstance(source.get("source_evidence_id"), str)
        or not isinstance(source.get("source_item"), dict)
        or source["source_item"].get("term") != source_term
        or source["source_item"].get("evidence_id")
        != source["source_evidence_id"]
        or not isinstance(bridge, dict)
        or bridge.keys() != _BRIDGE_STAMP_NAMES
        or bridge.get("kind") != "synthetic-non-live-bridge-implication"
        or type(bridge.get("source_item_index")) is not int
        or bridge["source_item_index"] != 0
        or bridge.get("rule") != "PMDerivedFromHandoff implication smoke"
    ):
        return False
    source_item = source["source_item"]
    stv = source_item.get("stv")
    if (
        source_item.keys() != _SOURCE_ITEM_NAMES
        or source_item.get("kind") != "patham9-pln-sentence-input"
        or source_item.get("source_status") != _SOURCE_STATUS
        or source_item.get("pi_pln_extension") != _PI_PLN_EXTENSION
        or any(
            not isinstance(source_item.get(name), str)
            or not source_item[name]
            or not _is_canonical_term(source_item[name])
            for name in (
                "belief_id",
                "cluster_id",
                "evidence_id",
                "promotion_domain",
                "promotion_event",
                "promotion_rule",
            )
        )
        or not isinstance(stv, dict)
        or stv.keys() != {"strength", "confidence"}
        or type(stv.get("strength")) not in (int, float)
        or type(stv.get("confidence")) not in (int, float)
        or not math.isfinite(stv["strength"])
        or not math.isfinite(stv["confidence"])
        or not 0.0 <= stv["strength"] <= 1.0
        or not 0.0 <= stv["confidence"] <= 1.0
    ):
        return False
    expected_source_atom = (
        f"(Sentence {source_term} (stv {stv['strength']} {stv['confidence']}) "
        f"({source['source_evidence_id']}))"
    )
    if source_item.get("atom") != expected_source_atom:
        return False
    source_sentence = (
        f"(Sentence ({source_term} (stv {stv['strength']} {stv['confidence']})) (0))"
    )
    bridge_sentence = (
        f"(Sentence ((Implication {source_term} {derived_term}) "
        "(stv 1.0 0.90)) (1))"
    )
    expected_strength = str(
        float(stv["strength"]) * 1.0 + 0.02 * (1.0 - float(stv["strength"]))
    )
    expected_confidence = str(float(stv["confidence"]) * 0.90)
    if (
        program.get("runtime_sentences") != [source_sentence, bridge_sentence]
        or program.get("expected_result")
        != f"((stv {expected_strength} {expected_confidence}) (0 1))"
    ):
        return False
    return True


def _expected_derivation_program(program: dict[str, Any]) -> str | None:
    runtime_sentences = program.get("runtime_sentences")
    derived_term = program.get("derived_term")
    expected_result = program.get("expected_result")
    if (
        not isinstance(runtime_sentences, list)
        or len(runtime_sentences) != 2
        or any(not isinstance(sentence, str) or not sentence for sentence in runtime_sentences)
        or not isinstance(derived_term, str)
        or not derived_term
        or not isinstance(expected_result, str)
        or not expected_result
    ):
        return None
    source_sentence, bridge_sentence = runtime_sentences
    return "\n".join(
        [
            "!(import! &self PLN)",
            "!(PLN.Init ())",
            f"!(Test (PLN.Query ({source_sentence}",
            f"                   {bridge_sentence})",
            f"                  {derived_term}",
            "                  2 5 8)",
            f"       {expected_result})",
            "",
        ]
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
        or not isinstance(inference.get("stderr_tail"), str)
        or len(inference["stderr_tail"]) > _MAX_RUNTIME_TAIL_CHARS
        or not isinstance(inference.get("stdout_tail"), str)
        or len(inference["stdout_tail"]) > _MAX_RUNTIME_TAIL_CHARS
        or not isinstance(program, dict)
        or program.keys() != _PROGRAM_NAMES
        or program.get("schema") != _PROGRAM_SCHEMA
        or program.get("mode") != _PROGRAM_MODE
        or program.get("boundary") != _PROGRAM_BOUNDARY
        or program.get("runtime_stamp_policy") != _RUNTIME_STAMP_POLICY
        or not _valid_derivation_provenance(program)
        or program.get("program") != _expected_derivation_program(program)
        or not isinstance(classification, dict)
        or classification.keys() != _CLASSIFICATION_NAMES
        or classification.get("test") != _INFERENCE_TEST
        or classification.get("status") != "passed"
        or classification.get("log") is not None
        or classification.get("reasons") != []
        or type(classification.get("returncode")) is not int
        or classification["returncode"] != 0
        or type(classification.get("passed_true_count")) is not int
        or classification["passed_true_count"] != 1
        or type(classification.get("passed_false_count")) is not int
        or classification["passed_false_count"] != 0
        or type(classification.get("error_markers")) is not int
        or classification["error_markers"] != 0
        or not isinstance(semantic_markers, dict)
        or semantic_markers.keys() != _SEMANTIC_MARKER_NAMES
        or semantic_markers.get("semantic_passed") is not True
        or not isinstance(semantic_markers.get("diagnostic_lines"), list)
        or any(
            not isinstance(line, str)
            for line in semantic_markers["diagnostic_lines"]
        )
        or any(
            line not in inference["stdout_tail"]
            and line not in inference["stderr_tail"]
            for line in semantic_markers["diagnostic_lines"]
        )
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
