"""Fail-closed admission for provider-free usability evidence bundles."""

from __future__ import annotations

import hashlib
import json
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


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bundle artifact is not a regular non-symlink file: {path.name}")
    if path.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"bundle artifact exceeds byte limit: {path.name}")
    return path.read_bytes()


def validate_provider_free_usability_bundle(root: Path | str) -> dict[str, Any]:
    """Validate and return the frozen schema-v2 summary without writing."""

    bundle = Path(root)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("bundle root must be a non-symlink directory")
    actual_names = tuple(sorted(path.name for path in bundle.iterdir()))
    if actual_names != tuple(sorted(ARTIFACT_NAMES)):
        raise ValueError("bundle artifact inventory does not match schema v2")

    try:
        summary = json.loads(
            _read_regular_file(bundle / "summary.json").decode("utf-8"),
            object_pairs_hook=_reject_duplicate_members,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bundle summary is not unambiguous UTF-8 JSON") from exc
    if not isinstance(summary, dict):
        raise ValueError("bundle summary must be a JSON object")
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
    if summary["retrieval.metta"] != summary["retrieval.after-restart.metta"]:
        raise ValueError("restart retrieval digests differ")
    return summary


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result
