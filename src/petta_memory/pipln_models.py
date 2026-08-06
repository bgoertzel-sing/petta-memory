"""Typed Phase-1 domain primitives for the patham9-backed πPLN runtime.

These records are deliberately independent of the persistent MediumMemoryStore.
They model immutable evidence and deterministic episode projection inputs without
turning derived STVs or chart priors into empirical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
from typing import Callable, Iterable, Literal, Mapping

from .sexpr import SExpr, SExpressionSyntaxError, parse_one_list, symbol_text, to_source


IndependenceStatus = Literal["PROVEN_DISJOINT", "ASSUMED", "COUPLED", "UNKNOWN"]
PacketStatus = Literal["ACTIVE", "QUARANTINED", "RETRACTED"]
PacketOrigin = Literal["OBSERVATION", "IMPORT", "REVIEWED_EXPORT"]

DEFAULT_MAX_COMPILED_SENTENCES = 256
DEFAULT_MAX_COMPILED_ATOM_CHARS = 1_000_000
DEFAULT_MAX_KERNEL_RESULT_CHARS = 65_536
DEFAULT_MAX_EPISODE_PROGRAM_CHARS = 2_000_000
DEFAULT_MAX_EPISODE_QUERY_STEPS = 10_000
DEFAULT_MAX_EPISODE_QUEUE_SIZE = 100_000
DEFAULT_MAX_KERNEL_CAPTURE_BYTES = 1_000_000
DEFAULT_MAX_PETTACHAINER_ARTIFACT_BYTES = 1_000_000
_EXECUTABLE_TERM_HEADS = frozenset({
    "!", "bind!", "case", "collapse", "eval", "if", "import!", "include", "let", "let*",
    "match", "pragma!", "superpose",
    # Stock patham9 evaluator/control entry points at pinned revision 55f1751.
    # These are data-looking symbols until the assembled program imports PLN,
    # where they become executable and must never arrive through evidence or a
    # caller-supplied query term.
    "PLN.Config", "PLN.Derive", "PLN.Init", "PLN.Query",
})


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _finite_nonnegative(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite non-negative number")


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _add_exception_note(error: BaseException, note: str) -> None:
    """Retain a secondary diagnostic on supported Python versions."""
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:
        error.__notes__ = [*getattr(error, "__notes__", ()), note]


def _load_unambiguous_json(
    path: str | Path, *, max_bytes: int = DEFAULT_MAX_PETTACHAINER_ARTIFACT_BYTES,
) -> object:
    """Load bounded JSON while rejecting duplicate members at every depth."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object member: {key}")
            result[key] = value
        return result

    artifact_path = Path(path)
    parent_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(os.fspath(artifact_path.parent), parent_flags)
    try:
        parent_metadata = os.fstat(parent_descriptor)
    except BaseException as admission_error:
        try:
            os.close(parent_descriptor)
        except OSError as close_error:
            _add_exception_note(
                admission_error,
                f"JSON artifact parent descriptor close failed: {close_error}"
            )
        raise
    if parent_metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        admission_error = ValueError(
            "JSON artifact parent must not be group- or world-writable"
        )
        try:
            os.close(parent_descriptor)
        except OSError as close_error:
            _add_exception_note(
                admission_error,
                f"JSON artifact parent descriptor close failed: {close_error}"
            )
        raise admission_error
    if parent_metadata.st_uid != os.geteuid():
        admission_error = ValueError(
            "JSON artifact parent must be owned by the current user"
        )
        try:
            os.close(parent_descriptor)
        except OSError as close_error:
            _add_exception_note(
                admission_error,
                f"JSON artifact parent descriptor close failed: {close_error}"
            )
        raise admission_error
    # Open nonblocking so a caller-supplied FIFO cannot stall artifact admission
    # before the descriptor's file type is checked below.  O_NONBLOCK has no
    # effect on ordinary regular-file reads.
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(artifact_path.name, flags, dir_fd=parent_descriptor)
    except BaseException as admission_error:
        try:
            os.close(parent_descriptor)
        except OSError as close_error:
            _add_exception_note(
                admission_error,
                f"JSON artifact parent descriptor close failed: {close_error}"
            )
        raise
    admission_error: BaseException | None = None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("JSON artifact must be a regular file")
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("JSON artifact must not be group- or world-writable")
        if metadata.st_uid != os.geteuid():
            raise ValueError("JSON artifact must be owned by the current user")
        if metadata.st_nlink != 1:
            raise ValueError("JSON artifact must have exactly one filesystem link")
        if metadata.st_size > max_bytes:
            raise ValueError(f"JSON artifact exceeds {max_bytes} byte limit")
        handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        stream_error: BaseException | None = None
        try:
            encoded = handle.read(max_bytes + 1)
            final_metadata = os.fstat(handle.fileno())
            stable_fields = (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
                "st_size", "st_mtime_ns", "st_ctime_ns",
            )
            if any(
                getattr(metadata, field) != getattr(final_metadata, field)
                for field in stable_fields
            ):
                raise ValueError("JSON artifact changed during admission")
            if len(encoded) != final_metadata.st_size:
                raise ValueError("JSON artifact byte count does not match file metadata")
        except BaseException as caught_error:
            stream_error = caught_error
            raise
        finally:
            try:
                handle.close()
            except OSError as close_error:
                if stream_error is None:
                    raise
                note = f"JSON artifact stream close failed: {close_error}"
                add_note = getattr(stream_error, "add_note", None)
                if add_note is not None:
                    add_note(note)
                else:
                    stream_error.__notes__ = [
                        *getattr(stream_error, "__notes__", ()), note,
                    ]
        final_parent_metadata = os.fstat(parent_descriptor)
        stable_parent_fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
        )
        if any(
            getattr(parent_metadata, field) != getattr(final_parent_metadata, field)
            for field in stable_parent_fields
        ):
            raise ValueError("JSON artifact parent changed during admission")
    except BaseException as caught_error:
        admission_error = caught_error
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as close_error:
                if admission_error is None:
                    raise
                note = f"JSON artifact descriptor close failed: {close_error}"
                add_note = getattr(admission_error, "add_note", None)
                if add_note is not None:
                    add_note(note)
                else:
                    admission_error.__notes__ = [
                        *getattr(admission_error, "__notes__", ()), note,
                    ]
        try:
            os.close(parent_descriptor)
        except OSError as close_error:
            if admission_error is None:
                raise
            note = f"JSON artifact parent descriptor close failed: {close_error}"
            add_note = getattr(admission_error, "add_note", None)
            if add_note is not None:
                add_note(note)
            else:
                admission_error.__notes__ = [
                    *getattr(admission_error, "__notes__", ()), note,
                ]
    if len(encoded) > max_bytes:
        raise ValueError(f"JSON artifact exceeds {max_bytes} byte limit")
    return json.loads(encoded.decode("utf-8"), object_pairs_hook=reject_duplicates)


def _sha256_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _snapshot_fingerprint(
    packet_content_digests: tuple[tuple[str, str], ...], *, context_id: str,
    assumption_fingerprint: str, ontology_fingerprint: str,
) -> str:
    return _canonical_hash({
        "packet_content_digests": packet_content_digests,
        "context_id": context_id,
        "assumption_fingerprint": assumption_fingerprint,
        "ontology_fingerprint": ontology_fingerprint,
    })


def _canonical_kernel_term(statement: str) -> str:
    """Canonicalize one data term and reject embedded MeTTa control forms."""
    _nonempty(statement, "statement")
    try:
        term = parse_one_list(statement)
    except SExpressionSyntaxError as error:
        raise ValueError(f"statement must be one valid S-expression list: {error}") from error

    def reject_executable_forms(expression: SExpr) -> None:
        if not isinstance(expression, tuple):
            return
        head = symbol_text(expression[0]) if expression else None
        if head in _EXECUTABLE_TERM_HEADS:
            raise ValueError(f"statement contains executable/control form: {head}")
        for child in expression:
            reject_executable_forms(child)

    reject_executable_forms(term)
    return to_source(term)


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _timestamp(value: str, field: str) -> datetime:
    _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


@dataclass(frozen=True)
class Phase0ReferenceArtifact:
    """Validated identity of one frozen stock-kernel reference episode."""

    example_name: str
    source_sha256: str
    source_cid: str
    runtime_executable_sha256: str
    kernel_commit: str
    semantic_result: str
    query_target: str
    output_sha256: str
    output_bytes: int


def validate_phase0_reference_artifact(
    manifest_path: str | Path, *, source_path: str | Path,
) -> Phase0ReferenceArtifact:
    """Admit a frozen reference only after its local content closes exactly.

    This validates a replay anchor; it does not launch the recorded runtime or
    imply that the semantic result may be promoted into memory.
    """
    path = Path(manifest_path)
    try:
        document = _load_unambiguous_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Phase-0 reference manifest") from error
    if not isinstance(document, dict) or document.get("schema") != "petta-memory-phase0-reference-artifact-v1":
        raise ValueError("invalid Phase-0 reference manifest schema")
    expected_members = {
        "schema",
        "determinism",
        "example",
        "runtime",
        "repositories",
        "result",
        "boundaries",
    }
    if document.keys() != expected_members:
        raise ValueError("Phase-0 reference manifest members do not match schema")

    def record(name: str, members: set[str]) -> dict[str, object]:
        value = document.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Phase-0 reference {name} must be an object")
        if value.keys() != members:
            raise ValueError(f"Phase-0 reference {name} members do not match schema")
        return value

    determinism = record("determinism", {"run1_sha256", "run2_sha256", "identical"})
    example = record("example", {"name", "source_sha256"})
    runtime = record("runtime", {"metta_binary_sha256"})
    repositories = record("repositories", {"patham9-pln"})
    result = record("result", {
        "passed",
        "semantic_result",
        "query_target",
        "passed_marker",
        "output_sha256",
        "output_bytes",
        "output_file",
    })
    boundaries = record("boundaries", {
        "no_memory_write",
        "no_inferred_belief_promotion",
        "no_live_omegaclaw_goalchainer_integration",
        "no_pettachainer_compileadd",
    })
    patham9 = repositories.get("patham9-pln")
    if not isinstance(patham9, dict) or patham9.keys() != {"commit"}:
        raise ValueError("Phase-0 reference must identify patham9-pln")

    digest_fields = {
        "source_sha256": example.get("source_sha256"),
        "runtime_executable_sha256": runtime.get("metta_binary_sha256"),
        "run1_sha256": determinism.get("run1_sha256"),
        "run2_sha256": determinism.get("run2_sha256"),
        "output_sha256": result.get("output_sha256"),
    }
    for field, value in digest_fields.items():
        _sha256_digest(value, f"Phase-0 reference {field}")
    output_digest = digest_fields["output_sha256"]
    if determinism.get("identical") is not True or {
        digest_fields["run1_sha256"], digest_fields["run2_sha256"], output_digest,
    } != {output_digest}:
        raise ValueError("Phase-0 reference determinism hashes do not close")

    output_name = result.get("output_file")
    if not isinstance(output_name, str) or not output_name or Path(output_name).name != output_name:
        raise ValueError("Phase-0 reference output_file must be one local filename")
    try:
        output = (path.parent / output_name).read_bytes()
        source = Path(source_path).read_bytes()
    except OSError as error:
        raise ValueError("Phase-0 reference content file is unavailable") from error
    if sha256(output).hexdigest() != output_digest:
        raise ValueError("Phase-0 reference output checksum mismatch")
    output_bytes = result.get("output_bytes")
    if isinstance(output_bytes, bool) or not isinstance(output_bytes, int) or output_bytes < 1:
        raise ValueError("Phase-0 reference output_bytes must be a positive integer")
    if len(output) != output_bytes:
        raise ValueError("Phase-0 reference output byte count mismatch")
    if sha256(source).hexdigest() != digest_fields["source_sha256"]:
        raise ValueError("Phase-0 reference source checksum mismatch")
    try:
        source_text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Phase-0 reference source must be valid UTF-8") from error

    semantic_result = result.get("semantic_result")
    query_target = result.get("query_target")
    if (
        not isinstance(semantic_result, str)
        or len(semantic_result) > DEFAULT_MAX_KERNEL_RESULT_CHARS
    ):
        raise ValueError("Phase-0 reference semantic_result must be bounded")
    try:
        parsed_result = parse_one_list(semantic_result)
    except SExpressionSyntaxError as error:
        raise ValueError(
            "Phase-0 reference semantic_result must be one valid result atom"
        ) from error
    if (
        to_source(parsed_result) != semantic_result
        or len(parsed_result) != 2
        or not isinstance(parsed_result[0], tuple)
        or len(parsed_result[0]) != 3
        or symbol_text(parsed_result[0][0]) != "stv"
        or not isinstance(parsed_result[1], tuple)
        or not parsed_result[1]
    ):
        raise ValueError(
            "Phase-0 reference semantic_result must be one canonical kernel result"
        )
    numeric_tokens = (
        symbol_text(parsed_result[0][1]), symbol_text(parsed_result[0][2]),
    )
    try:
        truth_values = tuple(float(value) for value in numeric_tokens)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Phase-0 reference semantic_result truth values must be numeric"
        ) from error
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in truth_values):
        raise ValueError(
            "Phase-0 reference semantic_result truth values must be finite and in [0, 1]"
        )
    stamps: list[int] = []
    for expression in parsed_result[1]:
        token = symbol_text(expression)
        try:
            stamp = int(token) if token is not None else -1
        except ValueError as error:
            raise ValueError(
                "Phase-0 reference semantic_result stamps must be canonical"
            ) from error
        if token != str(stamp) or stamp < 0:
            raise ValueError(
                "Phase-0 reference semantic_result stamps must be canonical"
            )
        stamps.append(stamp)
    if tuple(sorted(set(stamps))) != tuple(stamps):
        raise ValueError(
            "Phase-0 reference semantic_result stamps must be unique and sorted"
        )
    if not isinstance(query_target, str) or _canonical_kernel_term(query_target) != query_target:
        raise ValueError("Phase-0 reference query_target must be canonical declarative data")
    try:
        output_text = output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Phase-0 reference output must be valid UTF-8") from error
    if result.get("passed") is not True or result.get("passed_marker") != "#t":
        raise ValueError("Phase-0 reference must record one passing semantic marker")
    output_lines = tuple(line.strip() for line in output_text.splitlines() if line.strip())
    expected_output_lines = (
        f"[{semantic_result}]",
        "[((Passed: #t))]",
    )
    if output_lines.count(expected_output_lines[0]) != 1:
        raise ValueError(
            "Phase-0 reference output must contain exactly one semantic result as a standalone line"
        )
    if output_lines.count(expected_output_lines[1]) != 1:
        raise ValueError(
            "Phase-0 reference output must contain exactly one passing semantic marker as a standalone line"
        )
    if output_lines != expected_output_lines:
        raise ValueError(
            "Phase-0 reference output must contain only the ordered producer result and pass lines"
        )
    for field in (
        "no_memory_write", "no_inferred_belief_promotion",
        "no_live_omegaclaw_goalchainer_integration", "no_pettachainer_compileadd",
    ):
        if boundaries.get(field) is not True:
            raise ValueError(f"Phase-0 reference boundary {field} must be true")

    kernel_commit = patham9.get("commit")
    if not isinstance(kernel_commit, str) or len(kernel_commit) != 40 or any(
        character not in "0123456789abcdef" for character in kernel_commit
    ):
        raise ValueError("Phase-0 reference patham9 commit must be a full lowercase Git commit")
    example_name = example.get("name")
    if not isinstance(example_name, str) or not example_name:
        raise ValueError("Phase-0 reference example name must be non-empty")
    return Phase0ReferenceArtifact(
        example_name=example_name,
        source_sha256=digest_fields["source_sha256"],
        source_cid=_canonical_hash({
            "complete_program": source_text,
        }),
        runtime_executable_sha256=digest_fields["runtime_executable_sha256"],
        kernel_commit=kernel_commit,
        semantic_result=semantic_result,
        query_target=query_target,
        output_sha256=output_digest,
        output_bytes=output_bytes,
    )


@dataclass(frozen=True)
class EvidenceToken:
    id: str
    namespace: str
    source_id: str
    context_id: str
    observed_at: str
    minted_at: str
    source_event_id: str | None = None
    causal_group_id: str | None = None
    privacy_label: str = "local-private"
    payload_cid: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in ("id", "namespace", "source_id", "context_id", "observed_at", "minted_at", "privacy_label"):
            _nonempty(getattr(self, field), field)
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")


@dataclass(frozen=True)
class EvidencePacket:
    id: str
    statement: str
    context_id: str
    positive_delta: float
    negative_delta: float
    token_ids: tuple[str, ...]
    source_reliability: float
    temporal_relevance: float
    status: PacketStatus
    assumption_fingerprint: str
    ontology_fingerprint: str
    created_by: PacketOrigin
    parent_packet_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in ("id", "statement", "context_id", "assumption_fingerprint", "ontology_fingerprint"):
            _nonempty(getattr(self, field), field)
        _finite_nonnegative(self.positive_delta, "positive_delta")
        _finite_nonnegative(self.negative_delta, "negative_delta")
        for field in ("source_reliability", "temporal_relevance"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be finite and in [0, 1]")
        if self.status not in ("ACTIVE", "QUARANTINED", "RETRACTED"):
            raise ValueError("invalid packet status")
        if self.created_by not in ("OBSERVATION", "IMPORT", "REVIEWED_EXPORT"):
            raise ValueError("invalid packet origin")
        if not self.token_ids or tuple(sorted(set(self.token_ids))) != self.token_ids:
            raise ValueError("token_ids must be non-empty, unique, and sorted")
        if tuple(sorted(set(self.parent_packet_ids))) != self.parent_packet_ids:
            raise ValueError("parent_packet_ids must be unique and sorted")
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")

    @property
    def provenance_digest(self) -> str:
        return _canonical_hash(self.token_ids)


@dataclass(frozen=True)
class EvidenceBasis:
    basis_id: str
    member_token_ids: tuple[str, ...]
    causal_group_ids: tuple[str, ...]
    independence_status: IndependenceStatus
    justification_cid: str

    def __post_init__(self) -> None:
        _nonempty(self.basis_id, "basis_id")
        _nonempty(self.justification_cid, "justification_cid")
        if not self.member_token_ids or tuple(sorted(set(self.member_token_ids))) != self.member_token_ids:
            raise ValueError("member_token_ids must be non-empty, unique, and sorted")
        if tuple(sorted(set(self.causal_group_ids))) != self.causal_group_ids:
            raise ValueError("causal_group_ids must be unique and sorted")
        if self.independence_status not in ("PROVEN_DISJOINT", "ASSUMED", "COUPLED", "UNKNOWN"):
            raise ValueError("invalid independence_status")


@dataclass(frozen=True)
class StampMapEntry:
    episode_id: str
    stamp_int: int
    basis_id: str
    member_token_digest: str

    def __post_init__(self) -> None:
        _nonempty(self.episode_id, "episode_id")
        _nonempty(self.basis_id, "basis_id")
        if isinstance(self.stamp_int, bool) or not isinstance(self.stamp_int, int) or self.stamp_int < 0:
            raise ValueError("stamp_int must be a non-negative integer")
        _sha256_digest(self.member_token_digest, "member_token_digest")


@dataclass(frozen=True)
class KernelSentenceMeta:
    """Immutable provenance sidecar for one chart-compiled kernel sentence."""

    episode_id: str
    sentence_digest: str
    canonical_term: str
    projection_id: str
    context_id: str
    chart_id: str
    stamp_ints: tuple[int, ...]
    evidence_basis_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "episode_id", "sentence_digest", "canonical_term", "projection_id",
            "context_id", "chart_id",
        ):
            _nonempty(getattr(self, field), field)
        _sha256_digest(self.sentence_digest, "sentence_digest")
        _sha256_digest(self.projection_id, "projection_id")
        if tuple(sorted(set(self.stamp_ints))) != self.stamp_ints:
            raise ValueError("stamp_ints must be unique and sorted")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.stamp_ints):
            raise ValueError("stamp_ints must contain non-negative integers")
        if not self.evidence_basis_ids or tuple(sorted(set(self.evidence_basis_ids))) != self.evidence_basis_ids:
            raise ValueError("evidence_basis_ids must be non-empty, unique, and sorted")


@dataclass(frozen=True)
class CompiledSentence:
    """A patham9 sentence plus its non-lossy πPLN provenance sidecar."""

    atom: str
    projection: "ProjectionRecord"
    meta: KernelSentenceMeta

    def __post_init__(self) -> None:
        _nonempty(self.atom, "atom")
        canonical_term = _canonical_kernel_term(self.meta.canonical_term)
        if canonical_term != self.meta.canonical_term:
            raise ValueError("compiled sentence canonical_term is not canonical")
        expected_atom = (
            f"(Sentence ({canonical_term} (stv {self.projection.strength} "
            f"{self.projection.confidence})) "
            f"({' '.join(str(value) for value in self.meta.stamp_ints)}))"
        )
        if self.atom != expected_atom:
            raise ValueError("compiled sentence atom does not match typed metadata")
        if self.meta.sentence_digest != _canonical_hash({"atom": self.atom}):
            raise ValueError("compiled sentence digest does not match atom")


@dataclass(frozen=True)
class CompiledEpisodeInputs:
    """Pure deterministic output of the first Phase-2 compilation boundary."""

    episode_id: str
    chart_fingerprint: str
    evidence_snapshot_fingerprint: str
    stamp_map: tuple[StampMapEntry, ...]
    sentences: tuple[CompiledSentence, ...]

    def __post_init__(self) -> None:
        for field in ("episode_id", "chart_fingerprint", "evidence_snapshot_fingerprint"):
            _nonempty(getattr(self, field), field)
        _sha256_digest(self.chart_fingerprint, "chart_fingerprint")
        _sha256_digest(self.evidence_snapshot_fingerprint, "evidence_snapshot_fingerprint")
        if not self.sentences:
            raise ValueError("compiled episode must contain at least one sentence")
        basis_by_stamp: dict[int, str] = {}
        for entry in self.stamp_map:
            if entry.episode_id != self.episode_id:
                raise ValueError("stamp map episode mismatch")
            if entry.stamp_int in basis_by_stamp or entry.basis_id in basis_by_stamp.values():
                raise ValueError("stamp map must contain unique stamps and basis ids")
            basis_by_stamp[entry.stamp_int] = entry.basis_id
        if tuple(sorted(basis_by_stamp)) != tuple(range(len(basis_by_stamp))):
            raise ValueError("stamp map integers must be contiguous from zero")
        sentence_digests: set[str] = set()
        for sentence in self.sentences:
            if sentence.meta.episode_id != self.episode_id:
                raise ValueError("compiled sentence episode mismatch")
            if sentence.meta.sentence_digest in sentence_digests:
                raise ValueError("compiled sentence digests must be unique")
            sentence_digests.add(sentence.meta.sentence_digest)
            mapped_bases = tuple(basis_by_stamp.get(stamp) for stamp in sentence.meta.stamp_ints)
            if None in mapped_bases or mapped_bases != sentence.meta.evidence_basis_ids:
                raise ValueError("compiled sentence stamps do not match evidence bases")


@dataclass(frozen=True)
class PeTTaChainerInputStatement:
    """One compiler sentence adapted to PeTTaChainer's checked add shape."""

    atom: str
    proof_id: str
    sentence_digest: str
    canonical_term: str
    strength: float
    confidence: float
    stamp_ints: tuple[int, ...]
    evidence_basis_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256_digest(self.sentence_digest, "sentence_digest")
        if self.proof_id != f"pm-{self.sentence_digest}":
            raise ValueError("PeTTaChainer proof id must derive from the complete sentence digest")
        canonical_term = _canonical_kernel_term(self.canonical_term)
        if canonical_term != self.canonical_term:
            raise ValueError("PeTTaChainer statement term is not canonical")
        for value, field in ((self.strength, "strength"), (self.confidence, "confidence")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"PeTTaChainer statement {field} must be finite and in [0, 1]")
        expected_atom = f"(: {self.proof_id} {canonical_term} (STV {self.strength} {self.confidence}))"
        if self.atom != expected_atom:
            raise ValueError("PeTTaChainer statement atom does not match typed content")
        if tuple(sorted(set(self.stamp_ints))) != self.stamp_ints or not self.stamp_ints:
            raise ValueError("PeTTaChainer statement stamps must be non-empty, unique, and sorted")
        if tuple(sorted(set(self.evidence_basis_ids))) != self.evidence_basis_ids or not self.evidence_basis_ids:
            raise ValueError("PeTTaChainer statement evidence bases must be non-empty, unique, and sorted")


@dataclass(frozen=True)
class PeTTaChainerEpisodeContract:
    """Inert checked-add/query contract for one immutable compiled episode.

    PeTTaChainer is not the stock patham9 ``Sentence`` evaluator: it accepts
    ``(: proof term (STV s c))`` through ``compileadd`` and queries with a
    variable proof/truth-value pair.  This record makes that schema boundary
    explicit while retaining patham9 stamps and evidence bases as audit-only
    sidecars.  It does not invoke ``compileadd`` or authorize promotion.
    """

    episode_id: str
    chart_fingerprint: str
    statements: tuple[PeTTaChainerInputStatement, ...]
    query_term: str
    query_atom: str

    def __post_init__(self) -> None:
        _nonempty(self.episode_id, "episode_id")
        _sha256_digest(self.chart_fingerprint, "chart_fingerprint")
        if not self.statements:
            raise ValueError("PeTTaChainer contract must contain at least one statement")
        proof_ids = tuple(statement.proof_id for statement in self.statements)
        if len(set(proof_ids)) != len(proof_ids):
            raise ValueError("PeTTaChainer contract proof ids must be unique")
        canonical_query = _canonical_kernel_term(self.query_term)
        if canonical_query != self.query_term:
            raise ValueError("PeTTaChainer query term is not canonical")
        if self.query_atom != f"(: $prf {canonical_query} $tv)":
            raise ValueError("PeTTaChainer query atom does not match typed content")


@dataclass(frozen=True)
class PeTTaChainerStageCapture:
    """Content identity for one bounded isolated PeTTaChainer stage.

    The profiling runner intentionally retains hashes and byte counts rather
    than hundreds of kilobytes of diagnostic output.  This record makes that
    limitation explicit: it is process provenance, not semantic output.
    """

    label: str
    elapsed_seconds: float
    stdout_bytes: int
    stdout_sha256: str
    stderr_bytes: int
    stderr_sha256: str
    capture_digest: str

    def __post_init__(self) -> None:
        _nonempty(self.label, "PeTTaChainer stage label")
        _finite_nonnegative(self.elapsed_seconds, "PeTTaChainer stage elapsed_seconds")
        for field in ("stdout_bytes", "stderr_bytes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        _sha256_digest(self.stdout_sha256, "stdout_sha256")
        _sha256_digest(self.stderr_sha256, "stderr_sha256")
        expected = _canonical_hash({
            "label": self.label,
            "elapsed_seconds": float(self.elapsed_seconds),
            "stdout_bytes": self.stdout_bytes,
            "stdout_sha256": self.stdout_sha256,
            "stderr_bytes": self.stderr_bytes,
            "stderr_sha256": self.stderr_sha256,
        })
        if self.capture_digest != expected:
            raise ValueError("PeTTaChainer stage capture digest does not match typed content")


def build_pettachainer_stage_capture(
    *, label: str, elapsed_seconds: float, stdout_bytes: int,
    stdout_sha256: str, stderr_bytes: int, stderr_sha256: str,
) -> PeTTaChainerStageCapture:
    _finite_nonnegative(elapsed_seconds, "PeTTaChainer stage elapsed_seconds")
    payload = {
        "label": label,
        "elapsed_seconds": float(elapsed_seconds),
        "stdout_bytes": stdout_bytes,
        "stdout_sha256": stdout_sha256,
        "stderr_bytes": stderr_bytes,
        "stderr_sha256": stderr_sha256,
    }
    return PeTTaChainerStageCapture(**payload, capture_digest=_canonical_hash(payload))


@dataclass(frozen=True)
class PeTTaChainerDerivedResultCapture:
    """Typed, non-promoting capture of one compiler-bound derived result."""

    episode_id: str
    chart_fingerprint: str
    fact_sentence_digest: str
    rule_sentence_digest: str
    fact_proof_id: str
    rule_proof_id: str
    query_term: str
    derived_atom: str
    derived_proof: str
    strength: float
    confidence: float
    fact_stamp_ints: tuple[int, ...]
    rule_stamp_ints: tuple[int, ...]
    fact_evidence_basis_ids: tuple[str, ...]
    rule_evidence_basis_ids: tuple[str, ...]
    validator_capture: PeTTaChainerStageCapture
    runtime_capture: PeTTaChainerStageCapture
    result_digest: str

    def __post_init__(self) -> None:
        _nonempty(self.episode_id, "episode_id")
        _sha256_digest(self.chart_fingerprint, "chart_fingerprint")
        _sha256_digest(self.fact_sentence_digest, "fact_sentence_digest")
        _sha256_digest(self.rule_sentence_digest, "rule_sentence_digest")
        if self.fact_proof_id != f"pm-{self.fact_sentence_digest}":
            raise ValueError("fact proof id does not match its sentence digest")
        if self.rule_proof_id != f"pm-{self.rule_sentence_digest}":
            raise ValueError("rule proof id does not match its sentence digest")
        if self.fact_sentence_digest == self.rule_sentence_digest:
            raise ValueError("fact and rule sentence digests must be distinct")
        if self.fact_proof_id == self.rule_proof_id:
            raise ValueError("fact and rule proof ids must be distinct")
        if _canonical_kernel_term(self.query_term) != self.query_term:
            raise ValueError("PeTTaChainer derived query term is not canonical")
        expected_proof = f"(rule-proof {self.rule_proof_id} {self.fact_proof_id})"
        if self.derived_proof != expected_proof:
            raise ValueError("derived proof does not match compiler input proof ids")
        for value, field in ((self.strength, "strength"), (self.confidence, "confidence")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"derived result {field} must be finite and in [0, 1]")
        expected_atom = f"(: {self.derived_proof} {self.query_term} (STV {self.strength} {self.confidence}))"
        if self.derived_atom != expected_atom:
            raise ValueError("derived atom does not match typed result content")
        for stamps, bases, label in (
            (self.fact_stamp_ints, self.fact_evidence_basis_ids, "fact"),
            (self.rule_stamp_ints, self.rule_evidence_basis_ids, "rule"),
        ):
            if not isinstance(stamps, tuple) or not stamps or tuple(sorted(set(stamps))) != stamps:
                raise ValueError(f"{label} stamps must be a non-empty sorted unique tuple")
            if any(isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0 for stamp in stamps):
                raise ValueError(f"{label} stamps must be non-negative integers")
            if not isinstance(bases, tuple) or not bases or tuple(sorted(set(bases))) != bases:
                raise ValueError(f"{label} evidence bases must be a non-empty sorted unique tuple")
            if any(not isinstance(basis, str) or not basis.strip() for basis in bases):
                raise ValueError(f"{label} evidence bases must be non-empty strings")
            if len(bases) != len(stamps):
                raise ValueError(f"{label} evidence bases must close every stamp")
        if set(self.fact_stamp_ints) & set(self.rule_stamp_ints):
            raise ValueError("fact and rule stamps must be disjoint")
        if set(self.fact_evidence_basis_ids) & set(self.rule_evidence_basis_ids):
            raise ValueError("fact and rule evidence bases must be disjoint")
        if not isinstance(self.validator_capture, PeTTaChainerStageCapture) or not isinstance(self.runtime_capture, PeTTaChainerStageCapture):
            raise ValueError("derived result requires typed validator and runtime captures")
        if self.validator_capture.label != "validate_repaired_one_rule_derivation":
            raise ValueError("derived result validator capture has the wrong stage label")
        if self.runtime_capture.label != "repaired_one_rule_derivation":
            raise ValueError("derived result runtime capture has the wrong stage label")
        expected_digest = _canonical_hash(_pettachainer_derived_result_payload(self))
        if self.result_digest != expected_digest:
            raise ValueError("PeTTaChainer derived result digest does not match typed content")


@dataclass(frozen=True)
class PeTTaChainerRuleAttribution:
    """Compiler-bound attribution for the admitted one-rule derivation.

    This identifies the sole rule and fact permitted by the episode contract.
    It deliberately does not claim that opaque runtime diagnostics constitute
    a decoded execution trace.
    """

    result_digest: str
    inference_rule: str
    rule_sentence_digest: str
    rule_proof_id: str
    rule_stamp_ints: tuple[int, ...]
    rule_evidence_basis_ids: tuple[str, ...]
    fact_sentence_digest: str
    fact_proof_id: str
    fact_stamp_ints: tuple[int, ...]
    fact_evidence_basis_ids: tuple[str, ...]
    attribution_kind: str
    runtime_trace_decoded: bool
    attribution_digest: str

    def __post_init__(self) -> None:
        _sha256_digest(self.result_digest, "result_digest")
        if self.inference_rule != "TotalMP":
            raise ValueError("PeTTaChainer attribution requires the admitted TotalMP rule")
        if self.attribution_kind != "compiler-bound-single-rule":
            raise ValueError("PeTTaChainer attribution kind is not compiler-bound-single-rule")
        if self.runtime_trace_decoded is not False:
            raise ValueError("PeTTaChainer attribution cannot claim a decoded runtime trace")
        for digest, proof_id, label in (
            (self.rule_sentence_digest, self.rule_proof_id, "rule"),
            (self.fact_sentence_digest, self.fact_proof_id, "fact"),
        ):
            _sha256_digest(digest, f"{label}_sentence_digest")
            if proof_id != f"pm-{digest}":
                raise ValueError(f"{label} proof id does not match sentence digest")
        if self.fact_sentence_digest == self.rule_sentence_digest:
            raise ValueError("fact and rule attribution sentence digests must be distinct")
        if self.fact_proof_id == self.rule_proof_id:
            raise ValueError("fact and rule attribution proof ids must be distinct")
        for stamps, bases, label in (
            (self.rule_stamp_ints, self.rule_evidence_basis_ids, "rule"),
            (self.fact_stamp_ints, self.fact_evidence_basis_ids, "fact"),
        ):
            if not isinstance(stamps, tuple) or not stamps or tuple(sorted(set(stamps))) != stamps:
                raise ValueError(f"{label} attribution stamps must be a non-empty sorted unique tuple")
            if any(isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0 for stamp in stamps):
                raise ValueError(f"{label} attribution stamps must be non-negative integers")
            if not isinstance(bases, tuple) or not bases or tuple(sorted(set(bases))) != bases:
                raise ValueError(f"{label} attribution evidence bases must be a non-empty sorted unique tuple")
            if any(not isinstance(basis, str) or not basis.strip() for basis in bases):
                raise ValueError(f"{label} attribution evidence bases must be non-empty strings")
            if len(bases) != len(stamps):
                raise ValueError(f"{label} attribution evidence bases must close every stamp")
        if set(self.fact_stamp_ints) & set(self.rule_stamp_ints):
            raise ValueError("fact and rule attribution stamps must be disjoint")
        if set(self.fact_evidence_basis_ids) & set(self.rule_evidence_basis_ids):
            raise ValueError("fact and rule attribution evidence bases must be disjoint")
        expected = _canonical_hash({
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "attribution_digest"
        })
        if self.attribution_digest != expected:
            raise ValueError("PeTTaChainer attribution digest does not match typed content")


def build_pettachainer_rule_attribution(
    result: PeTTaChainerDerivedResultCapture,
) -> PeTTaChainerRuleAttribution:
    """Attribute the admitted result to its sole compiler-bound TotalMP rule."""
    if not isinstance(result, PeTTaChainerDerivedResultCapture):
        raise ValueError("rule attribution requires a typed PeTTaChainer derived result")
    values = {
        "result_digest": result.result_digest,
        "inference_rule": "TotalMP",
        "rule_sentence_digest": result.rule_sentence_digest,
        "rule_proof_id": result.rule_proof_id,
        "rule_stamp_ints": result.rule_stamp_ints,
        "rule_evidence_basis_ids": result.rule_evidence_basis_ids,
        "fact_sentence_digest": result.fact_sentence_digest,
        "fact_proof_id": result.fact_proof_id,
        "fact_stamp_ints": result.fact_stamp_ints,
        "fact_evidence_basis_ids": result.fact_evidence_basis_ids,
        "attribution_kind": "compiler-bound-single-rule",
        "runtime_trace_decoded": False,
    }
    return PeTTaChainerRuleAttribution(
        **values, attribution_digest=_canonical_hash(values),
    )


def pettachainer_rule_attribution_document(
    attribution: PeTTaChainerRuleAttribution,
) -> dict[str, object]:
    """Return a checksummed persistence envelope for compiler-bound attribution."""
    if not isinstance(attribution, PeTTaChainerRuleAttribution):
        raise ValueError("rule attribution document requires typed attribution")
    payload = {
        field: getattr(attribution, field)
        for field in attribution.__dataclass_fields__
    }
    for field in (
        "rule_stamp_ints", "rule_evidence_basis_ids",
        "fact_stamp_ints", "fact_evidence_basis_ids",
    ):
        payload[field] = list(payload[field])
    return {
        "schema": "petta-memory-pettachainer-rule-attribution-v1",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_pettachainer_rule_attribution(
    path: str | Path,
    attribution: PeTTaChainerRuleAttribution,
) -> None:
    """Create one immutable rule-attribution artifact; never replace it."""
    data = json.dumps(
        pettachainer_rule_attribution_document(attribution),
        sort_keys=True,
        indent=2,
    ) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_create_once_durable(destination, data)


def read_pettachainer_rule_attribution(
    path: str | Path,
    *,
    result: PeTTaChainerDerivedResultCapture,
) -> PeTTaChainerRuleAttribution:
    """Reload attribution and close it against one admitted derived capture."""
    if not isinstance(result, PeTTaChainerDerivedResultCapture):
        raise ValueError("rule attribution reload requires a typed derived result")
    document = _load_unambiguous_json(path)
    schema = "petta-memory-pettachainer-rule-attribution-v1"
    if (not isinstance(document, dict)
            or set(document) != {"schema", "payload", "document_digest"}
            or document.get("schema") != schema):
        raise ValueError("invalid PeTTaChainer rule attribution document schema")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("document_digest") != _canonical_hash(payload):
        raise ValueError("PeTTaChainer rule attribution document checksum mismatch")
    expected_fields = set(PeTTaChainerRuleAttribution.__dataclass_fields__)
    list_fields = {
        "rule_stamp_ints", "rule_evidence_basis_ids",
        "fact_stamp_ints", "fact_evidence_basis_ids",
    }
    if (set(payload) != expected_fields
            or any(not isinstance(payload[field], list) for field in list_fields)):
        raise ValueError("invalid PeTTaChainer rule attribution payload")
    values = dict(payload)
    for field in list_fields:
        values[field] = tuple(payload[field])
    attribution = PeTTaChainerRuleAttribution(**values)
    expected = build_pettachainer_rule_attribution(result)
    if attribution != expected:
        raise ValueError("PeTTaChainer rule attribution does not match derived result")
    return attribution


def _pettachainer_derived_result_payload(result: PeTTaChainerDerivedResultCapture) -> dict[str, object]:
    return {
        field: getattr(result, field)
        for field in result.__dataclass_fields__
        if field not in {"result_digest", "validator_capture", "runtime_capture"}
    } | {
        "validator_capture_digest": result.validator_capture.capture_digest,
        "runtime_capture_digest": result.runtime_capture.capture_digest,
    }


def build_pettachainer_derived_result_capture(
    *, episode_id: str, chart_fingerprint: str,
    fact: PeTTaChainerInputStatement, rule: PeTTaChainerInputStatement,
    query_term: str, derived_atom: str, derived_proof: str,
    strength: float, confidence: float,
    validator_capture: PeTTaChainerStageCapture,
    runtime_capture: PeTTaChainerStageCapture,
) -> PeTTaChainerDerivedResultCapture:
    for value, field in ((strength, "strength"), (confidence, "confidence")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"derived result {field} must be numeric")
    values = {
        "episode_id": episode_id,
        "chart_fingerprint": chart_fingerprint,
        "fact_sentence_digest": fact.sentence_digest,
        "rule_sentence_digest": rule.sentence_digest,
        "fact_proof_id": fact.proof_id,
        "rule_proof_id": rule.proof_id,
        "query_term": query_term,
        "derived_atom": derived_atom,
        "derived_proof": derived_proof,
        "strength": float(strength),
        "confidence": float(confidence),
        "fact_stamp_ints": fact.stamp_ints,
        "rule_stamp_ints": rule.stamp_ints,
        "fact_evidence_basis_ids": fact.evidence_basis_ids,
        "rule_evidence_basis_ids": rule.evidence_basis_ids,
        "validator_capture": validator_capture,
        "runtime_capture": runtime_capture,
    }
    digest_payload = {
        key: value for key, value in values.items()
        if key not in {"validator_capture", "runtime_capture"}
    } | {
        "validator_capture_digest": validator_capture.capture_digest,
        "runtime_capture_digest": runtime_capture.capture_digest,
    }
    return PeTTaChainerDerivedResultCapture(
        **values, result_digest=_canonical_hash(digest_payload),
    )


def pettachainer_derived_result_capture_document(
    result: PeTTaChainerDerivedResultCapture,
) -> dict[str, object]:
    """Return a checksummed persistence envelope for one derived capture."""
    if not isinstance(result, PeTTaChainerDerivedResultCapture):
        raise ValueError("derived result document requires a typed PeTTaChainer capture")
    payload = {
        field: getattr(result, field)
        for field in result.__dataclass_fields__
        if field not in {"validator_capture", "runtime_capture"}
    }
    payload["fact_stamp_ints"] = list(result.fact_stamp_ints)
    payload["rule_stamp_ints"] = list(result.rule_stamp_ints)
    payload["fact_evidence_basis_ids"] = list(result.fact_evidence_basis_ids)
    payload["rule_evidence_basis_ids"] = list(result.rule_evidence_basis_ids)
    for field in ("validator_capture", "runtime_capture"):
        capture = getattr(result, field)
        payload[field] = {
            capture_field: getattr(capture, capture_field)
            for capture_field in capture.__dataclass_fields__
        }
    return {
        "schema": "petta-memory-pettachainer-derived-result-capture-v1",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_pettachainer_derived_result_capture(
    path: str | Path,
    result: PeTTaChainerDerivedResultCapture,
) -> None:
    """Create one immutable derived-result capture artifact; never replace it."""
    data = json.dumps(
        pettachainer_derived_result_capture_document(result), sort_keys=True, indent=2,
    ) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_create_once_durable(destination, data)


def _write_create_once_durable(destination: Path, data: str) -> None:
    """Create and durably publish one artifact without replacing an existing path."""
    def record_cleanup_failure(error: BaseException, note: str) -> None:
        add_note = getattr(error, "add_note", None)
        if add_note is not None:
            add_note(note)
        else:
            # Python 3.10 lacks BaseException.add_note(), but retaining the
            # same attribute keeps cleanup diagnostics available to callers.
            error.__notes__ = [*getattr(error, "__notes__", ()), note]

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(destination.parent, directory_flags)
    file_created = False
    file_synced = False
    publication_error: BaseException | None = None
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if parent_metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(
                "artifact publication parent must not be group- or world-writable"
            )
        if parent_metadata.st_uid != os.geteuid():
            raise ValueError(
                "artifact publication parent must be owned by the current user"
            )
        descriptor = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
        file_created = True
        try:
            handle = os.fdopen(descriptor, "w", encoding="utf-8")
        except BaseException as stream_open_error:
            try:
                os.close(descriptor)
            except OSError as close_error:
                record_cleanup_failure(
                    stream_open_error,
                    f"artifact descriptor close failed: {close_error}",
                )
            raise
        stream_error: BaseException | None = None
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            file_synced = True
        except BaseException as caught_error:
            stream_error = caught_error
            raise
        finally:
            try:
                handle.close()
            except OSError as close_error:
                if stream_error is None:
                    raise
                record_cleanup_failure(
                    stream_error,
                    f"artifact stream close failed: {close_error}",
                )
        final_parent_metadata = os.fstat(parent_descriptor)
        stable_parent_fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
        )
        if any(
            getattr(parent_metadata, field) != getattr(final_parent_metadata, field)
            for field in stable_parent_fields
        ):
            raise ValueError(
                "artifact publication parent changed during publication"
            )
        os.fsync(parent_descriptor)
    except BaseException as caught_error:
        publication_error = caught_error
        # Before the completed file is synced, a partial artifact is safe to
        # remove.  After that point its directory entry may already survive a
        # crash even when the parent fsync reports failure; retaining it keeps
        # create-once semantics and prevents a later caller from overwriting
        # an artifact whose publication state is uncertain.
        if file_created and not file_synced:
            try:
                os.unlink(destination.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                record_cleanup_failure(
                    publication_error,
                    f"partial artifact cleanup failed: {cleanup_error}"
                )
            else:
                # Make cleanup of a partially written directory entry durable
                # before reporting the original write/sync failure.  Without
                # this sync, a crash could resurrect the rejected artifact.
                try:
                    os.fsync(parent_descriptor)
                except OSError as cleanup_error:
                    record_cleanup_failure(
                        publication_error,
                        f"partial artifact cleanup sync failed: {cleanup_error}"
                    )
        raise
    finally:
        try:
            os.close(parent_descriptor)
        except OSError as close_error:
            if publication_error is None:
                raise
            record_cleanup_failure(
                publication_error,
                f"parent directory descriptor close failed: {close_error}",
            )


def read_pettachainer_derived_result_capture(
    path: str | Path,
    *,
    contract: PeTTaChainerEpisodeContract,
) -> PeTTaChainerDerivedResultCapture:
    """Reload a capture and close its compiler provenance against one contract."""
    if not isinstance(contract, PeTTaChainerEpisodeContract):
        raise ValueError("derived result reload requires a typed PeTTaChainer contract")
    document = _load_unambiguous_json(path)
    schema = "petta-memory-pettachainer-derived-result-capture-v1"
    if (not isinstance(document, dict)
            or set(document) != {"schema", "payload", "document_digest"}
            or document.get("schema") != schema):
        raise ValueError("invalid PeTTaChainer derived result capture document schema")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("document_digest") != _canonical_hash(payload):
        raise ValueError("PeTTaChainer derived result capture document checksum mismatch")
    expected_fields = set(PeTTaChainerDerivedResultCapture.__dataclass_fields__)
    list_fields = {
        "fact_stamp_ints", "rule_stamp_ints",
        "fact_evidence_basis_ids", "rule_evidence_basis_ids",
    }
    capture_fields = {"validator_capture", "runtime_capture"}
    if (set(payload) != expected_fields
            or any(not isinstance(payload[field], list) for field in list_fields)
            or any(not isinstance(payload[field], dict) for field in capture_fields)
            or any(set(payload[field]) != set(PeTTaChainerStageCapture.__dataclass_fields__)
                   for field in capture_fields)):
        raise ValueError("invalid PeTTaChainer derived result capture payload")
    if (payload["episode_id"] != contract.episode_id
            or payload["chart_fingerprint"] != contract.chart_fingerprint
            or payload["query_term"] != contract.query_term):
        raise ValueError("PeTTaChainer derived result capture does not match episode contract")
    statements = {statement.sentence_digest: statement for statement in contract.statements}
    fact = statements.get(payload["fact_sentence_digest"])
    rule = statements.get(payload["rule_sentence_digest"])
    if (fact is None or rule is None
            or payload["fact_proof_id"] != fact.proof_id
            or payload["rule_proof_id"] != rule.proof_id
            or tuple(payload["fact_stamp_ints"]) != fact.stamp_ints
            or tuple(payload["rule_stamp_ints"]) != rule.stamp_ints
            or tuple(payload["fact_evidence_basis_ids"]) != fact.evidence_basis_ids
            or tuple(payload["rule_evidence_basis_ids"]) != rule.evidence_basis_ids):
        raise ValueError("PeTTaChainer derived result capture compiler provenance mismatch")
    values = dict(payload)
    for field in list_fields:
        values[field] = tuple(payload[field])
    for field in capture_fields:
        values[field] = PeTTaChainerStageCapture(**payload[field])
    return PeTTaChainerDerivedResultCapture(**values)


@dataclass(frozen=True)
class PeTTaChainerEpisodeManifest:
    """Audit identity for one completed repaired PeTTaChainer episode.

    This is deliberately distinct from the stock patham9 ``EpisodeManifest``:
    the isolated PeTTaChainer runner retains content identities for its noisy
    streams rather than their raw contents, and a derived result retains fact
    and rule evidence sidecars separately.  The record is non-promoting.
    """

    episode_id: str
    chart_fingerprint: str
    contract_cid: str
    result_cid: str
    attribution_cid: str
    validator_capture_cid: str
    runtime_capture_cid: str
    kernel_name: str
    kernel_version: str
    kernel_capabilities_cid: str
    repair_profile_id: str
    repaired_source_cid: str
    controller_envelope_cid: str
    seed: int
    budget: "EpisodeBudget"
    started_at: str
    finished_at: str
    result_classification: str
    promotion_authorized: bool
    manifest_digest: str

    def __post_init__(self) -> None:
        for field in (
            "episode_id", "kernel_name", "kernel_version", "repair_profile_id",
            "result_classification",
        ):
            _nonempty(getattr(self, field), field)
        for field in (
            "chart_fingerprint", "contract_cid", "result_cid",
            "attribution_cid",
            "validator_capture_cid", "runtime_capture_cid",
            "kernel_capabilities_cid", "repaired_source_cid",
            "controller_envelope_cid", "manifest_digest",
        ):
            _sha256_digest(getattr(self, field), field)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.budget, EpisodeBudget):
            raise ValueError("budget must be an EpisodeBudget")
        if _timestamp(self.finished_at, "finished_at") < _timestamp(self.started_at, "started_at"):
            raise ValueError("finished_at must not precede started_at")
        if self.result_classification != "compiler-bound-one-rule-derived-result":
            raise ValueError("unsupported PeTTaChainer result classification")
        if self.promotion_authorized is not False:
            raise ValueError("PeTTaChainer episode manifests cannot authorize promotion")
        expected = _canonical_hash(_pettachainer_episode_manifest_payload(self, include_digest=False))
        if self.manifest_digest != expected:
            raise ValueError("PeTTaChainer manifest digest does not match typed content")


def _pettachainer_contract_payload(contract: PeTTaChainerEpisodeContract) -> dict[str, object]:
    return {
        "episode_id": contract.episode_id,
        "chart_fingerprint": contract.chart_fingerprint,
        "statements": [
            {
                "atom": statement.atom,
                "proof_id": statement.proof_id,
                "sentence_digest": statement.sentence_digest,
                "canonical_term": statement.canonical_term,
                "strength": statement.strength,
                "confidence": statement.confidence,
                "stamp_ints": list(statement.stamp_ints),
                "evidence_basis_ids": list(statement.evidence_basis_ids),
            }
            for statement in contract.statements
        ],
        "query_term": contract.query_term,
        "query_atom": contract.query_atom,
    }


def _pettachainer_episode_manifest_payload(
    manifest: PeTTaChainerEpisodeManifest, *, include_digest: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        field: getattr(manifest, field)
        for field in manifest.__dataclass_fields__
        if field not in {"budget", "manifest_digest"}
    }
    payload["budget"] = {
        field: getattr(manifest.budget, field)
        for field in manifest.budget.__dataclass_fields__
    }
    if include_digest:
        payload["manifest_digest"] = manifest.manifest_digest
    return payload


def build_pettachainer_episode_manifest(
    *, contract: PeTTaChainerEpisodeContract,
    result: PeTTaChainerDerivedResultCapture,
    attribution: PeTTaChainerRuleAttribution,
    kernel_name: str,
    kernel_version: str,
    kernel_capabilities_cid: str,
    repair_profile_id: str,
    repaired_source_cid: str,
    controller_envelope_cid: str,
    seed: int,
    budget: "EpisodeBudget",
    started_at: str,
    finished_at: str,
) -> PeTTaChainerEpisodeManifest:
    """Adapt one compiler-bound derived capture into a non-promoting manifest."""
    if not isinstance(contract, PeTTaChainerEpisodeContract):
        raise ValueError("contract must be an immutable PeTTaChainer episode contract")
    if not isinstance(result, PeTTaChainerDerivedResultCapture):
        raise ValueError("result must be a typed PeTTaChainer derived capture")
    if (not isinstance(attribution, PeTTaChainerRuleAttribution)
            or attribution.result_digest != result.result_digest
            or attribution != build_pettachainer_rule_attribution(result)):
        raise ValueError("attribution must close against the PeTTaChainer derived capture")
    statements = {statement.sentence_digest: statement for statement in contract.statements}
    fact = statements.get(result.fact_sentence_digest)
    rule = statements.get(result.rule_sentence_digest)
    if (
        result.episode_id != contract.episode_id
        or result.chart_fingerprint != contract.chart_fingerprint
        or result.query_term != contract.query_term
        or len(contract.statements) != 2
        or fact is None
        or rule is None
        or result.fact_proof_id != fact.proof_id
        or result.rule_proof_id != rule.proof_id
        or result.fact_stamp_ints != fact.stamp_ints
        or result.rule_stamp_ints != rule.stamp_ints
        or result.fact_evidence_basis_ids != fact.evidence_basis_ids
        or result.rule_evidence_basis_ids != rule.evidence_basis_ids
    ):
        raise ValueError("PeTTaChainer result does not close against the episode contract")
    values = {
        "episode_id": contract.episode_id,
        "chart_fingerprint": contract.chart_fingerprint,
        "contract_cid": _canonical_hash(_pettachainer_contract_payload(contract)),
        "result_cid": result.result_digest,
        "attribution_cid": attribution.attribution_digest,
        "validator_capture_cid": result.validator_capture.capture_digest,
        "runtime_capture_cid": result.runtime_capture.capture_digest,
        "kernel_name": kernel_name,
        "kernel_version": kernel_version,
        "kernel_capabilities_cid": kernel_capabilities_cid,
        "repair_profile_id": repair_profile_id,
        "repaired_source_cid": repaired_source_cid,
        "controller_envelope_cid": controller_envelope_cid,
        "seed": seed,
        "budget": budget,
        "started_at": started_at,
        "finished_at": finished_at,
        "result_classification": "compiler-bound-one-rule-derived-result",
        "promotion_authorized": False,
    }
    digest_payload = dict(values)
    digest_payload["budget"] = {
        field: getattr(budget, field) for field in budget.__dataclass_fields__
    }
    return PeTTaChainerEpisodeManifest(
        **values, manifest_digest=_canonical_hash(digest_payload),
    )


def pettachainer_episode_manifest_document(
    manifest: PeTTaChainerEpisodeManifest,
) -> dict[str, object]:
    """Return a checksummed document for one non-promoting manifest."""
    if not isinstance(manifest, PeTTaChainerEpisodeManifest):
        raise ValueError("manifest must be a typed PeTTaChainer episode manifest")
    payload = _pettachainer_episode_manifest_payload(manifest)
    return {
        "schema": "petta-memory-pettachainer-episode-manifest-v2",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_pettachainer_episode_manifest(
    path: str | Path, manifest: PeTTaChainerEpisodeManifest,
) -> None:
    """Create one immutable PeTTaChainer manifest artifact."""
    data = json.dumps(
        pettachainer_episode_manifest_document(manifest), sort_keys=True, indent=2,
    ) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_create_once_durable(destination, data)


def read_pettachainer_episode_manifest(
    path: str | Path, *, contract: PeTTaChainerEpisodeContract,
    result: PeTTaChainerDerivedResultCapture,
    attribution: PeTTaChainerRuleAttribution,
) -> PeTTaChainerEpisodeManifest:
    """Reload a checksummed manifest and close it against its typed inputs."""
    if not isinstance(contract, PeTTaChainerEpisodeContract):
        raise ValueError("contract must be an immutable PeTTaChainer episode contract")
    if not isinstance(result, PeTTaChainerDerivedResultCapture):
        raise ValueError("result must be a typed PeTTaChainer derived capture")
    if (not isinstance(attribution, PeTTaChainerRuleAttribution)
            or attribution.result_digest != result.result_digest
            or attribution != build_pettachainer_rule_attribution(result)):
        raise ValueError("attribution must close against the PeTTaChainer derived capture")
    document = _load_unambiguous_json(path)
    if (not isinstance(document, dict)
            or set(document) != {"schema", "payload", "document_digest"}
            or document.get("schema")
            != "petta-memory-pettachainer-episode-manifest-v2"):
        raise ValueError("invalid PeTTaChainer episode manifest document schema")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("document_digest") != _canonical_hash(payload):
        raise ValueError("PeTTaChainer episode manifest document checksum mismatch")
    expected_fields = set(PeTTaChainerEpisodeManifest.__dataclass_fields__) - {"budget"}
    expected_fields.add("budget")
    if (set(payload) != expected_fields
            or not isinstance(payload["budget"], dict)
            or set(payload["budget"]) != set(EpisodeBudget.__dataclass_fields__)):
        raise ValueError("invalid PeTTaChainer episode manifest payload")
    expected_contract_cid = _canonical_hash(_pettachainer_contract_payload(contract))
    if (payload["episode_id"] != contract.episode_id
            or payload["chart_fingerprint"] != contract.chart_fingerprint
            or payload["contract_cid"] != expected_contract_cid
            or payload["result_cid"] != result.result_digest
            or payload["attribution_cid"] != attribution.attribution_digest
            or payload["validator_capture_cid"] != result.validator_capture.capture_digest
            or payload["runtime_capture_cid"] != result.runtime_capture.capture_digest):
        raise ValueError("PeTTaChainer episode manifest input provenance mismatch")
    values = dict(payload)
    values["budget"] = EpisodeBudget(**payload["budget"])
    return PeTTaChainerEpisodeManifest(**values)


@dataclass(frozen=True)
class ValidatedKernelResult:
    """One structurally validated, provenance-closed patham9 result atom."""

    episode_id: str
    chart_fingerprint: str
    query_term: str
    strength: float
    confidence: float
    stamp_ints: tuple[int, ...]
    evidence_basis_ids: tuple[str, ...]
    result_digest: str

    def __post_init__(self) -> None:
        _nonempty(self.episode_id, "episode_id")
        _sha256_digest(self.chart_fingerprint, "chart_fingerprint")
        canonical_term = _canonical_kernel_term(self.query_term)
        if canonical_term != self.query_term:
            raise ValueError("kernel result query_term is not canonical")
        for field in ("strength", "confidence"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"kernel result {field} must be finite and in [0, 1]")
        if not self.stamp_ints or tuple(sorted(set(self.stamp_ints))) != self.stamp_ints:
            raise ValueError("kernel result stamps must be non-empty, unique, and sorted")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in self.stamp_ints):
            raise ValueError("kernel result stamps must be non-negative integers")
        if len(self.evidence_basis_ids) != len(self.stamp_ints):
            raise ValueError("kernel result evidence bases must close every stamp")
        for basis_id in self.evidence_basis_ids:
            _nonempty(basis_id, "evidence_basis_id")
        expected_digest = _canonical_hash({
            "episode_id": self.episode_id,
            "chart_fingerprint": self.chart_fingerprint,
            "query_term": self.query_term,
            "strength": float(self.strength),
            "confidence": float(self.confidence),
            "stamp_ints": self.stamp_ints,
            "evidence_basis_ids": self.evidence_basis_ids,
        })
        if self.result_digest != expected_digest:
            raise ValueError("kernel result digest does not match typed content")


@dataclass(frozen=True)
class EpisodeBudget:
    """Explicit resource envelope recorded for one isolated kernel episode."""

    max_steps: int
    max_runtime_ms: int
    max_output_chars: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _positive_int(getattr(self, field), field)


@dataclass(frozen=True)
class EpisodeManifest:
    """Immutable audit identity for one completed, non-promoting kernel episode."""

    episode_id: str
    parent_episode_ids: tuple[str, ...]
    chart_id: str
    context_id: str
    evidence_snapshot_id: str
    compiled_program_cid: str
    stamp_map_cid: str
    kernel_name: str
    kernel_version: str
    kernel_capabilities_cid: str
    rule_profile_id: str
    projection_policy_ids: tuple[str, ...]
    controller_envelope_cid: str
    seed: int
    budget: EpisodeBudget
    started_at: str
    finished_at: str
    return_code: int
    stdout_cid: str
    stderr_cid: str
    result_cid: str
    manifest_digest: str

    def __post_init__(self) -> None:
        for field in (
            "episode_id", "chart_id", "context_id", "evidence_snapshot_id", "kernel_name",
            "kernel_version", "rule_profile_id",
        ):
            _nonempty(getattr(self, field), field)
        if tuple(sorted(set(self.parent_episode_ids))) != self.parent_episode_ids:
            raise ValueError("parent_episode_ids must be unique and sorted")
        if self.episode_id in self.parent_episode_ids:
            raise ValueError("an episode cannot be its own parent")
        if not self.projection_policy_ids or tuple(sorted(set(self.projection_policy_ids))) != self.projection_policy_ids:
            raise ValueError("projection_policy_ids must be non-empty, unique, and sorted")
        for value in self.parent_episode_ids + self.projection_policy_ids:
            _nonempty(value, "manifest identity")
        for field in (
            "compiled_program_cid", "stamp_map_cid", "kernel_capabilities_cid",
            "controller_envelope_cid", "stdout_cid", "stderr_cid", "result_cid",
            "manifest_digest",
        ):
            _sha256_digest(getattr(self, field), field)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.return_code, bool) or not isinstance(self.return_code, int):
            raise ValueError("return_code must be an integer")
        if _timestamp(self.finished_at, "finished_at") < _timestamp(self.started_at, "started_at"):
            raise ValueError("finished_at must not precede started_at")
        expected_digest = _canonical_hash(_episode_manifest_payload(self, include_digest=False))
        if self.manifest_digest != expected_digest:
            raise ValueError("manifest digest does not match typed content")


def assemble_legacy_kernel_query_program(
    *,
    compiled: CompiledEpisodeInputs,
    query_term: str,
    max_steps: int,
    task_queue_size: int,
    belief_queue_size: int,
    max_program_chars: int = DEFAULT_MAX_EPISODE_PROGRAM_CHARS,
    parse_check: Callable[[str], None] | None = None,
) -> str:
    """Assemble the only admitted stock patham9 query-program template.

    Compiler-emitted sentences and a declarative query are inserted into fixed
    ``PLN.Init``/``PLN.Query`` control forms.  Callers cannot supply imports,
    rules, or arbitrary executable text through this boundary.  The returned
    program is an inert string; this function neither invokes the kernel nor
    grants promotion authority.  Callers may explicitly supply a local
    ``parse_check`` hook to fail closed on final-program syntax before a
    separately reviewed runner receives it.
    """
    for value, field in (
        (max_steps, "max_steps"),
        (task_queue_size, "task_queue_size"),
        (belief_queue_size, "belief_queue_size"),
        (max_program_chars, "max_program_chars"),
    ):
        _positive_int(value, field)
    if max_steps > DEFAULT_MAX_EPISODE_QUERY_STEPS:
        raise ValueError(
            f"max_steps exceeds bounded limit {DEFAULT_MAX_EPISODE_QUERY_STEPS}"
        )
    if task_queue_size > DEFAULT_MAX_EPISODE_QUEUE_SIZE:
        raise ValueError(
            f"task_queue_size exceeds bounded limit {DEFAULT_MAX_EPISODE_QUEUE_SIZE}"
        )
    if belief_queue_size > DEFAULT_MAX_EPISODE_QUEUE_SIZE:
        raise ValueError(
            f"belief_queue_size exceeds bounded limit {DEFAULT_MAX_EPISODE_QUEUE_SIZE}"
        )
    if not compiled.sentences:
        raise ValueError("compiled episode must contain at least one sentence")

    canonical_query = _canonical_kernel_term(query_term)
    if canonical_query != query_term:
        raise ValueError("query_term must already be canonical")
    beliefs = "\n    ".join(sentence.atom for sentence in compiled.sentences)
    program = "\n".join((
        "!(import! &self PLN)",
        "!(PLN.Init ())",
        "!(PLN.Query (",
        f"    {beliefs}",
        ")",
        f"    {canonical_query}",
        f"    {max_steps} {task_queue_size} {belief_queue_size})",
        "",
    ))
    if len(program) > max_program_chars:
        raise ValueError("assembled kernel program exceeds max_program_chars")
    if parse_check is not None:
        if not callable(parse_check):
            raise ValueError("parse_check must be callable")
        parse_check(program)
    return program


def validate_kernel_result(
    result_atom: str,
    *,
    query_term: str,
    compiled: CompiledEpisodeInputs,
    max_result_chars: int = DEFAULT_MAX_KERNEL_RESULT_CHARS,
) -> ValidatedKernelResult:
    """Validate a patham9 ``((stv S C) (stamps...))`` output fail-closed.

    This boundary validates structure, finite unit-interval truth values, and
    complete stamp-to-evidence-basis closure. It records an ephemeral result;
    it does not infer rule identity or authorize evidence promotion.
    """
    if not isinstance(compiled, CompiledEpisodeInputs):
        raise ValueError("compiled must be immutable compiled episode inputs")
    _positive_int(max_result_chars, "max_result_chars")
    if not isinstance(result_atom, str) or len(result_atom) > max_result_chars:
        raise ValueError("kernel result exceeds max_result_chars")
    canonical_query = _canonical_kernel_term(query_term)
    try:
        parsed = parse_one_list(result_atom)
    except SExpressionSyntaxError as error:
        raise ValueError(f"kernel result must be one valid S-expression list: {error}") from error
    if (len(parsed) != 2 or not isinstance(parsed[0], tuple) or len(parsed[0]) != 3
            or symbol_text(parsed[0][0]) != "stv" or not isinstance(parsed[1], tuple)):
        raise ValueError("kernel result must have shape ((stv strength confidence) (stamps...))")

    numeric_tokens = (symbol_text(parsed[0][1]), symbol_text(parsed[0][2]))
    if None in numeric_tokens:
        raise ValueError("kernel result truth values must be numeric atoms")
    try:
        strength, confidence = (float(value) for value in numeric_tokens)
    except ValueError as error:
        raise ValueError("kernel result truth values must be numeric atoms") from error
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (strength, confidence)):
        raise ValueError("kernel result truth values must be finite and in [0, 1]")

    stamps: list[int] = []
    for expression in parsed[1]:
        token = symbol_text(expression)
        try:
            stamp = int(token) if token is not None else -1
        except ValueError as error:
            raise ValueError("kernel result stamps must be canonical non-negative integers") from error
        if token != str(stamp) or stamp < 0:
            raise ValueError("kernel result stamps must be canonical non-negative integers")
        stamps.append(stamp)
    stamp_ints = tuple(stamps)
    if not stamp_ints or tuple(sorted(set(stamp_ints))) != stamp_ints:
        raise ValueError("kernel result stamps must be non-empty, unique, and sorted")

    basis_by_stamp = {entry.stamp_int: entry.basis_id for entry in compiled.stamp_map}
    unknown = tuple(stamp for stamp in stamp_ints if stamp not in basis_by_stamp)
    if unknown:
        raise ValueError(f"kernel result contains unknown episode stamps: {unknown}")
    evidence_basis_ids = tuple(basis_by_stamp[stamp] for stamp in stamp_ints)
    digest_payload = {
        "episode_id": compiled.episode_id,
        "chart_fingerprint": compiled.chart_fingerprint,
        "query_term": canonical_query,
        "strength": strength,
        "confidence": confidence,
        "stamp_ints": stamp_ints,
        "evidence_basis_ids": evidence_basis_ids,
    }
    return ValidatedKernelResult(
        episode_id=compiled.episode_id,
        chart_fingerprint=compiled.chart_fingerprint,
        query_term=canonical_query,
        strength=strength,
        confidence=confidence,
        stamp_ints=stamp_ints,
        evidence_basis_ids=evidence_basis_ids,
        result_digest=_canonical_hash(digest_payload),
    )


def validate_exact_kernel_replay(
    result_atom: str,
    *,
    expected: ValidatedKernelResult,
    compiled: CompiledEpisodeInputs,
    max_result_chars: int = DEFAULT_MAX_KERNEL_RESULT_CHARS,
) -> ValidatedKernelResult:
    """Validate one replay output and require exact semantic equivalence.

    The expected result must itself close against the supplied immutable
    compiler inputs.  This function compares typed semantic content rather
    than raw output formatting, but it does not execute a kernel or establish
    rule/trace identity.
    """
    if not isinstance(expected, ValidatedKernelResult):
        raise ValueError("expected must be a validated kernel result")
    if not isinstance(compiled, CompiledEpisodeInputs):
        raise ValueError("compiled must be immutable compiled episode inputs")
    if (expected.episode_id != compiled.episode_id
            or expected.chart_fingerprint != compiled.chart_fingerprint):
        raise ValueError("expected kernel result does not match compiled episode")
    basis_by_stamp = {entry.stamp_int: entry.basis_id for entry in compiled.stamp_map}
    unknown = tuple(stamp for stamp in expected.stamp_ints if stamp not in basis_by_stamp)
    if unknown:
        raise ValueError(f"expected kernel result contains unknown episode stamps: {unknown}")
    expected_bases = tuple(basis_by_stamp[stamp] for stamp in expected.stamp_ints)
    if expected.evidence_basis_ids != expected_bases:
        raise ValueError("expected kernel result evidence bases do not match compiled episode stamps")

    replayed = validate_kernel_result(
        result_atom,
        query_term=expected.query_term,
        compiled=compiled,
        max_result_chars=max_result_chars,
    )
    if replayed.result_digest != expected.result_digest:
        raise ValueError("kernel replay result does not exactly match expected semantic result")
    return replayed


@dataclass(frozen=True)
class KernelProcessCapture:
    """Bounded raw process result; it is not a validated PLN result."""

    argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    program_cid: str | None = None
    executable_sha256: str | None = None
    program_sha256: str | None = None
    cwd: str | None = None
    env: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self) -> None:
        def require_utf8(value: str, field: str) -> None:
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(f"{field} must be valid UTF-8 text") from error

        if (not isinstance(self.argv, tuple) or not self.argv
                or any(
                    not isinstance(arg, str) or not arg or "\0" in arg
                    for arg in self.argv
                )):
            raise ValueError(
                "argv must be a non-empty tuple of non-empty strings without NUL bytes"
            )
        for arg in self.argv:
            require_utf8(arg, "argv entries")
        if isinstance(self.return_code, bool) or not isinstance(self.return_code, int):
            raise ValueError("return_code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("stdout and stderr must be strings")
        require_utf8(self.stdout, "stdout")
        require_utf8(self.stderr, "stderr")
        if self.program_cid is not None:
            _sha256_digest(self.program_cid, "program_cid")
        if self.executable_sha256 is not None:
            _sha256_digest(self.executable_sha256, "executable_sha256")
        if self.program_sha256 is not None:
            _sha256_digest(self.program_sha256, "program_sha256")
        if self.cwd is not None and (
            not isinstance(self.cwd, str) or not self.cwd or "\0" in self.cwd
        ):
            raise ValueError("cwd must be a non-empty string without NUL bytes")
        if self.cwd is not None:
            require_utf8(self.cwd, "cwd")
            cwd_path = Path(self.cwd)
            if not cwd_path.is_absolute() or cwd_path != cwd_path.resolve(strict=False):
                raise ValueError("cwd must be an absolute normalized path")
        if self.env is not None and (
            not isinstance(self.env, tuple)
            or any(
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0]
                or not isinstance(entry[1], str)
                or "\0" in entry[0]
                or "\0" in entry[1]
                or "=" in entry[0]
                for entry in self.env
            )
            or self.env != tuple(sorted(self.env))
            or len({key for key, _ in self.env}) != len(self.env)
        ):
            raise ValueError(
                "env must be canonical sorted unique process environment entries"
            )
        if self.env is not None:
            for key, value in self.env:
                require_utf8(key, "env keys")
                require_utf8(value, "env values")


def kernel_process_capture_document(
    capture: KernelProcessCapture,
) -> dict[str, object]:
    """Return a checksummed document for one bounded raw process capture."""
    if not isinstance(capture, KernelProcessCapture):
        raise ValueError("capture must be a bounded kernel process capture")
    payload = {
        "argv": list(capture.argv),
        "return_code": capture.return_code,
        "stdout": capture.stdout,
        "stderr": capture.stderr,
        "program_cid": capture.program_cid,
        "executable_sha256": capture.executable_sha256,
        "program_sha256": capture.program_sha256,
        "cwd": capture.cwd,
        "env": None if capture.env is None else [list(entry) for entry in capture.env],
    }
    return {
        "schema": "petta-memory-kernel-process-capture-v1",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_kernel_process_capture(
    path: str | Path, capture: KernelProcessCapture,
) -> None:
    """Create one immutable bounded raw-process capture artifact."""
    data = json.dumps(
        kernel_process_capture_document(capture), sort_keys=True, indent=2,
    ) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_create_once_durable(destination, data)


def read_kernel_process_capture(path: str | Path) -> KernelProcessCapture:
    """Reload a checksummed raw capture through its complete typed boundary."""
    document = _load_unambiguous_json(path)
    if (not isinstance(document, dict)
            or set(document) != {"schema", "payload", "document_digest"}
            or document.get("schema") != "petta-memory-kernel-process-capture-v1"):
        raise ValueError("invalid kernel process capture document schema")
    payload = document.get("payload")
    fields = set(KernelProcessCapture.__dataclass_fields__)
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("invalid kernel process capture payload")
    if document.get("document_digest") != _canonical_hash(payload):
        raise ValueError("kernel process capture document checksum mismatch")
    argv = payload.get("argv")
    env = payload.get("env")
    if not isinstance(argv, list):
        raise ValueError("invalid kernel process capture argv")
    if env is not None and (
        not isinstance(env, list)
        or any(not isinstance(entry, list) or len(entry) != 2 for entry in env)
    ):
        raise ValueError("invalid kernel process capture environment")
    values = dict(payload)
    values["argv"] = tuple(argv)
    values["env"] = None if env is None else tuple(tuple(entry) for entry in env)
    return KernelProcessCapture(**values)


def validate_kernel_capture_result(
    capture: KernelProcessCapture,
    *,
    result_atom: str,
    query_term: str,
    compiled: CompiledEpisodeInputs,
    max_result_chars: int = DEFAULT_MAX_KERNEL_RESULT_CHARS,
) -> ValidatedKernelResult:
    """Admit one result atom as originating in a successful raw capture.

    The result atom must occur exactly once as a complete LF-delimited output
    record in bounded stdout, after which the normal typed result validator
    closes its stamps against the immutable episode inputs.  This is a
    non-promoting process/result boundary; callers must still construct and
    persist an ``EpisodeManifest`` separately.
    """
    if not isinstance(capture, KernelProcessCapture):
        raise ValueError("capture must be a bounded kernel process capture")
    if capture.return_code != 0:
        raise ValueError("kernel process did not exit successfully")
    if capture.stderr:
        raise ValueError("kernel process emitted unexpected stderr")
    if not isinstance(result_atom, str) or not result_atom:
        raise ValueError("result_atom must be a non-empty string")
    # Kernel captures commit the exact stdout text.  Split only on the actual
    # LF framing byte instead of treating other Unicode control characters as
    # record boundaries through str.splitlines().
    result_lines = capture.stdout.split("\n")
    if result_lines.count(result_atom) != 1:
        raise ValueError(
            "kernel result atom must occur exactly once as a complete captured stdout line"
        )
    return validate_kernel_result(
        result_atom,
        query_term=query_term,
        compiled=compiled,
        max_result_chars=max_result_chars,
    )


def validate_exact_kernel_capture_replay(
    capture: KernelProcessCapture,
    *,
    result_atom: str,
    expected: ValidatedKernelResult,
    compiled: CompiledEpisodeInputs,
    max_result_chars: int = DEFAULT_MAX_KERNEL_RESULT_CHARS,
) -> ValidatedKernelResult:
    """Require an exact semantic replay from one bounded process capture.

    Unlike :func:`validate_exact_kernel_replay`, this boundary also proves that
    the candidate result is a unique complete stdout record from a successful
    process with no stderr.  It remains non-promoting and does not claim
    rule/trace identity.
    """
    if not isinstance(expected, ValidatedKernelResult):
        raise ValueError("expected must be a validated kernel result")
    if not isinstance(compiled, CompiledEpisodeInputs):
        raise ValueError("compiled must be immutable compiled episode inputs")
    captured = validate_kernel_capture_result(
        capture,
        result_atom=result_atom,
        query_term=expected.query_term,
        compiled=compiled,
        max_result_chars=max_result_chars,
    )
    replayed = validate_exact_kernel_replay(
        result_atom,
        expected=expected,
        compiled=compiled,
        max_result_chars=max_result_chars,
    )
    if captured != replayed:
        raise ValueError("captured kernel replay does not match validated replay")
    return replayed


def build_captured_episode_manifest(
    *, capture: KernelProcessCapture, result_atom: str, query_term: str,
    compiled: CompiledEpisodeInputs, chart: "PiChart",
    evidence_snapshot: "EvidenceSnapshot", complete_program: str,
    kernel_name: str, kernel_capabilities_cid: str,
    controller_envelope_cid: str, seed: int, budget: EpisodeBudget,
    started_at: str, finished_at: str,
    parent_episode_ids: Iterable[str] = (),
    max_result_chars: int = DEFAULT_MAX_KERNEL_RESULT_CHARS,
    max_program_chars: int = DEFAULT_MAX_EPISODE_PROGRAM_CHARS,
) -> EpisodeManifest:
    """Close one successful process capture directly into an episode manifest.

    Result admission and manifest construction consume the same immutable
    capture, and the capture must commit to the exact complete program supplied
    to the manifest.  This prevents callers from validating one process result
    while accidentally recording another process's inputs or outputs.  This is
    still a non-promoting audit boundary and grants no memory-write authority.
    """
    result = validate_kernel_capture_result(
        capture,
        result_atom=result_atom,
        query_term=query_term,
        compiled=compiled,
        max_result_chars=max_result_chars,
    )
    expected_program_cid = _canonical_hash({"complete_program": complete_program})
    if capture.program_cid != expected_program_cid:
        raise ValueError("kernel capture program does not match complete_program")
    return build_episode_manifest(
        compiled=compiled,
        chart=chart,
        evidence_snapshot=evidence_snapshot,
        result=result,
        complete_program=complete_program,
        kernel_name=kernel_name,
        kernel_capabilities_cid=kernel_capabilities_cid,
        controller_envelope_cid=controller_envelope_cid,
        seed=seed,
        budget=budget,
        started_at=started_at,
        finished_at=finished_at,
        return_code=capture.return_code,
        stdout=capture.stdout,
        stderr=capture.stderr,
        parent_episode_ids=parent_episode_ids,
        max_program_chars=max_program_chars,
    )


def validate_phase0_reference_replay(
    reference: Phase0ReferenceArtifact,
    capture: KernelProcessCapture,
) -> KernelProcessCapture:
    """Close one fresh bounded capture against an admitted Phase-0 anchor.

    This gate requires successful execution and byte-exact stdout replay.  It
    does not construct a Phase-2 episode, infer rule identity, or authorize a
    belief promotion or memory write.
    """
    if not isinstance(reference, Phase0ReferenceArtifact):
        raise ValueError("reference must be an admitted Phase-0 artifact")
    if not isinstance(capture, KernelProcessCapture):
        raise ValueError("capture must be a bounded kernel process capture")
    if capture.return_code != 0:
        raise ValueError("Phase-0 reference replay process did not exit successfully")
    if capture.stderr:
        raise ValueError("Phase-0 reference replay emitted unexpected stderr")
    executable = Path(capture.argv[0])
    if not executable.is_absolute() or executable != executable.resolve(strict=False):
        raise ValueError(
            "Phase-0 reference replay executable path must be absolute and normalized"
        )
    if len(capture.argv) != 1:
        raise ValueError(
            "Phase-0 reference replay must use the frozen stdin-only launch shape"
        )
    if capture.cwd is not None:
        raise ValueError(
            "Phase-0 reference replay must use the frozen inherited working directory"
        )
    if capture.env is not None:
        raise ValueError(
            "Phase-0 reference replay must use the frozen inherited process environment"
        )
    if capture.executable_sha256 != reference.runtime_executable_sha256:
        raise ValueError("Phase-0 reference replay executable checksum mismatch")
    if capture.program_sha256 != reference.source_sha256:
        raise ValueError("Phase-0 reference replay program checksum mismatch")
    if capture.program_cid != reference.source_cid:
        raise ValueError("Phase-0 reference replay program CID mismatch")
    stdout = capture.stdout.encode("utf-8")
    if len(stdout) != reference.output_bytes:
        raise ValueError("Phase-0 reference replay output byte count mismatch")
    if sha256(stdout).hexdigest() != reference.output_sha256:
        raise ValueError("Phase-0 reference replay output checksum mismatch")
    if reference.semantic_result not in capture.stdout or "(Passed: #t)" not in capture.stdout:
        raise ValueError("Phase-0 reference replay semantic markers are missing")
    return capture


def run_kernel_subprocess(
    program: str,
    *,
    argv: Iterable[str],
    timeout_ms: int,
    max_program_bytes: int = DEFAULT_MAX_EPISODE_PROGRAM_CHARS,
    max_argv_bytes: int = 16_384,
    max_cwd_bytes: int = 4_096,
    max_env_bytes: int = 65_536,
    max_capture_bytes: int = DEFAULT_MAX_KERNEL_CAPTURE_BYTES,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    expected_executable_sha256: str | None = None,
) -> KernelProcessCapture:
    """Run one already-assembled program through a bounded shell-free process.

    The program is supplied on stdin. Timeout, non-UTF-8 output, and output over
    either per-stream byte ceiling fail closed. A successful capture still needs
    result parsing, provenance closure, and manifest construction before use.
    """
    if not isinstance(program, str) or not program:
        raise ValueError("program must be a non-empty string")
    if "\0" in program:
        raise ValueError("program must not contain NUL bytes")
    _positive_int(max_program_bytes, "max_program_bytes")
    try:
        encoded_program = program.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("program must be valid UTF-8 text") from error
    if len(encoded_program) > max_program_bytes:
        raise ValueError("program exceeds max_program_bytes")
    if isinstance(argv, (str, bytes)):
        raise ValueError("argv must be an iterable of argument strings, not text or bytes")
    _positive_int(max_argv_bytes, "max_argv_bytes")
    try:
        iterator = iter(argv)
    except TypeError as error:
        raise ValueError("argv must be an iterable of argument strings") from error
    command_items: list[str] = []
    argv_bytes = 0
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            raise ValueError("argv iteration failed") from error
        if not isinstance(item, str) or not item:
            raise ValueError("argv must contain non-empty strings")
        if "\0" in item:
            raise ValueError("argv must not contain NUL bytes")
        try:
            argv_bytes += len(item.encode("utf-8")) + 1
        except UnicodeEncodeError as error:
            raise ValueError("argv entries must be valid UTF-8 text") from error
        if argv_bytes > max_argv_bytes:
            raise ValueError("argv exceeds max_argv_bytes")
        command_items.append(item)
    command = tuple(command_items)
    if not command:
        raise ValueError("argv must contain non-empty strings")

    # Account for the terminating NUL carried by each OS argv entry as well as
    # the caller-visible UTF-8 payload.
    def argv_size_bytes(items: tuple[str, ...]) -> int:
        return sum(len(item.encode("utf-8")) + 1 for item in items)
    executable_sha256: str | None = None
    if expected_executable_sha256 is not None:
        if (not isinstance(expected_executable_sha256, str)
                or len(expected_executable_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_executable_sha256)):
            raise ValueError("expected_executable_sha256 must be a lowercase SHA-256 digest")
        executable = Path(command[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("pinned executable must name an absolute regular file")
        try:
            executable = executable.resolve(strict=True)
        except OSError as error:
            raise ValueError("pinned executable could not be resolved") from error
        command = (str(executable), *command[1:])
        if argv_size_bytes(command) > max_argv_bytes:
            raise ValueError("resolved argv exceeds max_argv_bytes")
        digest = sha256()
        try:
            with executable.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise ValueError("pinned executable could not be read") from error
        if digest.hexdigest() != expected_executable_sha256:
            raise ValueError("kernel executable SHA-256 does not match expected_executable_sha256")
        executable_sha256 = expected_executable_sha256
    _positive_int(max_cwd_bytes, "max_cwd_bytes")
    normalized_cwd: str | None = None
    if cwd is not None:
        try:
            normalized_cwd = os.fspath(cwd)
        except TypeError as error:
            raise ValueError("cwd must be a string or path-like value") from error
        if not isinstance(normalized_cwd, str) or not normalized_cwd:
            raise ValueError("cwd must be a non-empty string or path-like value")
        if "\0" in normalized_cwd:
            raise ValueError("cwd must not contain NUL bytes")
        try:
            cwd_bytes = len(normalized_cwd.encode("utf-8")) + 1
        except UnicodeEncodeError as error:
            raise ValueError("cwd must be valid UTF-8 text") from error
        if cwd_bytes > max_cwd_bytes:
            raise ValueError("cwd exceeds max_cwd_bytes")
        try:
            cwd_path = Path(normalized_cwd).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("cwd could not be resolved") from error
        if not cwd_path.is_dir():
            raise ValueError("cwd must resolve to a directory")
        normalized_cwd = str(cwd_path)
        # Resolution can expand a relative path or symlink, so bind the limit
        # to the exact path retained in the capture and delivered to Popen.
        if len(normalized_cwd.encode("utf-8")) + 1 > max_cwd_bytes:
            raise ValueError("resolved cwd exceeds max_cwd_bytes")
    _positive_int(max_env_bytes, "max_env_bytes")
    normalized_env: dict[str, str] | None = None
    if env is not None:
        if not isinstance(env, Mapping):
            raise ValueError("env must be a mapping of strings")
        normalized_env = {}
        env_bytes = 0
        try:
            env_iterator = iter(env.items())
        except Exception as error:
            raise ValueError("env iteration failed") from error
        while True:
            try:
                item = next(env_iterator)
            except StopIteration:
                break
            except Exception as error:
                raise ValueError("env iteration failed") from error
            if not isinstance(item, tuple) or len(item) != 2:
                error = ValueError("environment item is not a two-element tuple")
                raise ValueError("env items must be key-value pairs") from error
            key, value = item
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise ValueError("env must map non-empty string keys to string values")
            if "\0" in key or "\0" in value or "=" in key:
                raise ValueError("env keys and values must be valid process environment strings")
            if key in normalized_env:
                raise ValueError("env must not contain duplicate keys")
            # The process environment serializes each entry as KEY=VALUE\0.
            try:
                env_bytes += len(key.encode("utf-8")) + len(value.encode("utf-8")) + 2
            except UnicodeEncodeError as error:
                raise ValueError("env keys and values must be valid UTF-8 text") from error
            if env_bytes > max_env_bytes:
                raise ValueError("env exceeds max_env_bytes")
            normalized_env[key] = value
    _positive_int(timeout_ms, "timeout_ms")
    _positive_int(max_capture_bytes, "max_capture_bytes")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=normalized_cwd,
            env=normalized_env,
            shell=False,
            start_new_session=True,
        )
    except Exception as error:
        raise ValueError("kernel subprocess could not be launched") from error
    if process.stdin is None or process.stdout is None or process.stderr is None:
        cleanup_errors: list[BaseException] = []
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as error:
            cleanup_errors.append(error)
        try:
            process.wait()
        except Exception as error:
            cleanup_errors.append(error)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception as error:
                    cleanup_errors.append(error)
        if cleanup_errors:
            raise ValueError("kernel subprocess pipe validation cleanup failed") from cleanup_errors[0]
        raise ValueError("kernel subprocess did not provide requested pipes")
    captures: dict[str, bytes] = {}
    overflow: list[str] = []
    capture_errors: list[BaseException] = []
    stdin_errors: list[BaseException] = []
    process_cleanup_errors: list[BaseException] = []
    stream_close_errors: list[BaseException] = []
    thread_join_errors: list[BaseException] = []
    process_wait_errors: list[BaseException] = []
    timeout_errors: list[BaseException] = []
    timeout_cleanup_errors: list[BaseException] = []
    thread_start_errors: list[BaseException] = []
    thread_start_cleanup_errors: list[BaseException] = []

    def kill_process_tree() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as error:
            process_cleanup_errors.append(error)

    def bounded_read(name: str, stream: object) -> None:
        chunks: list[bytes] = []
        size = 0
        try:
            while True:
                chunk = stream.read(min(65_536, max_capture_bytes + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > max_capture_bytes:
                    overflow.append(name)
                    kill_process_tree()
                    break
            captures[name] = b"".join(chunks)
        except Exception as error:
            capture_errors.append(error)
            kill_process_tree()

    def write_program() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(encoded_program)
            process.stdin.flush()
        except Exception as error:
            stdin_errors.append(error)
        finally:
            try:
                process.stdin.close()
            except Exception as error:
                stdin_errors.append(error)

    try:
        readers = [
            threading.Thread(target=bounded_read, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=bounded_read, args=("stderr", process.stderr), daemon=True),
        ]
        writer = threading.Thread(target=write_program, daemon=True)
    except Exception as error:
        kill_process_tree()
        construction_cleanup_errors: list[BaseException] = list(process_cleanup_errors)
        try:
            process.wait()
        except Exception as cleanup_error:
            construction_cleanup_errors.append(cleanup_error)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception as cleanup_error:
                    construction_cleanup_errors.append(cleanup_error)
        if construction_cleanup_errors:
            raise ValueError("kernel subprocess worker construction cleanup failed") from construction_cleanup_errors[0]
        raise ValueError("kernel subprocess worker construction failed") from error
    started_workers: list[threading.Thread] = []
    try:
        try:
            for worker in (*readers, writer):
                worker.start()
                started_workers.append(worker)
        except Exception as error:
            thread_start_errors.append(error)
            kill_process_tree()
            try:
                process.wait()
            except Exception as cleanup_error:
                thread_start_cleanup_errors.append(cleanup_error)
        if not thread_start_errors:
            process.wait(timeout=timeout_ms / 1000)
    except subprocess.TimeoutExpired as error:
        kill_process_tree()
        try:
            process.wait()
        except Exception as cleanup_error:
            timeout_cleanup_errors.append(cleanup_error)
        timeout_errors.append(error)
    except Exception as error:
        kill_process_tree()
        process_wait_errors.append(error)
    finally:
        # A kernel must not extend the capture lifetime by leaving descendants
        # holding inherited stdout/stderr pipes after its direct process exits.
        kill_process_tree()
        for worker in started_workers:
            try:
                worker.join()
            except Exception as error:
                thread_join_errors.append(error)
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except Exception as error:
                stream_close_errors.append(error)
    if stream_close_errors:
        raise ValueError("kernel subprocess stream cleanup failed") from stream_close_errors[0]
    if thread_join_errors:
        raise ValueError("kernel subprocess worker cleanup failed") from thread_join_errors[0]
    if process_cleanup_errors:
        raise ValueError("kernel subprocess process-group cleanup failed") from process_cleanup_errors[0]
    if timeout_cleanup_errors:
        raise ValueError("kernel subprocess timeout cleanup failed") from timeout_cleanup_errors[0]
    if timeout_errors:
        raise ValueError("kernel subprocess exceeded timeout_ms") from timeout_errors[0]
    if process_wait_errors:
        raise ValueError("kernel subprocess wait failed") from process_wait_errors[0]
    if thread_start_cleanup_errors:
        raise ValueError("kernel subprocess worker startup cleanup failed") from thread_start_cleanup_errors[0]
    if thread_start_errors:
        raise ValueError("kernel subprocess worker startup failed") from thread_start_errors[0]
    if overflow:
        raise ValueError(f"kernel subprocess {overflow[0]} exceeded max_capture_bytes")
    if capture_errors:
        raise ValueError("kernel subprocess output capture failed") from capture_errors[0]
    if stdin_errors:
        raise ValueError("kernel subprocess closed stdin before complete program delivery") from stdin_errors[0]
    stdout_bytes = captures["stdout"]
    stderr_bytes = captures["stderr"]
    try:
        stdout = stdout_bytes.decode("utf-8", errors="strict")
        stderr = stderr_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("kernel subprocess output must be valid UTF-8") from error
    return KernelProcessCapture(
        command,
        process.returncode,
        stdout,
        stderr,
        _canonical_hash({"complete_program": program}),
        executable_sha256,
        sha256(program.encode("utf-8")).hexdigest(),
        normalized_cwd,
        None if normalized_env is None else tuple(sorted(normalized_env.items())),
    )


def _episode_manifest_payload(
    manifest: EpisodeManifest, *, include_digest: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "episode_id": manifest.episode_id,
        "parent_episode_ids": list(manifest.parent_episode_ids),
        "chart_id": manifest.chart_id,
        "context_id": manifest.context_id,
        "evidence_snapshot_id": manifest.evidence_snapshot_id,
        "compiled_program_cid": manifest.compiled_program_cid,
        "stamp_map_cid": manifest.stamp_map_cid,
        "kernel_name": manifest.kernel_name,
        "kernel_version": manifest.kernel_version,
        "kernel_capabilities_cid": manifest.kernel_capabilities_cid,
        "rule_profile_id": manifest.rule_profile_id,
        "projection_policy_ids": list(manifest.projection_policy_ids),
        "controller_envelope_cid": manifest.controller_envelope_cid,
        "seed": manifest.seed,
        "budget": {
            "max_steps": manifest.budget.max_steps,
            "max_runtime_ms": manifest.budget.max_runtime_ms,
            "max_output_chars": manifest.budget.max_output_chars,
        },
        "started_at": manifest.started_at,
        "finished_at": manifest.finished_at,
        "return_code": manifest.return_code,
        "stdout_cid": manifest.stdout_cid,
        "stderr_cid": manifest.stderr_cid,
        "result_cid": manifest.result_cid,
    }
    if include_digest:
        payload["manifest_digest"] = manifest.manifest_digest
    return payload


def build_episode_manifest(
    *, compiled: CompiledEpisodeInputs, chart: "PiChart", evidence_snapshot: "EvidenceSnapshot",
    result: ValidatedKernelResult, complete_program: str, kernel_name: str,
    kernel_capabilities_cid: str, controller_envelope_cid: str, seed: int,
    budget: EpisodeBudget, started_at: str, finished_at: str, return_code: int,
    stdout: str, stderr: str, parent_episode_ids: Iterable[str] = (),
    max_program_chars: int = DEFAULT_MAX_EPISODE_PROGRAM_CHARS,
) -> EpisodeManifest:
    """Close a completed episode manifest over immutable inputs and captured output.

    The supplied program is content-addressed and must contain every compiler-emitted
    sentence exactly once. Runtime invocation and artifact storage remain caller
    responsibilities; constructing this record grants no promotion authority.
    """
    if not isinstance(compiled, CompiledEpisodeInputs):
        raise ValueError("compiled must be immutable compiled episode inputs")
    if not isinstance(result, ValidatedKernelResult):
        raise ValueError("result must be a validated kernel result")
    if not isinstance(chart, PiChart):
        raise ValueError("chart must be an immutable pi chart")
    if not isinstance(evidence_snapshot, EvidenceSnapshot):
        raise ValueError("evidence_snapshot must be an immutable evidence snapshot")
    if not isinstance(budget, EpisodeBudget):
        raise ValueError("budget must be an immutable episode budget")
    _positive_int(max_program_chars, "max_program_chars")
    if not isinstance(complete_program, str) or not complete_program or len(complete_program) > max_program_chars:
        raise ValueError("complete_program must be non-empty and within max_program_chars")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise ValueError("stdout and stderr must be strings")
    if compiled.episode_id != result.episode_id or compiled.chart_fingerprint != result.chart_fingerprint:
        raise ValueError("kernel result does not match compiled episode")
    basis_by_stamp = {entry.stamp_int: entry.basis_id for entry in compiled.stamp_map}
    if (any(stamp not in basis_by_stamp for stamp in result.stamp_ints)
            or result.evidence_basis_ids != tuple(basis_by_stamp[stamp] for stamp in result.stamp_ints)):
        raise ValueError("kernel result stamps do not match compiled episode evidence bases")
    if compiled.chart_fingerprint != chart.chart_fingerprint:
        raise ValueError("chart does not match compiled episode")
    if (compiled.evidence_snapshot_fingerprint != evidence_snapshot.snapshot_fingerprint
            or chart.evidence_snapshot_id != evidence_snapshot.id
            or chart.evidence_snapshot_fingerprint != evidence_snapshot.snapshot_fingerprint):
        raise ValueError("evidence snapshot does not match compiled episode chart")
    if any(sentence.meta.chart_id != chart.id or sentence.meta.context_id != chart.context_id
           for sentence in compiled.sentences):
        raise ValueError("compiled sentence chart/context identity mismatch")
    missing_or_duplicate = tuple(
        sentence.meta.sentence_digest for sentence in compiled.sentences
        if complete_program.count(sentence.atom) != 1
    )
    if missing_or_duplicate:
        raise ValueError("complete_program must contain every compiled sentence exactly once")
    if result.query_term not in complete_program:
        raise ValueError("complete_program must contain the validated result query term")

    parents = tuple(sorted(tuple(parent_episode_ids)))
    projection_ids = tuple(sorted({
        chart.policy.projection_policy_id,
        chart.policy.kernel_projection_policy_id,
    }))
    stamp_payload = [
        {
            "episode_id": entry.episode_id,
            "stamp_int": entry.stamp_int,
            "basis_id": entry.basis_id,
            "member_token_digest": entry.member_token_digest,
        }
        for entry in compiled.stamp_map
    ]
    values = dict(
        episode_id=compiled.episode_id, parent_episode_ids=parents, chart_id=chart.id,
        context_id=chart.context_id, evidence_snapshot_id=evidence_snapshot.id,
        compiled_program_cid=_canonical_hash({"complete_program": complete_program}),
        stamp_map_cid=_canonical_hash(stamp_payload), kernel_name=kernel_name,
        kernel_version=chart.policy.kernel_version,
        kernel_capabilities_cid=kernel_capabilities_cid,
        rule_profile_id=chart.policy.rule_profile_id,
        projection_policy_ids=projection_ids,
        controller_envelope_cid=controller_envelope_cid, seed=seed, budget=budget,
        started_at=started_at, finished_at=finished_at, return_code=return_code,
        stdout_cid=_canonical_hash({"stdout": stdout}),
        stderr_cid=_canonical_hash({"stderr": stderr}), result_cid=result.result_digest,
    )
    digest_payload = dict(values)
    digest_payload["parent_episode_ids"] = list(parents)
    digest_payload["projection_policy_ids"] = list(projection_ids)
    digest_payload["budget"] = {
        "max_steps": budget.max_steps,
        "max_runtime_ms": budget.max_runtime_ms,
        "max_output_chars": budget.max_output_chars,
    }
    return EpisodeManifest(**values, manifest_digest=_canonical_hash(digest_payload))


def episode_manifest_document(manifest: EpisodeManifest) -> dict[str, object]:
    if not isinstance(manifest, EpisodeManifest):
        raise ValueError("manifest must be a typed episode manifest")
    payload = _episode_manifest_payload(manifest)
    return {
        "schema": "petta-memory-pipln-episode-manifest-v1",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_episode_manifest(path: str | Path, manifest: EpisodeManifest) -> None:
    """Create one immutable episode-manifest artifact; never replace a path."""
    data = json.dumps(episode_manifest_document(manifest), sort_keys=True, indent=2) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_create_once_durable(destination, data)


def read_episode_manifest(
    path: str | Path, *, compiled: CompiledEpisodeInputs | None = None,
    result: ValidatedKernelResult | None = None,
    complete_program: str | None = None,
    capture: KernelProcessCapture | None = None,
) -> EpisodeManifest:
    """Load a checksummed manifest and optionally close its replay provenance."""
    if compiled is not None and not isinstance(compiled, CompiledEpisodeInputs):
        raise ValueError("compiled must be immutable compiled episode inputs")
    if result is not None and not isinstance(result, ValidatedKernelResult):
        raise ValueError("result must be a validated kernel result")
    if capture is not None and not isinstance(capture, KernelProcessCapture):
        raise ValueError("episode manifest capture must be a KernelProcessCapture")
    if complete_program is not None and (
        not isinstance(complete_program, str) or not complete_program
    ):
        raise ValueError("complete_program must be a non-empty string")
    document = _load_unambiguous_json(path)
    if (not isinstance(document, dict)
            or set(document) != {"schema", "payload", "document_digest"}
            or document.get("schema") != "petta-memory-pipln-episode-manifest-v1"):
        raise ValueError("invalid episode manifest document schema")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("document_digest") != _canonical_hash(payload):
        raise ValueError("episode manifest document checksum mismatch")
    expected_fields = set(EpisodeManifest.__dataclass_fields__) - {"budget"}
    expected_fields.add("budget")
    if (set(payload) != expected_fields
            or not isinstance(payload["parent_episode_ids"], list)
            or not isinstance(payload["projection_policy_ids"], list)
            or not isinstance(payload["budget"], dict)
            or set(payload["budget"]) != set(EpisodeBudget.__dataclass_fields__)):
        raise ValueError("invalid episode manifest payload")
    values = dict(payload)
    values["parent_episode_ids"] = tuple(payload["parent_episode_ids"])
    values["projection_policy_ids"] = tuple(payload["projection_policy_ids"])
    values["budget"] = EpisodeBudget(**payload["budget"])
    manifest = EpisodeManifest(**values)
    if complete_program is not None:
        if (manifest.compiled_program_cid
                != _canonical_hash({"complete_program": complete_program})):
            raise ValueError("episode manifest does not match complete program")
    if capture is not None:
        if (manifest.return_code != capture.return_code
                or manifest.stdout_cid != _canonical_hash({"stdout": capture.stdout})
                or manifest.stderr_cid != _canonical_hash({"stderr": capture.stderr})):
            raise ValueError("episode manifest does not match kernel process capture")
        if (capture.program_cid is None
                or manifest.compiled_program_cid != capture.program_cid):
            raise ValueError("episode manifest does not match captured program")
    if compiled is not None:
        compiled_chart_ids = {sentence.meta.chart_id for sentence in compiled.sentences}
        compiled_context_ids = {sentence.meta.context_id for sentence in compiled.sentences}
        stamp_payload = [
            {
                "episode_id": entry.episode_id,
                "stamp_int": entry.stamp_int,
                "basis_id": entry.basis_id,
                "member_token_digest": entry.member_token_digest,
            }
            for entry in compiled.stamp_map
        ]
        if (manifest.episode_id != compiled.episode_id
                or compiled_chart_ids != {manifest.chart_id}
                or compiled_context_ids != {manifest.context_id}
                or manifest.stamp_map_cid != _canonical_hash(stamp_payload)):
            raise ValueError("episode manifest does not match compiled episode")
        if (complete_program is not None
                and any(complete_program.count(sentence.atom) != 1
                        for sentence in compiled.sentences)):
            raise ValueError("complete program does not match compiled episode")
    if result is not None:
        if (manifest.episode_id != result.episode_id
                or manifest.result_cid != result.result_digest):
            raise ValueError("episode manifest does not match validated result")
        if complete_program is not None and result.query_term not in complete_program:
            raise ValueError("complete program does not match validated result")
        if compiled is not None and (
                result.episode_id != compiled.episode_id
                or result.chart_fingerprint != compiled.chart_fingerprint):
            raise ValueError("validated result does not match compiled episode")
        if compiled is not None:
            basis_by_stamp = {
                entry.stamp_int: entry.basis_id for entry in compiled.stamp_map
            }
            if (any(stamp not in basis_by_stamp for stamp in result.stamp_ints)
                    or result.evidence_basis_ids
                    != tuple(basis_by_stamp[stamp] for stamp in result.stamp_ints)):
                raise ValueError(
                    "validated result stamps do not match compiled episode evidence bases"
                )
    return manifest


def validated_kernel_result_document(result: ValidatedKernelResult) -> dict[str, object]:
    """Return a checksummed artifact for one provenance-closed kernel result."""
    if not isinstance(result, ValidatedKernelResult):
        raise ValueError("result must be a typed validated kernel result")
    payload = {
        "episode_id": result.episode_id,
        "chart_fingerprint": result.chart_fingerprint,
        "query_term": result.query_term,
        "strength": result.strength,
        "confidence": result.confidence,
        "stamp_ints": list(result.stamp_ints),
        "evidence_basis_ids": list(result.evidence_basis_ids),
        "result_digest": result.result_digest,
    }
    return {
        "schema": "petta-memory-pipln-validated-kernel-result-v1",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_validated_kernel_result(path: str | Path, result: ValidatedKernelResult) -> None:
    """Create one immutable validated-result artifact; never replace a path."""
    data = json.dumps(validated_kernel_result_document(result), sort_keys=True, indent=2) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_create_once_durable(destination, data)


def read_validated_kernel_result(
    path: str | Path,
    *,
    compiled: CompiledEpisodeInputs,
) -> ValidatedKernelResult:
    """Load a result and close its stamps and basis IDs against one episode."""
    if not isinstance(compiled, CompiledEpisodeInputs):
        raise ValueError("compiled must be immutable compiled episode inputs")
    document = _load_unambiguous_json(path)
    schema = "petta-memory-pipln-validated-kernel-result-v1"
    if (not isinstance(document, dict) or set(document) != {"schema", "payload", "document_digest"}
            or document.get("schema") != schema):
        raise ValueError("invalid validated kernel result document schema")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("document_digest") != _canonical_hash(payload):
        raise ValueError("validated kernel result document checksum mismatch")
    expected_fields = {
        "episode_id", "chart_fingerprint", "query_term", "strength", "confidence",
        "stamp_ints", "evidence_basis_ids", "result_digest",
    }
    if (set(payload) != expected_fields or not isinstance(payload["stamp_ints"], list)
            or not isinstance(payload["evidence_basis_ids"], list)):
        raise ValueError("invalid validated kernel result payload")
    if (payload["episode_id"] != compiled.episode_id
            or payload["chart_fingerprint"] != compiled.chart_fingerprint):
        raise ValueError("validated kernel result does not match compiled episode")

    stamps = tuple(payload["stamp_ints"])
    if (not stamps or any(isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0
                          for stamp in stamps)
            or tuple(sorted(set(stamps))) != stamps):
        raise ValueError("validated kernel result stamps must be canonical non-negative integers")
    if any(not isinstance(basis_id, str) or not basis_id.strip()
           for basis_id in payload["evidence_basis_ids"]):
        raise ValueError("invalid validated kernel result evidence basis ids")
    basis_by_stamp = {entry.stamp_int: entry.basis_id for entry in compiled.stamp_map}
    unknown = tuple(stamp for stamp in stamps if stamp not in basis_by_stamp)
    if unknown:
        raise ValueError(f"validated kernel result contains unknown episode stamps: {unknown}")
    expected_bases = tuple(basis_by_stamp[stamp] for stamp in stamps)
    if tuple(payload["evidence_basis_ids"]) != expected_bases:
        raise ValueError("validated kernel result evidence bases do not match compiled episode stamps")
    return ValidatedKernelResult(
        episode_id=payload["episode_id"],
        chart_fingerprint=payload["chart_fingerprint"],
        query_term=payload["query_term"],
        strength=payload["strength"],
        confidence=payload["confidence"],
        stamp_ints=stamps,
        evidence_basis_ids=tuple(payload["evidence_basis_ids"]),
        result_digest=payload["result_digest"],
    )


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Immutable packet selection plus the versions that define its meaning."""

    id: str
    packet_ids: tuple[str, ...]
    context_id: str
    assumption_fingerprint: str
    ontology_fingerprint: str
    created_at: str
    snapshot_fingerprint: str
    packet_content_digests: tuple[tuple[str, str], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in (
            "id", "context_id", "assumption_fingerprint", "ontology_fingerprint",
            "created_at", "snapshot_fingerprint",
        ):
            _nonempty(getattr(self, field), field)
        if not self.packet_ids or tuple(sorted(set(self.packet_ids))) != self.packet_ids:
            raise ValueError("packet_ids must be non-empty, unique, and sorted")
        for packet_id in self.packet_ids:
            _nonempty(packet_id, "packet_id")
        _sha256_digest(self.snapshot_fingerprint, "snapshot_fingerprint")
        digest_ids = tuple(packet_id for packet_id, _ in self.packet_content_digests)
        if digest_ids != self.packet_ids:
            raise ValueError("packet_content_digests must exactly match sorted packet_ids")
        for packet_id, digest in self.packet_content_digests:
            _nonempty(packet_id, "packet_content_digest packet_id")
            _sha256_digest(digest, "packet_content_digest")
        expected_fingerprint = _snapshot_fingerprint(
            self.packet_content_digests, context_id=self.context_id,
            assumption_fingerprint=self.assumption_fingerprint,
            ontology_fingerprint=self.ontology_fingerprint,
        )
        if self.snapshot_fingerprint != expected_fingerprint:
            raise ValueError("snapshot_fingerprint does not match packet content digests")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")


def _snapshot_packet_payload(packet: EvidencePacket) -> dict[str, object]:
    """Return the complete packet content committed by an evidence snapshot."""
    return {
        "id": packet.id,
        "statement": packet.statement,
        "context_id": packet.context_id,
        "positive_delta": packet.positive_delta,
        "negative_delta": packet.negative_delta,
        "token_ids": packet.token_ids,
        "source_reliability": packet.source_reliability,
        "temporal_relevance": packet.temporal_relevance,
        "status": packet.status,
        "assumption_fingerprint": packet.assumption_fingerprint,
        "ontology_fingerprint": packet.ontology_fingerprint,
        "created_by": packet.created_by,
        "parent_packet_ids": packet.parent_packet_ids,
        "schema_version": packet.schema_version,
    }


def build_evidence_snapshot(
    *, snapshot_id: str, packets: Iterable[EvidencePacket], context_id: str,
    assumption_fingerprint: str, ontology_fingerprint: str, created_at: str,
) -> EvidenceSnapshot:
    """Freeze active, version-compatible packets and derive a stable content fingerprint."""
    packet_list = tuple(packets)
    ids = tuple(sorted(packet.id for packet in packet_list))
    if len(ids) != len(set(ids)):
        raise ValueError("packet ids must be unique within a snapshot")
    for packet in packet_list:
        if packet.status != "ACTIVE":
            raise ValueError(f"snapshot packet is not ACTIVE: {packet.id}")
        if packet.context_id != context_id:
            raise ValueError(f"snapshot packet context mismatch: {packet.id}")
        if packet.assumption_fingerprint != assumption_fingerprint:
            raise ValueError(f"snapshot packet assumption mismatch: {packet.id}")
        if packet.ontology_fingerprint != ontology_fingerprint:
            raise ValueError(f"snapshot packet ontology mismatch: {packet.id}")
    ordered_packets = tuple(sorted(packet_list, key=lambda item: item.id))
    packet_payloads = [_snapshot_packet_payload(packet) for packet in ordered_packets]
    packet_content_digests = tuple(
        (packet.id, _canonical_hash(payload))
        for packet, payload in zip(ordered_packets, packet_payloads)
    )
    fingerprint = _snapshot_fingerprint(
        packet_content_digests, context_id=context_id,
        assumption_fingerprint=assumption_fingerprint,
        ontology_fingerprint=ontology_fingerprint,
    )
    return EvidenceSnapshot(snapshot_id, ids, context_id, assumption_fingerprint,
                            ontology_fingerprint, created_at, fingerprint,
                            packet_content_digests)


def evidence_snapshot_document(snapshot: EvidenceSnapshot) -> dict[str, object]:
    """Return the canonical, checksummed persistence envelope for a snapshot."""
    if not isinstance(snapshot, EvidenceSnapshot):
        raise ValueError("snapshot must be an immutable evidence snapshot")
    payload = {
        "id": snapshot.id,
        "packet_ids": list(snapshot.packet_ids),
        "context_id": snapshot.context_id,
        "assumption_fingerprint": snapshot.assumption_fingerprint,
        "ontology_fingerprint": snapshot.ontology_fingerprint,
        "created_at": snapshot.created_at,
        "snapshot_fingerprint": snapshot.snapshot_fingerprint,
        "packet_content_digests": [list(item) for item in snapshot.packet_content_digests],
        "schema_version": snapshot.schema_version,
    }
    return {
        "schema": "petta-memory-pipln-evidence-snapshot-v2",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_evidence_snapshot(path: str | Path, snapshot: EvidenceSnapshot) -> None:
    """Create an immutable snapshot document; never replace an existing path."""
    data = json.dumps(evidence_snapshot_document(snapshot), sort_keys=True, indent=2) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_create_once_durable(destination, data)


def read_evidence_snapshot(path: str | Path) -> EvidenceSnapshot:
    """Load a snapshot document and fail closed on schema or checksum drift."""
    document = _load_unambiguous_json(path)
    if (not isinstance(document, dict)
            or set(document) != {"schema", "payload", "document_digest"}
            or document.get("schema") != "petta-memory-pipln-evidence-snapshot-v2"):
        raise ValueError("invalid evidence snapshot document schema")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("document_digest") != _canonical_hash(payload):
        raise ValueError("evidence snapshot document checksum mismatch")
    expected = {
        "id", "packet_ids", "context_id", "assumption_fingerprint", "ontology_fingerprint",
        "created_at", "snapshot_fingerprint", "packet_content_digests", "schema_version",
    }
    if (set(payload) != expected or not isinstance(payload["packet_ids"], list)
            or not isinstance(payload["packet_content_digests"], list)
            or any(not isinstance(item, list) or len(item) != 2
                   for item in payload["packet_content_digests"])):
        raise ValueError("invalid evidence snapshot payload")
    return EvidenceSnapshot(
        id=payload["id"], packet_ids=tuple(payload["packet_ids"]), context_id=payload["context_id"],
        assumption_fingerprint=payload["assumption_fingerprint"],
        ontology_fingerprint=payload["ontology_fingerprint"], created_at=payload["created_at"],
        snapshot_fingerprint=payload["snapshot_fingerprint"],
        packet_content_digests=tuple(tuple(item) for item in payload["packet_content_digests"]),
        schema_version=payload["schema_version"],
    )


class EvidenceSnapshotRepository:
    """Content-addressed, create-once snapshot discovery boundary.

    Repository files are named by the semantic snapshot fingerprint. Lookup scans
    and validates every document so filename drift, duplicate snapshot IDs, and
    conflicting content fail closed instead of selecting an arbitrary record.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def snapshots(self) -> tuple[EvidenceSnapshot, ...]:
        if not self.root.exists():
            return ()
        if not self.root.is_dir():
            raise ValueError("snapshot repository root must be a directory")
        snapshots: list[EvidenceSnapshot] = []
        ids: set[str] = set()
        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.suffix != ".json":
                raise ValueError(f"unexpected snapshot repository entry: {path.name}")
            snapshot = read_evidence_snapshot(path)
            if path.stem != snapshot.snapshot_fingerprint:
                raise ValueError(f"snapshot filename fingerprint mismatch: {path.name}")
            if snapshot.id in ids:
                raise ValueError(f"duplicate snapshot id in repository: {snapshot.id}")
            ids.add(snapshot.id)
            snapshots.append(snapshot)
        return tuple(sorted(snapshots, key=lambda item: item.id))

    def get(self, snapshot_id: str) -> EvidenceSnapshot:
        _nonempty(snapshot_id, "snapshot_id")
        matches = [snapshot for snapshot in self.snapshots() if snapshot.id == snapshot_id]
        if not matches:
            raise KeyError(snapshot_id)
        return matches[0]

    def add(self, snapshot: EvidenceSnapshot) -> Path:
        existing = {item.id: item for item in self.snapshots()}
        if snapshot.id in existing:
            raise FileExistsError(f"snapshot id already exists: {snapshot.id}")
        destination = self.root / f"{snapshot.snapshot_fingerprint}.json"
        write_evidence_snapshot(destination, snapshot)
        return destination


@dataclass(frozen=True)
class PiContext:
    """Semantic context identity and its versioned applicability policies."""

    id: str
    language_fragment_id: str
    universe_id: str
    guard: str
    guard_version: str
    task_class: str
    assumption_package_id: str
    ontology_id: str
    ontology_version: str
    weakness_policy_id: str
    relevance_policy_id: str
    parent_context_ids: tuple[str, ...] = ()
    created_by_revision_id: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in (
            "id", "language_fragment_id", "universe_id", "guard", "guard_version",
            "task_class", "assumption_package_id", "ontology_id", "ontology_version",
            "weakness_policy_id", "relevance_policy_id",
        ):
            _nonempty(getattr(self, field), field)
        if tuple(sorted(set(self.parent_context_ids))) != self.parent_context_ids:
            raise ValueError("parent_context_ids must be unique and sorted")
        if self.id in self.parent_context_ids:
            raise ValueError("a context cannot be its own parent")
        if self.created_by_revision_id is not None:
            _nonempty(self.created_by_revision_id, "created_by_revision_id")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")


@dataclass(frozen=True)
class ChartPolicy:
    factorization_policy_id: str
    projection_policy_id: str
    kernel_projection_policy_id: str
    rule_profile_id: str
    kernel_version: str
    translator_version: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _nonempty(getattr(self, field), field)


@dataclass(frozen=True)
class PiChart:
    id: str
    context_id: str
    prior_strength_p0: float
    prior_weight_k: float
    prior_provenance: str
    policy: ChartPolicy
    selected_packet_ids: tuple[str, ...]
    evidence_snapshot_id: str
    evidence_snapshot_fingerprint: str
    adequacy_certificate_id: str
    chart_fingerprint: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in ("id", "context_id", "prior_provenance", "evidence_snapshot_id", "evidence_snapshot_fingerprint", "adequacy_certificate_id", "chart_fingerprint"):
            _nonempty(getattr(self, field), field)
        if isinstance(self.prior_strength_p0, bool) or not isinstance(self.prior_strength_p0, (int, float)) or not math.isfinite(self.prior_strength_p0) or not 0 <= self.prior_strength_p0 <= 1:
            raise ValueError("prior_strength_p0 must be finite and in [0, 1]")
        if isinstance(self.prior_weight_k, bool) or not isinstance(self.prior_weight_k, (int, float)) or not math.isfinite(self.prior_weight_k) or self.prior_weight_k <= 0:
            raise ValueError("prior_weight_k must be finite and positive")
        if not self.selected_packet_ids or tuple(sorted(set(self.selected_packet_ids))) != self.selected_packet_ids:
            raise ValueError("selected_packet_ids must be non-empty, unique, and sorted")
        for packet_id in self.selected_packet_ids:
            _nonempty(packet_id, "selected_packet_id")
        _sha256_digest(self.evidence_snapshot_fingerprint, "evidence_snapshot_fingerprint")
        _sha256_digest(self.chart_fingerprint, "chart_fingerprint")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")


def build_pi_chart(
    *, chart_id: str, context: PiContext, prior_strength_p0: float, prior_weight_k: float,
    prior_provenance: str, policy: ChartPolicy, selected_packet_ids: Iterable[str],
    evidence_snapshot: EvidenceSnapshot, adequacy_certificate_id: str,
) -> PiChart:
    """Freeze a chart against one validated immutable evidence snapshot."""
    if not isinstance(context, PiContext):
        raise ValueError("context must be an immutable pi context")
    if not isinstance(policy, ChartPolicy):
        raise ValueError("policy must be an immutable chart policy")
    if not isinstance(evidence_snapshot, EvidenceSnapshot):
        raise ValueError("evidence_snapshot must be an immutable evidence snapshot")
    supplied_packet_ids = tuple(selected_packet_ids)
    packet_ids = tuple(sorted(supplied_packet_ids))
    if not packet_ids:
        raise ValueError("selected_packet_ids must be non-empty")
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError("selected_packet_ids must be unique")
    for packet_id in packet_ids:
        _nonempty(packet_id, "selected_packet_id")
    if evidence_snapshot.context_id != context.id:
        raise ValueError("evidence snapshot context does not match chart context")
    missing_packet_ids = sorted(set(packet_ids) - set(evidence_snapshot.packet_ids))
    if missing_packet_ids:
        raise ValueError(
            "selected packets are absent from evidence snapshot: "
            + ", ".join(missing_packet_ids)
        )
    fingerprint = _canonical_hash({
        "context_id": context.id,
        "context_guard_version": context.guard_version,
        "ontology_version": context.ontology_version,
        "assumption_package_id": context.assumption_package_id,
        "prior_strength_p0": prior_strength_p0,
        "prior_weight_k": prior_weight_k,
        "factorization_policy_id": policy.factorization_policy_id,
        "projection_policy_id": policy.projection_policy_id,
        "kernel_projection_policy_id": policy.kernel_projection_policy_id,
        "rule_profile_id": policy.rule_profile_id,
        "selected_packet_ids": packet_ids,
        "evidence_snapshot_id": evidence_snapshot.id,
        "evidence_snapshot_fingerprint": evidence_snapshot.snapshot_fingerprint,
        "adequacy_certificate_id": adequacy_certificate_id,
        "kernel_version": policy.kernel_version,
        "translator_version": policy.translator_version,
    })
    return PiChart(chart_id, context.id, prior_strength_p0, prior_weight_k, prior_provenance,
                   policy, packet_ids, evidence_snapshot.id, evidence_snapshot.snapshot_fingerprint,
                   adequacy_certificate_id, fingerprint)


def evidence_basis_from_packet(
    packet: EvidencePacket,
    tokens: Iterable[EvidenceToken],
    *,
    independence_status: IndependenceStatus,
    justification_cid: str,
) -> EvidenceBasis:
    """Build one reviewed basis unit covering exactly one packet's token set.

    The caller must make the independence decision explicitly. This constructor
    only validates provenance closure and gives the basis a deterministic ID;
    it never infers disjointness from distinct token identifiers.
    """
    token_by_id: dict[str, EvidenceToken] = {}
    for token in tokens:
        if token.id in token_by_id:
            raise ValueError(f"duplicate token metadata: {token.id}")
        token_by_id[token.id] = token
    supplied_ids = tuple(sorted(token_by_id))
    if supplied_ids != packet.token_ids:
        missing = sorted(set(packet.token_ids) - set(supplied_ids))
        extra = sorted(set(supplied_ids) - set(packet.token_ids))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"extra: {', '.join(extra)}")
        raise ValueError(f"packet token metadata mismatch ({'; '.join(details)})")
    causal_groups = tuple(sorted({
        token.causal_group_id
        for token in token_by_id.values()
        if token.causal_group_id is not None
    }))
    basis_id = f"basis-{_canonical_hash({'packet_id': packet.id, 'token_ids': supplied_ids})}"
    return EvidenceBasis(basis_id, supplied_ids, causal_groups, independence_status, justification_cid)


def deterministic_stamp_map(episode_id: str, bases: Iterable[EvidenceBasis]) -> tuple[StampMapEntry, ...]:
    """Assign collision-free integers after sorting stable basis identifiers."""
    _nonempty(episode_id, "episode_id")
    ordered = sorted(bases, key=lambda basis: basis.basis_id)
    ids = [basis.basis_id for basis in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("basis_id values must be unique within an episode")
    return tuple(
        StampMapEntry(episode_id, index, basis.basis_id, _canonical_hash(basis.member_token_ids))
        for index, basis in enumerate(ordered)
    )


@dataclass(frozen=True)
class EvidenceContribution:
    """One channel-specific contribution from one evidence basis unit."""

    basis_id: str
    positive_weight: float = 0.0
    negative_weight: float = 0.0

    def __post_init__(self) -> None:
        _nonempty(self.basis_id, "basis_id")
        _finite_nonnegative(self.positive_weight, "positive_weight")
        _finite_nonnegative(self.negative_weight, "negative_weight")
        if self.positive_weight == 0 and self.negative_weight == 0:
            raise ValueError("an evidence contribution must have positive or negative weight")


@dataclass(frozen=True)
class EvidenceCapsule:
    """Exact evidence algebra keyed by basis, preventing duplicate count addition."""

    contributions: tuple[EvidenceContribution, ...]

    def __post_init__(self) -> None:
        ids = tuple(item.basis_id for item in self.contributions)
        if not ids or ids != tuple(sorted(set(ids))):
            raise ValueError("contributions must be non-empty, unique, and sorted by basis_id")

    @property
    def positive_count(self) -> float:
        return sum(item.positive_weight for item in self.contributions)

    @property
    def negative_count(self) -> float:
        return sum(item.negative_weight for item in self.contributions)

    @property
    def basis_ids(self) -> tuple[str, ...]:
        return tuple(item.basis_id for item in self.contributions)


def merge_evidence_capsules(
    left: EvidenceCapsule,
    right: EvidenceCapsule,
    *,
    bases: Iterable[EvidenceBasis] | None = None,
) -> EvidenceCapsule:
    """Union exact capsules by basis, optionally enforcing reviewed overlap metadata.

    With ``bases`` supplied, every contribution must resolve to one basis. Distinct
    bases may not share tokens, and UNKNOWN basis units cannot be combined with
    other units. This deliberately fails closed instead of treating partial or
    unknown overlap as independent evidence.
    """
    merged = {item.basis_id: item for item in left.contributions}
    for item in right.contributions:
        existing = merged.get(item.basis_id)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting contribution for shared basis_id: {item.basis_id}")
        merged[item.basis_id] = item

    if bases is not None:
        registry: dict[str, EvidenceBasis] = {}
        for basis in bases:
            if basis.basis_id in registry:
                raise ValueError(f"duplicate basis metadata: {basis.basis_id}")
            registry[basis.basis_id] = basis
        missing = sorted(set(merged) - set(registry))
        if missing:
            raise ValueError(f"missing basis metadata: {', '.join(missing)}")
        ordered_bases = [registry[basis_id] for basis_id in sorted(merged)]
        if len(ordered_bases) > 1:
            unknown = [basis.basis_id for basis in ordered_bases if basis.independence_status == "UNKNOWN"]
            if unknown:
                raise ValueError(f"cannot combine UNKNOWN basis units: {', '.join(unknown)}")
        for index, basis in enumerate(ordered_bases):
            tokens = set(basis.member_token_ids)
            for other in ordered_bases[index + 1 :]:
                overlap = sorted(tokens.intersection(other.member_token_ids))
                if overlap:
                    raise ValueError(
                        f"partial overlap between distinct bases {basis.basis_id} and "
                        f"{other.basis_id}: {', '.join(overlap)}"
                    )

    return EvidenceCapsule(tuple(merged[key] for key in sorted(merged)))


@dataclass(frozen=True)
class ProjectionRecord:
    positive_count: float
    negative_count: float
    evidence_mass: float
    conflict_balance: float
    signed_tendency: float
    strength: float
    confidence: float
    beta_alpha: float
    beta_beta: float

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field} must be a finite number")
        for field in ("positive_count", "negative_count", "evidence_mass", "conflict_balance", "strength", "confidence", "beta_alpha", "beta_beta"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")
        for field in ("conflict_balance", "strength", "confidence"):
            if getattr(self, field) > 1:
                raise ValueError(f"{field} must be in [0, 1]")
        if not -1 <= self.signed_tendency <= 1:
            raise ValueError("signed_tendency must be in [-1, 1]")


def canonical_projection_from_beta(
    beta_alpha: float,
    beta_beta: float,
    *,
    prior_strength: float,
    prior_weight: float,
) -> ProjectionRecord:
    """Recover empirical counts from beta parameters and reproject canonically.

    This is the reviewed prior-cycling boundary: the old prior pseudo-counts are
    subtracted before projection, so changing priors cannot turn prior mass into
    empirical evidence. Small floating-point cancellation errors are clamped;
    materially negative recovered counts fail closed.
    """
    _finite_nonnegative(beta_alpha, "beta_alpha")
    _finite_nonnegative(beta_beta, "beta_beta")
    if isinstance(prior_strength, bool) or not isinstance(prior_strength, (int, float)) or not math.isfinite(prior_strength) or not 0 <= prior_strength <= 1:
        raise ValueError("prior_strength must be finite and in [0, 1]")
    if isinstance(prior_weight, bool) or not isinstance(prior_weight, (int, float)) or not math.isfinite(prior_weight) or prior_weight <= 0:
        raise ValueError("prior_weight must be finite and positive")
    positive_count = beta_alpha - prior_weight * prior_strength
    negative_count = beta_beta - prior_weight * (1 - prior_strength)
    tolerance = 1e-12 * max(1.0, beta_alpha, beta_beta, prior_weight)
    if positive_count < -tolerance or negative_count < -tolerance:
        raise ValueError("beta parameters contain less mass than the declared prior")
    return canonical_local_chart_projection(
        max(0.0, positive_count), max(0.0, negative_count),
        prior_strength=prior_strength, prior_weight=prior_weight,
    )


def cycle_local_chart_prior(
    projection: ProjectionRecord,
    *,
    old_prior_strength: float,
    old_prior_weight: float,
    new_prior_strength: float,
    new_prior_weight: float,
) -> ProjectionRecord:
    """Remove one declared prior from a projection and apply another."""
    recovered = canonical_projection_from_beta(
        projection.beta_alpha, projection.beta_beta,
        prior_strength=old_prior_strength, prior_weight=old_prior_weight,
    )
    return canonical_local_chart_projection(
        recovered.positive_count, recovered.negative_count,
        prior_strength=new_prior_strength, prior_weight=new_prior_weight,
    )


def canonical_local_chart_projection(
    positive_count: float,
    negative_count: float,
    *,
    prior_strength: float,
    prior_weight: float,
) -> ProjectionRecord:
    """Implement specification policy ``pipl-local-chart-v1``."""
    _finite_nonnegative(positive_count, "positive_count")
    _finite_nonnegative(negative_count, "negative_count")
    if isinstance(prior_strength, bool) or not isinstance(prior_strength, (int, float)) or not math.isfinite(prior_strength) or not 0 <= prior_strength <= 1:
        raise ValueError("prior_strength must be finite and in [0, 1]")
    if isinstance(prior_weight, bool) or not isinstance(prior_weight, (int, float)) or not math.isfinite(prior_weight) or prior_weight <= 0:
        raise ValueError("prior_weight must be finite and positive")
    mass = positive_count + negative_count
    strength = (positive_count + prior_weight * prior_strength) / (mass + prior_weight)
    confidence = mass / (mass + prior_weight)
    conflict = 0.0 if mass == 0 else 2 * min(positive_count, negative_count) / mass
    tendency = 0.0 if mass == 0 else (positive_count - negative_count) / mass
    return ProjectionRecord(
        positive_count=float(positive_count),
        negative_count=float(negative_count),
        evidence_mass=float(mass),
        conflict_balance=float(conflict),
        signed_tendency=float(tendency),
        strength=float(strength),
        confidence=float(confidence),
        beta_alpha=float(prior_weight * prior_strength + positive_count),
        beta_beta=float(prior_weight * (1 - prior_strength) + negative_count),
    )


def compile_episode_inputs(
    *, episode_id: str, chart: PiChart, evidence_snapshot: EvidenceSnapshot,
    packets: Iterable[EvidencePacket], bases: Iterable[EvidenceBasis],
    max_sentences: int = DEFAULT_MAX_COMPILED_SENTENCES,
    max_atom_chars: int = DEFAULT_MAX_COMPILED_ATOM_CHARS,
) -> CompiledEpisodeInputs:
    """Compile validated chart evidence into deterministic patham9 inputs.

    This pure boundary does not invoke patham9, write a manifest, or persist a
    derived result. Each selected packet must close against the frozen snapshot
    and exactly one packet-derived evidence basis.
    """
    _nonempty(episode_id, "episode_id")
    _positive_int(max_sentences, "max_sentences")
    _positive_int(max_atom_chars, "max_atom_chars")
    if chart.evidence_snapshot_id != evidence_snapshot.id or chart.evidence_snapshot_fingerprint != evidence_snapshot.snapshot_fingerprint:
        raise ValueError("chart does not match evidence snapshot")
    if chart.context_id != evidence_snapshot.context_id:
        raise ValueError("chart context does not match evidence snapshot")

    packet_by_id: dict[str, EvidencePacket] = {}
    for packet in packets:
        if packet.id in packet_by_id:
            raise ValueError(f"duplicate packet: {packet.id}")
        packet_by_id[packet.id] = packet
    if set(packet_by_id) != set(chart.selected_packet_ids):
        raise ValueError("packets must exactly match the chart selection")
    snapshot_packet_digests = dict(evidence_snapshot.packet_content_digests)
    for packet_id, packet in packet_by_id.items():
        if _canonical_hash(_snapshot_packet_payload(packet)) != snapshot_packet_digests[packet_id]:
            raise ValueError(f"packet content does not match evidence snapshot: {packet_id}")
    if len(packet_by_id) > max_sentences:
        raise ValueError("chart selection exceeds max_sentences")

    basis_by_id: dict[str, EvidenceBasis] = {}
    for basis in bases:
        if basis.basis_id in basis_by_id:
            raise ValueError(f"duplicate basis: {basis.basis_id}")
        basis_by_id[basis.basis_id] = basis
    packet_basis: dict[str, EvidenceBasis] = {}
    for packet_id in chart.selected_packet_ids:
        packet = packet_by_id[packet_id]
        if packet.status != "ACTIVE" or packet.context_id != chart.context_id:
            raise ValueError(f"packet is not active in the chart context: {packet_id}")
        expected_basis_id = f"basis-{_canonical_hash({'packet_id': packet.id, 'token_ids': packet.token_ids})}"
        basis = basis_by_id.get(expected_basis_id)
        if basis is None or basis.member_token_ids != packet.token_ids:
            raise ValueError(f"missing exact packet basis: {packet_id}")
        packet_basis[packet_id] = basis
    if set(basis_by_id) != {basis.basis_id for basis in packet_basis.values()}:
        raise ValueError("bases must exactly match the chart packets")

    stamp_map = deterministic_stamp_map(episode_id, basis_by_id.values())
    stamp_by_basis = {entry.basis_id: entry.stamp_int for entry in stamp_map}
    sentences: list[CompiledSentence] = []
    emitted_atom_chars = 0
    for packet_id in chart.selected_packet_ids:
        packet = packet_by_id[packet_id]
        basis = packet_basis[packet_id]
        canonical_term = _canonical_kernel_term(packet.statement)
        projection = canonical_local_chart_projection(
            packet.positive_delta, packet.negative_delta,
            prior_strength=chart.prior_strength_p0, prior_weight=chart.prior_weight_k,
        )
        projection_id = _canonical_hash({
            "chart_fingerprint": chart.chart_fingerprint,
            "packet_id": packet.id,
            "policy_id": chart.policy.projection_policy_id,
            "positive_count": projection.positive_count,
            "negative_count": projection.negative_count,
            "strength": projection.strength,
            "confidence": projection.confidence,
        })
        stamps = (stamp_by_basis[basis.basis_id],)
        atom = (
            f"(Sentence ({canonical_term} (stv {projection.strength} "
            f"{projection.confidence})) ({' '.join(str(value) for value in stamps)}))"
        )
        emitted_atom_chars += len(atom)
        if emitted_atom_chars > max_atom_chars:
            raise ValueError("compiled Sentence atoms exceed max_atom_chars")
        meta = KernelSentenceMeta(
            episode_id=episode_id,
            sentence_digest=_canonical_hash({"atom": atom}),
            canonical_term=canonical_term,
            projection_id=projection_id,
            context_id=chart.context_id,
            chart_id=chart.id,
            stamp_ints=stamps,
            evidence_basis_ids=(basis.basis_id,),
        )
        sentences.append(CompiledSentence(atom, projection, meta))
    return CompiledEpisodeInputs(
        episode_id, chart.chart_fingerprint, evidence_snapshot.snapshot_fingerprint,
        stamp_map, tuple(sentences),
    )


def build_pettachainer_episode_contract(
    *, compiled: CompiledEpisodeInputs, query_term: str,
    max_atom_chars: int = DEFAULT_MAX_COMPILED_ATOM_CHARS,
) -> PeTTaChainerEpisodeContract:
    """Adapt immutable compiler output to PeTTaChainer add/query syntax.

    Truth values are copied from the chart projection.  Stamp and evidence-basis
    provenance remain in the typed sidecar because PeTTaChainer's public
    statement shape has no patham9 stamp field.  The returned atoms are inert;
    a separately bounded runtime gate must validate/add/query them.
    """
    if not isinstance(compiled, CompiledEpisodeInputs):
        raise ValueError("compiled must be immutable compiled episode inputs")
    _positive_int(max_atom_chars, "max_atom_chars")
    canonical_query = _canonical_kernel_term(query_term)
    if canonical_query != query_term:
        raise ValueError("query_term must already be canonical")

    statements: list[PeTTaChainerInputStatement] = []
    emitted_chars = 0
    for sentence in compiled.sentences:
        proof_id = f"pm-{sentence.meta.sentence_digest}"
        atom = (
            f"(: {proof_id} {sentence.meta.canonical_term} "
            f"(STV {sentence.projection.strength} {sentence.projection.confidence}))"
        )
        emitted_chars += len(atom)
        if emitted_chars > max_atom_chars:
            raise ValueError("PeTTaChainer contract atoms exceed max_atom_chars")
        statements.append(PeTTaChainerInputStatement(
            atom=atom,
            proof_id=proof_id,
            sentence_digest=sentence.meta.sentence_digest,
            canonical_term=sentence.meta.canonical_term,
            strength=sentence.projection.strength,
            confidence=sentence.projection.confidence,
            stamp_ints=sentence.meta.stamp_ints,
            evidence_basis_ids=sentence.meta.evidence_basis_ids,
        ))
    query_atom = f"(: $prf {canonical_query} $tv)"
    if emitted_chars + len(query_atom) > max_atom_chars:
        raise ValueError("PeTTaChainer contract atoms exceed max_atom_chars")
    return PeTTaChainerEpisodeContract(
        episode_id=compiled.episode_id,
        chart_fingerprint=compiled.chart_fingerprint,
        statements=tuple(statements),
        query_term=canonical_query,
        query_atom=query_atom,
    )


def compiled_episode_inputs_document(compiled: CompiledEpisodeInputs) -> dict[str, object]:
    """Return a canonical checksummed artifact for exact compiler-output replay."""
    if not isinstance(compiled, CompiledEpisodeInputs):
        raise ValueError("compiled must be immutable compiled episode inputs")
    payload = {
        "episode_id": compiled.episode_id,
        "chart_fingerprint": compiled.chart_fingerprint,
        "evidence_snapshot_fingerprint": compiled.evidence_snapshot_fingerprint,
        "stamp_map": [
            {
                "episode_id": entry.episode_id,
                "stamp_int": entry.stamp_int,
                "basis_id": entry.basis_id,
                "member_token_digest": entry.member_token_digest,
            }
            for entry in compiled.stamp_map
        ],
        "sentences": [
            {
                "atom": sentence.atom,
                "projection": {
                    field: getattr(sentence.projection, field)
                    for field in sentence.projection.__dataclass_fields__
                },
                "meta": {
                    "episode_id": sentence.meta.episode_id,
                    "sentence_digest": sentence.meta.sentence_digest,
                    "canonical_term": sentence.meta.canonical_term,
                    "projection_id": sentence.meta.projection_id,
                    "context_id": sentence.meta.context_id,
                    "chart_id": sentence.meta.chart_id,
                    "stamp_ints": list(sentence.meta.stamp_ints),
                    "evidence_basis_ids": list(sentence.meta.evidence_basis_ids),
                },
            }
            for sentence in compiled.sentences
        ],
    }
    return {
        "schema": "petta-memory-pipln-compiled-episode-inputs-v1",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_compiled_episode_inputs(path: str | Path, compiled: CompiledEpisodeInputs) -> None:
    """Create one immutable compiler-output artifact; never replace a path."""
    data = json.dumps(compiled_episode_inputs_document(compiled), sort_keys=True, indent=2) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_create_once_durable(destination, data)


def read_compiled_episode_inputs(path: str | Path) -> CompiledEpisodeInputs:
    """Load and fully validate an immutable compiler-output artifact."""
    document = _load_unambiguous_json(path)
    if not isinstance(document, dict) or document.get("schema") != "petta-memory-pipln-compiled-episode-inputs-v1":
        raise ValueError("invalid compiled episode inputs document schema")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("document_digest") != _canonical_hash(payload):
        raise ValueError("compiled episode inputs document checksum mismatch")
    expected_payload = {
        "episode_id", "chart_fingerprint", "evidence_snapshot_fingerprint", "stamp_map", "sentences",
    }
    if set(payload) != expected_payload or not isinstance(payload["stamp_map"], list) or not isinstance(payload["sentences"], list):
        raise ValueError("invalid compiled episode inputs payload")
    stamp_fields = {"episode_id", "stamp_int", "basis_id", "member_token_digest"}
    stamps: list[StampMapEntry] = []
    for item in payload["stamp_map"]:
        if not isinstance(item, dict) or set(item) != stamp_fields:
            raise ValueError("invalid compiled episode stamp map")
        stamps.append(StampMapEntry(**item))
    projection_fields = set(ProjectionRecord.__dataclass_fields__)
    meta_fields = {
        "episode_id", "sentence_digest", "canonical_term", "projection_id", "context_id",
        "chart_id", "stamp_ints", "evidence_basis_ids",
    }
    sentences: list[CompiledSentence] = []
    for item in payload["sentences"]:
        if not isinstance(item, dict) or set(item) != {"atom", "projection", "meta"}:
            raise ValueError("invalid compiled sentence payload")
        projection = item["projection"]
        meta = item["meta"]
        if not isinstance(projection, dict) or set(projection) != projection_fields:
            raise ValueError("invalid compiled sentence projection")
        if not isinstance(meta, dict) or set(meta) != meta_fields:
            raise ValueError("invalid compiled sentence metadata")
        if not isinstance(meta["stamp_ints"], list) or not isinstance(meta["evidence_basis_ids"], list):
            raise ValueError("invalid compiled sentence metadata collections")
        sentence_meta = KernelSentenceMeta(
            episode_id=meta["episode_id"], sentence_digest=meta["sentence_digest"],
            canonical_term=meta["canonical_term"], projection_id=meta["projection_id"],
            context_id=meta["context_id"], chart_id=meta["chart_id"],
            stamp_ints=tuple(meta["stamp_ints"]), evidence_basis_ids=tuple(meta["evidence_basis_ids"]),
        )
        sentences.append(CompiledSentence(item["atom"], ProjectionRecord(**projection), sentence_meta))
    return CompiledEpisodeInputs(
        episode_id=payload["episode_id"], chart_fingerprint=payload["chart_fingerprint"],
        evidence_snapshot_fingerprint=payload["evidence_snapshot_fingerprint"],
        stamp_map=tuple(stamps), sentences=tuple(sentences),
    )
