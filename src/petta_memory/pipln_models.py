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
from typing import Callable, Iterable, Literal

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
_EXECUTABLE_TERM_HEADS = frozenset({
    "!", "bind!", "case", "collapse", "eval", "if", "import!", "include", "let", "let*",
    "match", "pragma!", "superpose",
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
    payload = _episode_manifest_payload(manifest)
    return {
        "schema": "petta-memory-pipln-episode-manifest-v1",
        "payload": payload,
        "document_digest": _canonical_hash(payload),
    }


def write_episode_manifest(path: str | Path, manifest: EpisodeManifest) -> None:
    """Create one immutable episode-manifest artifact; never replace a path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(episode_manifest_document(manifest), sort_keys=True, indent=2) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def read_episode_manifest(path: str | Path) -> EpisodeManifest:
    """Load a checksummed manifest and reconstruct all typed invariants."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
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
    return EpisodeManifest(**values)


def validated_kernel_result_document(result: ValidatedKernelResult) -> dict[str, object]:
    """Return a checksummed artifact for one provenance-closed kernel result."""
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(validated_kernel_result_document(result), sort_keys=True, indent=2) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def read_validated_kernel_result(
    path: str | Path,
    *,
    compiled: CompiledEpisodeInputs,
) -> ValidatedKernelResult:
    """Load a result and close its stamps and basis IDs against one episode."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(evidence_snapshot_document(snapshot), sort_keys=True, indent=2) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def read_evidence_snapshot(path: str | Path) -> EvidenceSnapshot:
    """Load a snapshot document and fail closed on schema or checksum drift."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != "petta-memory-pipln-evidence-snapshot-v2":
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


def compiled_episode_inputs_document(compiled: CompiledEpisodeInputs) -> dict[str, object]:
    """Return a canonical checksummed artifact for exact compiler-output replay."""
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(compiled_episode_inputs_document(compiled), sort_keys=True, indent=2) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def read_compiled_episode_inputs(path: str | Path) -> CompiledEpisodeInputs:
    """Load and fully validate an immutable compiler-output artifact."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
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
