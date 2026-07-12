"""Typed Phase-1 domain primitives for the patham9-backed πPLN runtime.

These records are deliberately independent of the persistent MediumMemoryStore.
They model immutable evidence and deterministic episode projection inputs without
turning derived STVs or chart priors into empirical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Iterable, Literal


IndependenceStatus = Literal["PROVEN_DISJOINT", "ASSUMED", "COUPLED", "UNKNOWN"]
PacketStatus = Literal["ACTIVE", "QUARANTINED", "RETRACTED"]
PacketOrigin = Literal["OBSERVATION", "IMPORT", "REVIEWED_EXPORT"]


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
