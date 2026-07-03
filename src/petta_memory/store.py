from __future__ import annotations

from dataclasses import dataclass
import fcntl
import math
from pathlib import Path
import os
import re
import tempfile
from typing import Callable, Iterable, Optional

from .sexpr import SExpressionSyntaxError, SExpr, parse_one_list, parse_top_level_lists, symbol_text, to_source


class ValidationError(ValueError):
    """Raised when a memory atom or cluster is invalid."""


SCHEMA_VERSION = "medium-memory-v1"
_BEGIN_PREFIX = ";;; BEGIN MemoryCluster "
_END_PREFIX = ";;; END MemoryCluster "
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_REQUIRED_CLUSTER_PREDICATES = (
    "MemoryCluster",
    "SchemaVersion",
    "ClusterType",
    "ClusterOpenedAt",
    "Contains",
)
_DEFAULT_PLN_EXCLUDED = {
    "RawUtterance",
    "ClaimText",
    "Said",
    "SpeechEvent",
    "QuotedClaim",
    "ClaimSource",
    "ClaimStatus",
}
_ID_DECLARING_PREDICATES = {
    "MemoryCluster",
    "ObservedEvent",
    "SpeechEvent",
    "QuotedClaim",
    "DerivedBelief",
    "Decision",
    "Hypothesis",
    "OpenQuestion",
    "Commitment",
    "Boundary",
    "Artifact",
    "StatusEvent",
    "SalienceEvent",
    "PromotionEvent",
    "TruthValueEvent",
}
_BINARY_RELATION_PREDICATES = {
    "SchemaVersion",
    "ClusterType",
    "ClusterOpenedAt",
    "ClusterSource",
    "HasProvenance",
    "ClusterStatus",
    "EpistemicRole",
    "About",
    "HasStatus",
    "StatusSubject",
    "StatusValue",
    "Supersedes",
    "SalienceSubject",
    "SalienceValue",
    "CommitmentText",
    "QuestionText",
    "BoundaryName",
    "ArtifactPath",
    "RawUtterance",
    "ClaimText",
    "ClaimSource",
    "ClaimStatus",
    "Said",
    "BeliefContent",
    "TruthValue",
    "EvidenceFor",
    "EvidenceSupportCount",
    "EvidenceOppositionCount",
    "PromotesFrom",
    "PromotesTo",
    "PromotionRule",
    "PromotionTrust",
    "PromotionDomain",
}


@dataclass(frozen=True)
class MemoryCluster:
    """A complete append unit from the canonical journal."""

    cluster_id: str
    atoms: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.atoms) + "\n"

    @property
    def record_text(self) -> str:
        """Canonical serialized form with unambiguous record delimiters."""
        return f"{_BEGIN_PREFIX}{self.cluster_id}\n{self.text}{_END_PREFIX}{self.cluster_id}\n"


class MediumMemoryStore:
    """Append-only `.metta` cluster journal with bounded deterministic queries.

    The canonical journal is auditable text. Each write is one delimited
    `MemoryCluster` record, written through a temporary file replacement to avoid
    partial-record corruption in the local prototype. Full AtomSpace/PLN loading is
    intentionally left to later integration code.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_atom_chars: int = 4096,
        max_cluster_chars: int = 65536,
        max_query_chars: int = 20000,
        parse_checker: Optional[Callable[[MemoryCluster], None]] = None,
    ) -> None:
        self.path = Path(path)
        self.max_atom_chars = max_atom_chars
        self.max_cluster_chars = max_cluster_chars
        self.max_query_chars = max_query_chars
        self.parse_checker = parse_checker

    def append_cluster(self, text: str) -> MemoryCluster:
        cluster = self.validate_cluster(text)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._reject_duplicate_ids(cluster)
                old = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
                separator = "" if not old or old.endswith("\n\n") else "\n"
                new = old + separator + cluster.record_text
                fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(new)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_name, self.path)
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return cluster

    def validate_cluster(self, text: str) -> MemoryCluster:
        if len(text.encode("utf-8")) > self.max_cluster_chars:
            raise ValidationError("cluster exceeds max_cluster_chars")
        atoms = tuple(_parse_atom_texts(text))
        if not atoms:
            raise ValidationError("cluster is empty")
        for atom in atoms:
            self.validate_atom(atom)
        cluster_ids = _objects_for_predicate(atoms, "MemoryCluster")
        if len(cluster_ids) != 1:
            raise ValidationError("cluster must contain exactly one MemoryCluster atom")
        cluster_id = cluster_ids[0]
        if not _ID_RE.match(cluster_id):
            raise ValidationError(f"invalid cluster id: {cluster_id}")
        for pred in _REQUIRED_CLUSTER_PREDICATES:
            if not _has_predicate(atoms, pred):
                raise ValidationError(f"missing required cluster predicate: {pred}")
        versions = _second_objects_for_subject(atoms, "SchemaVersion", cluster_id)
        if versions != [SCHEMA_VERSION]:
            raise ValidationError(f"cluster must declare (SchemaVersion {cluster_id} {SCHEMA_VERSION})")
        if not (_has_predicate(atoms, "ClusterSource") or _has_predicate(atoms, "HasProvenance")):
            raise ValidationError("cluster needs ClusterSource or HasProvenance")
        self._validate_id_declarations(atoms)
        self._validate_relation_arities(atoms)
        self._validate_relation_values(atoms)
        ids = _declared_ids(atoms)
        invalid_ids = sorted({ident for ident in ids if not _ID_RE.match(ident)})
        if invalid_ids:
            raise ValidationError(f"invalid declared ids: {', '.join(invalid_ids)}")
        duplicates = sorted({ident for ident in ids if ids.count(ident) > 1})
        if duplicates:
            raise ValidationError(f"duplicate declared ids in cluster: {', '.join(duplicates)}")
        self._validate_contains_edges(atoms, cluster_id=cluster_id, declared_ids=set(ids))
        cluster = MemoryCluster(cluster_id=cluster_id, atoms=atoms)
        self._run_parse_checker(cluster)
        return cluster

    def _validate_id_declarations(self, atoms: tuple[str, ...]) -> None:
        """Ensure unary type/id declarations cannot hide extra fields.

        Predicates such as ``ObservedEvent`` and ``Decision`` introduce record
        ids. Keeping those declarations to exactly one id argument avoids
        accepting ambiguous append units that later index/query code would
        silently collapse to the first argument.
        """
        for atom in atoms:
            parsed = _parse_single_atom(atom)
            if parsed and parsed[0] in _ID_DECLARING_PREDICATES and len(parsed) != 2:
                raise ValidationError(f"id declaration must have exactly one id argument: {atom}")

    def _validate_relation_arities(self, atoms: tuple[str, ...]) -> None:
        """Reject schema/retrieval relations with hidden extra fields.

        Most v0 metadata predicates are binary relations of the form
        ``(Predicate subject object)``. Enforcing exact arity keeps append units
        from smuggling additional arguments that direct query/index/prompt code
        would otherwise ignore when it reads only the subject/object pair.
        """
        for atom in atoms:
            parsed = _parse_single_atom(atom)
            if parsed and parsed[0] in _BINARY_RELATION_PREDICATES and len(parsed) != 3:
                raise ValidationError(f"binary relation must have exactly subject and object arguments: {atom}")

    def _validate_relation_values(self, atoms: tuple[str, ...]) -> None:
        """Validate values whose range is part of the v0 schema boundary."""
        for atom in atoms:
            parsed = _parse_single_atom(atom)
            if parsed and parsed[0] in {"EvidenceSupportCount", "EvidenceOppositionCount"}:
                value = _render_sexpr(parsed[2])
                if not _is_non_negative_number(value):
                    raise ValidationError(f"evidence count must be non-negative numeric value: {atom}")

    def _validate_contains_edges(self, atoms: tuple[str, ...], *, cluster_id: str, declared_ids: set[str]) -> None:
        """Ensure the cluster envelope only lists local declared memory records.

        `Contains` is the read/write boundary for one append unit: a cluster may
        contain records declared inside the same append, but should not appear to
        claim ownership of external ids or use a mismatched cluster id.
        """
        contained: list[str] = []
        for atom in atoms:
            parsed = _parse_single_atom(atom)
            if parsed and parsed[0] == "Contains":
                if len(parsed) != 3:
                    raise ValidationError(f"Contains must have exactly cluster and target ids: {atom}")
                owner = _render_sexpr(parsed[1])
                target = _render_sexpr(parsed[2])
                if owner != cluster_id:
                    raise ValidationError(f"Contains owner must be cluster id {cluster_id}: {atom}")
                if target == cluster_id:
                    raise ValidationError(f"Contains target must be a record id, not the cluster id: {atom}")
                contained.append(target)
        missing = sorted({target for target in contained if target not in declared_ids})
        if missing:
            raise ValidationError(f"Contains target is not declared in cluster: {', '.join(missing)}")

    def _run_parse_checker(self, cluster: MemoryCluster) -> None:
        """Run an optional external runtime parser check after local validation.

        This is a deliberately small integration seam for a future PeTTa/MeTTa
        runtime parse check. The checker receives the canonicalized cluster and
        should raise on parse failure; external runtime setup stays outside the
        store so v0 remains local and dependency-free.
        """
        if self.parse_checker is None:
            return
        try:
            self.parse_checker(cluster)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"external parse checker rejected cluster {cluster.cluster_id}: {exc}") from exc

    def _reject_duplicate_ids(self, cluster: MemoryCluster) -> None:
        existing_ids: set[str] = set()
        for existing in self.clusters():
            existing_ids.update(_declared_ids(existing.atoms))
        duplicates = sorted(existing_ids & set(_declared_ids(cluster.atoms)))
        if duplicates:
            raise ValidationError(f"duplicate ids already exist: {', '.join(duplicates)}")

    def validate_atom(self, atom: str) -> None:
        if len(atom.encode("utf-8")) > self.max_atom_chars:
            raise ValidationError("atom exceeds max_atom_chars")
        try:
            parsed = _parse_single_atom(atom)
        except ValidationError as exc:
            raise ValidationError(f"malformed atom: {atom}") from exc
        if not _is_symbol(parsed[0]):
            raise ValidationError(f"malformed atom: {atom}")

    def clusters(self) -> list[MemoryCluster]:
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        chunks = _split_cluster_chunks(text)
        return [self.validate_cluster(chunk) for chunk in chunks if chunk.strip()]

    def tail(self, max_chars: Optional[int] = None) -> str:
        limit = max_chars if max_chars is not None else self.max_query_chars
        if limit < 0:
            raise ValidationError("max_chars must be non-negative")
        if not self.path.exists():
            return ""
        text = self.path.read_text(encoding="utf-8")
        return text[-limit:]

    def audit_view(self, *, limit_chars: Optional[int] = None) -> str:
        """Return recent complete canonical cluster records for human audit.

        Unlike ``tail()``, this view never slices through a MemoryCluster record.
        It keeps the newest records that fit within the budget and returns them
        in journal order, preserving begin/end delimiters for review tools.
        """
        limit = limit_chars if limit_chars is not None else self.max_query_chars
        if limit < 0:
            raise ValidationError("limit_chars must be non-negative")
        selected: list[MemoryCluster] = []
        used = 0
        for cluster in reversed(self.clusters()):
            record = cluster.record_text
            separator = 1 if selected else 0
            addition = len(record) + separator
            if used + addition > limit:
                break
            selected.append(cluster)
            used += addition
        return "\n".join(cluster.record_text.rstrip("\n") for cluster in reversed(selected)) + ("\n" if selected else "")

    def query_id(self, identifier: str, *, limit: int = 20) -> list[MemoryCluster]:
        return self._bounded([c for c in self.clusters() if _cluster_mentions_id(c, identifier)], limit)

    def query_cluster(self, cluster_id: str) -> Optional[MemoryCluster]:
        for cluster in self.clusters():
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def query_type(self, type_name: str, *, limit: int = 20) -> list[MemoryCluster]:
        pat = re.compile(rf"^\({re.escape(type_name)}\s+", re.MULTILINE)
        return self._bounded([c for c in self.clusters() if pat.search(c.text)], limit)

    def query_about(self, entity: str, *, limit: int = 20) -> list[MemoryCluster]:
        pat = re.compile(rf"^\(About\s+[^\s()]+\s+{re.escape(entity)}\)", re.MULTILINE)
        return self._bounded([c for c in self.clusters() if pat.search(c.text)], limit)

    def query_status(self, status: str, *, limit: int = 20) -> list[MemoryCluster]:
        matches: list[MemoryCluster] = []
        superseded_events = self._superseded_event_ids()
        for cluster in self.clusters():
            matched = False
            for event_id in _objects_for_predicate(cluster.atoms, "StatusEvent"):
                values = _second_objects_for_subject(cluster.atoms, "StatusValue", event_id)
                if values and values[-1] == status and event_id not in superseded_events:
                    matched = True
            if not matched and re.search(rf"\((?:ClusterStatus|HasStatus)\s+[^\s()]+\s+{re.escape(status)}\)", cluster.text):
                matched = True
            if matched:
                matches.append(cluster)
        return self._bounded(matches, limit)

    def query_role(self, role: str, *, limit: int = 20) -> list[MemoryCluster]:
        pat = re.compile(rf"^\(EpistemicRole\s+[^\s()]+\s+{re.escape(role)}\)", re.MULTILINE)
        return self._bounded([c for c in self.clusters() if pat.search(c.text)], limit)

    def current_status(self, subject_id: str) -> Optional[str]:
        current: Optional[str] = None
        superseded_events = self._superseded_event_ids()
        events: list[tuple[str, str]] = []
        for cluster in self.clusters():
            atoms = cluster.atoms
            for event_id in _objects_for_predicate(atoms, "StatusEvent"):
                if _second_objects_for_subject(atoms, "StatusSubject", event_id) == [subject_id]:
                    values = _second_objects_for_subject(atoms, "StatusValue", event_id)
                    if values:
                        events.append((event_id, values[-1]))
        for event_id, value in events:
            if event_id not in superseded_events:
                current = value
        return current

    def _superseded_event_ids(self) -> set[str]:
        superseded_events: set[str] = set()
        for cluster in self.clusters():
            superseded_events.update(_second_objects_for_predicate(cluster.atoms, "Supersedes"))
        return superseded_events

    def prompt_view(
        self,
        *,
        limit_chars: int = 4000,
        topics: Optional[set[str]] = None,
        statuses: Optional[set[str]] = None,
    ) -> str:
        if limit_chars < 0:
            raise ValidationError("limit_chars must be non-negative")
        pieces: list[str] = []
        allowed = {
            "CommitmentText",
            "QuestionText",
            "BoundaryName",
            "ArtifactPath",
            "HasStatus",
            "ClusterStatus",
            "StatusValue",
            "About",
            "SalienceValue",
        }
        clusters = self._prompt_ranked_clusters(topics=topics or set(), statuses=statuses or set())
        for cluster in clusters:
            for atom in cluster.atoms:
                if _predicate(atom) in allowed:
                    pieces.append(atom)
        return _join_bounded_atom_lines(pieces, limit_chars)

    def _prompt_ranked_clusters(self, *, topics: set[str], statuses: set[str]) -> list[MemoryCluster]:
        clusters = self.clusters()
        recency_rank = {cluster.cluster_id: index for index, cluster in enumerate(reversed(clusters))}
        return sorted(
            clusters,
            key=lambda cluster: (
                -_prompt_cluster_score(cluster, topics=topics, statuses=statuses),
                recency_rank[cluster.cluster_id],
            ),
        )

    def index_view(self, *, limit_chars: int = 8000) -> str:
        """Return a bounded generated index for deterministic retrieval.

        The index is derived from the append-only journal and is not written back
        to it.  It keeps retrieval predicates small and explicit so callers can
        inspect which cluster introduced each id/type/about/status/role edge.
        """
        if limit_chars < 0:
            raise ValidationError("limit_chars must be non-negative")
        atoms = _index_atoms(self.clusters(), superseded_event_ids=self._superseded_event_ids())
        return _join_bounded_atom_lines(atoms, limit_chars)

    def pettachainer_evidence_view(self, *, limit_chars: Optional[int] = None) -> str:
        """Export promoted beliefs as PeTTaChainer proof statements.

        This is a runtime-specific normalized PLN view.  Each eligible
        ``DerivedBelief`` becomes a PeTTaChainer-checkable atom of the form
        ``(: proof-id statement (STV strength confidence))``.  The source
        ``TruthValue`` strength is preserved, while confidence is capped by the
        bounded ``PromotionTrust`` value so the explicit promotion decision can
        only reduce, not inflate, the exported confidence. Evidence-count/EC
        export is intentionally left for a later mapping once support and
        opposition counts are represented explicitly in the memory schema.
        """
        if limit_chars is not None and limit_chars < 0:
            raise ValidationError("limit_chars must be non-negative")
        promoted_beliefs = self._promoted_belief_metadata()
        atoms: list[str] = []
        for cluster in self.clusters():
            for belief_id, meta in promoted_beliefs.items():
                contents = _second_objects_for_subject(cluster.atoms, "BeliefContent", belief_id)
                truth_values = _second_objects_for_subject(cluster.atoms, "TruthValue", belief_id)
                if not contents or not truth_values:
                    continue
                stv = _pettachainer_stv(truth_values[-1], promotion_trust=meta["trust"])
                if stv:
                    atoms.append(f"(: {belief_id} {contents[-1]} {stv})")
        if limit_chars is not None:
            return _join_bounded_atom_lines(atoms, limit_chars)
        return "\n".join(atoms) + ("\n" if atoms else "")

    def pettachainer_evidence_packet_view(self, *, limit_chars: Optional[int] = None) -> str:
        """Export promoted beliefs with explicit EC counts as EvidencePackets.

        This view is emitted only when a promoted ``DerivedBelief`` has both
        ``EvidenceSupportCount`` and ``EvidenceOppositionCount`` atoms.  The
        packet shape follows PeTTaChainer's context modules:
        ``(EvidencePacket statement (EC pos neg) features provenance)``.  Counts
        are explicit schema data; they are not inferred from ``TruthValue``.
        """
        if limit_chars is not None and limit_chars < 0:
            raise ValidationError("limit_chars must be non-negative")
        promoted_beliefs = self._promoted_belief_metadata()
        atoms: list[str] = []
        for cluster in self.clusters():
            for belief_id, meta in promoted_beliefs.items():
                contents = _second_objects_for_subject(cluster.atoms, "BeliefContent", belief_id)
                support_counts = _second_objects_for_subject(cluster.atoms, "EvidenceSupportCount", belief_id)
                opposition_counts = _second_objects_for_subject(cluster.atoms, "EvidenceOppositionCount", belief_id)
                if not (contents and support_counts and opposition_counts):
                    continue
                support = support_counts[-1]
                opposition = opposition_counts[-1]
                if not (_is_non_negative_number(support) and _is_non_negative_number(opposition)):
                    continue
                atoms.append(
                    f"(EvidencePacket {contents[-1]} (EC {support} {opposition}) "
                    f"((domain {meta['domain']}) (promotion-rule {meta['rule']})) {meta['event']})"
                )
        if limit_chars is not None:
            return _join_bounded_atom_lines(atoms, limit_chars)
        return "\n".join(atoms) + ("\n" if atoms else "")

    def pettachainer_handoff_cache(
        self,
        *,
        cache_id: str = "petta-memory-pettachainer-handoff",
        statement_checker: Optional[Callable[[str], bool]] = None,
    ) -> dict[str, object]:
        """Return a non-live cache of PLN-ready PeTTaChainer handoff inputs.

        The cache deliberately contains precompiled/exported atoms only:
        promoted STV proof statements and promoted EvidencePackets with explicit
        support/opposition counts.  It is a stable handoff contract for review,
        OmegaClaw/GoalChainer mapping, and future PeTTaChainer ingestion; it is
        not appended to the journal and its items are not inferred beliefs.

        A caller may pass ``statement_checker`` (for example
        ``pettachainer.check_stmt``) to require runtime validation of every STV
        proof statement before the cache is accepted.  Without that opt-in hook,
        items are still limited to locally validated, promotion-eligible exports
        but are marked as not runtime-checked in the cache metadata.
        """
        if not _ID_RE.match(cache_id):
            raise ValidationError(f"invalid cache id: {cache_id}")
        promoted_beliefs = self._promoted_belief_metadata()
        items: list[dict[str, str]] = []
        statement_check_mode = "runtime-checker-not-run"
        if statement_checker is not None:
            statement_check_mode = "runtime-checker-passed"
        for cluster in self.clusters():
            for belief_id, meta in promoted_beliefs.items():
                contents = _second_objects_for_subject(cluster.atoms, "BeliefContent", belief_id)
                truth_values = _second_objects_for_subject(cluster.atoms, "TruthValue", belief_id)
                if contents and truth_values:
                    stv = _pettachainer_stv(truth_values[-1], promotion_trust=meta["trust"])
                    if stv:
                        atom = f"(: {belief_id} {contents[-1]} {stv})"
                        if statement_checker is not None:
                            try:
                                ok = statement_checker(atom)
                            except Exception as exc:
                                raise ValidationError(f"statement checker rejected {belief_id}: {exc}") from exc
                            if not ok:
                                raise ValidationError(f"statement checker rejected {belief_id}")
                        items.append(
                            {
                                "kind": "pettachainer-stv-statement",
                                "atom": atom,
                                "belief_id": belief_id,
                                "cluster_id": cluster.cluster_id,
                                "promotion_event": meta["event"],
                                "promotion_rule": meta["rule"],
                                "promotion_domain": meta["domain"],
                                "promotion_trust": meta["trust"],
                                "item_status": "pln-ready-input-not-inferred-belief",
                                "statement_check": statement_check_mode,
                            }
                        )
                support_counts = _second_objects_for_subject(cluster.atoms, "EvidenceSupportCount", belief_id)
                opposition_counts = _second_objects_for_subject(cluster.atoms, "EvidenceOppositionCount", belief_id)
                if contents and support_counts and opposition_counts:
                    support = support_counts[-1]
                    opposition = opposition_counts[-1]
                    if _is_non_negative_number(support) and _is_non_negative_number(opposition):
                        atom = (
                            f"(EvidencePacket {contents[-1]} (EC {support} {opposition}) "
                            f"((domain {meta['domain']}) (promotion-rule {meta['rule']})) {meta['event']})"
                        )
                        items.append(
                            {
                                "kind": "pettachainer-evidence-packet",
                                "atom": atom,
                                "belief_id": belief_id,
                                "cluster_id": cluster.cluster_id,
                                "promotion_event": meta["event"],
                                "promotion_rule": meta["rule"],
                                "promotion_domain": meta["domain"],
                                "promotion_trust": meta["trust"],
                                "item_status": "pln-ready-input-not-inferred-belief",
                                "evidence_count_check": "explicit-non-negative-support-opposition",
                            }
                        )
        return {
            "schema": "petta-memory-pettachainer-handoff-v1",
            "cache_id": cache_id,
            "mode": "non-live-precompiled-statement-cache",
            "compileadd_gate": "disabled-pending-materialize-stmt-lambdas-mm2compile-instrumentation-or-precompiled-add-api",
            "canonical_source": "append-only MemoryCluster journal via promotion-eligible PeTTaChainer views",
            "boundary": "read-only handoff artifact; do not append as memory and do not treat as inferred belief",
            "item_count": len(items),
            "items": items,
        }

    def pln_view(
        self,
        *,
        excluded_predicates: Optional[set[str]] = None,
        normalized: bool = False,
        limit_chars: Optional[int] = None,
    ) -> str:
        if limit_chars is not None and limit_chars < 0:
            raise ValidationError("limit_chars must be non-negative")
        excluded = _DEFAULT_PLN_EXCLUDED | (excluded_predicates or set())
        safe_atoms: list[str] = []
        promoted_beliefs = self._promoted_belief_metadata()
        promoted_belief_ids = set(promoted_beliefs)
        excluded_ids = self._pln_excluded_subject_ids(excluded)
        for cluster in self.clusters():
            for atom in cluster.atoms:
                pred = _predicate(atom)
                if pred in excluded:
                    continue
                subject = _first_arg(atom)
                if subject in excluded_ids:
                    continue
                if _atom_mentions_any(atom, excluded_ids) and pred in {"About", "EvidenceFor", "HasProvenance", "EpistemicRole"}:
                    continue
                if pred in {"DerivedBelief", "BeliefContent", "TruthValue", "EvidenceFor"}:
                    if subject not in promoted_belief_ids:
                        continue
                safe_atoms.append(atom)
                if normalized and pred == "DerivedBelief" and subject in promoted_beliefs:
                    meta = promoted_beliefs[subject]
                    safe_atoms.extend(
                        [
                            f"(MM-PLNPremise {subject})",
                            f"(MM-PLNDomain {subject} {meta['domain']})",
                            f"(MM-PLNTrust {subject} {meta['trust']})",
                            f"(MM-PLNPromotionRule {subject} {meta['rule']})",
                        ]
                    )
        if limit_chars is not None:
            return _join_bounded_atom_lines(safe_atoms, limit_chars)
        return "\n".join(safe_atoms) + ("\n" if safe_atoms else "")

    def _pln_excluded_subject_ids(self, excluded: set[str]) -> set[str]:
        out: set[str] = set()
        for cluster in self.clusters():
            for atom in cluster.atoms:
                pred = _predicate(atom)
                if pred in excluded:
                    out.add(_first_arg(atom))
                elif pred == "EpistemicRole" and "quoted-utterance" in atom:
                    out.add(_first_arg(atom))
        return out

    def _promoted_belief_metadata(self) -> dict[str, dict[str, str]]:
        """Return PLN-eligible promoted belief ids and their promotion metadata.

        A belief is PLN-eligible only when an explicit PromotionEvent names the
        promoted belief, promotion rule, bounded trust value, and intended PLN
        domain. The belief must also carry a TruthValue and EvidenceFor atom.
        """

        candidates: dict[str, dict[str, str]] = {}
        truth_subjects: set[str] = set()
        evidence_subjects: set[str] = set()
        for cluster in self.clusters():
            for atom in cluster.atoms:
                pred = _predicate(atom)
                subject = _first_arg(atom)
                if pred == "PromotionEvent":
                    targets = _second_objects_for_subject(cluster.atoms, "PromotesTo", subject)
                    rules = _second_objects_for_subject(cluster.atoms, "PromotionRule", subject)
                    trusts = _second_objects_for_subject(cluster.atoms, "PromotionTrust", subject)
                    domains = _second_objects_for_subject(cluster.atoms, "PromotionDomain", subject)
                    if targets and rules and trusts and domains and _is_bounded_probability(trusts[-1]):
                        candidates[targets[-1]] = {
                            "event": subject,
                            "rule": rules[-1],
                            "trust": trusts[-1],
                            "domain": domains[-1],
                        }
                elif pred == "TruthValue":
                    truth_subjects.add(subject)
                elif pred == "EvidenceFor":
                    evidence_subjects.add(subject)
        return {
            belief_id: meta
            for belief_id, meta in candidates.items()
            if belief_id in truth_subjects and belief_id in evidence_subjects
        }

    def _promoted_belief_ids(self) -> set[str]:
        return set(self._promoted_belief_metadata())

    def _bounded(self, clusters: list[MemoryCluster], limit: int) -> list[MemoryCluster]:
        if limit < 0:
            raise ValidationError("limit must be non-negative")
        return clusters[:limit]


def _parse_atom_texts(text: str) -> Iterable[str]:
    """Parse and canonicalize top-level MeTTa/S-expression atoms.

    Comments are ignored outside strings, quoted strings may contain whitespace
    and parentheses, nested expressions are parsed recursively, and every
    top-level form must be a non-empty list whose head is a symbol.
    """
    try:
        for expr in parse_top_level_lists(text):
            if not _is_symbol(expr[0]):
                raise ValidationError("top-level atom predicate must be a symbol")
            yield to_source(expr)
    except SExpressionSyntaxError as exc:
        raise ValidationError(f"malformed cluster syntax: {exc}") from exc


def _parse_single_atom(text: str) -> tuple[SExpr, ...]:
    try:
        expr = parse_one_list(text)
    except SExpressionSyntaxError as exc:
        raise ValidationError(str(exc)) from exc
    if not expr:
        raise ValidationError("expected non-empty list atom")
    return expr


def _render_sexpr(expr: SExpr) -> str:
    return to_source(expr)


def _is_symbol(expr: SExpr) -> bool:
    return symbol_text(expr) is not None


def _predicate(atom: str) -> str:
    parsed = _parse_single_atom(atom)
    head = parsed[0]
    if not _is_symbol(head):
        raise ValidationError(f"malformed atom: {atom}")
    return str(head)


def _first_arg(atom: str) -> str:
    parsed = _parse_single_atom(atom)
    return _render_sexpr(parsed[1]) if len(parsed) > 1 else ""


def _has_predicate(atoms: Iterable[str], predicate: str) -> bool:
    return any(_predicate(atom) == predicate for atom in atoms)


def _objects_for_predicate(atoms: Iterable[str], predicate: str) -> list[str]:
    out: list[str] = []
    for atom in atoms:
        parsed = _parse_single_atom(atom)
        if parsed and parsed[0] == predicate:
            out.append(_render_sexpr(parsed[1]) if len(parsed) > 1 else "")
    return out


def _declared_ids(atoms: Iterable[str]) -> list[str]:
    ids: list[str] = []
    for atom in atoms:
        if _predicate(atom) in _ID_DECLARING_PREDICATES:
            ident = _first_arg(atom)
            if ident:
                ids.append(ident)
    return ids


def _second_objects_for_predicate(atoms: Iterable[str], predicate: str) -> list[str]:
    out: list[str] = []
    for atom in atoms:
        parsed = _parse_single_atom(atom)
        if parsed and parsed[0] == predicate and len(parsed) >= 3:
            out.append(_render_sexpr(parsed[2]))
    return out


def _second_objects_for_subject(atoms: Iterable[str], predicate: str, subject: str) -> list[str]:
    out: list[str] = []
    for atom in atoms:
        parsed = _parse_single_atom(atom)
        if parsed and parsed[0] == predicate and len(parsed) >= 3 and _render_sexpr(parsed[1]) == subject:
            out.append(_render_sexpr(parsed[2]))
    return out


def _cluster_mentions_id(cluster: MemoryCluster, identifier: str) -> bool:
    return any(identifier in _atom_symbol_tokens(atom) for atom in cluster.atoms)


def _atom_mentions_any(atom: str, identifiers: set[str]) -> bool:
    if not identifiers:
        return False
    tokens = _atom_symbol_tokens(atom)
    return any(identifier in tokens for identifier in identifiers)


def _prompt_cluster_score(cluster: MemoryCluster, *, topics: set[str], statuses: set[str]) -> int:
    score = 0
    subjects_with_requested_topic: set[str] = set()
    for atom in cluster.atoms:
        pred = _predicate(atom)
        if pred == "About" and _second_arg(atom) in topics:
            subjects_with_requested_topic.add(_first_arg(atom))
            score += 100
        elif pred in {"ClusterStatus", "HasStatus", "StatusValue"} and _second_arg(atom) in statuses:
            score += 50
        elif pred == "SalienceValue":
            score += _salience_points(_second_arg(atom))
    if subjects_with_requested_topic:
        for atom in cluster.atoms:
            if _predicate(atom) in {"CommitmentText", "QuestionText", "ArtifactPath", "BoundaryName"}:
                if _first_arg(atom) in subjects_with_requested_topic:
                    score += 10
    return score


def _second_arg(atom: str) -> str:
    parsed = _parse_single_atom(atom)
    return _render_sexpr(parsed[2]) if len(parsed) >= 3 else ""


def _index_atoms(clusters: list[MemoryCluster], *, superseded_event_ids: set[str]) -> list[str]:
    atoms: list[str] = []
    seen: set[str] = set()

    def add(atom: str) -> None:
        if atom not in seen:
            seen.add(atom)
            atoms.append(atom)

    for cluster in clusters:
        add(f"(MM-index {cluster.cluster_id})")
        add(f"(MM-index-id {cluster.cluster_id} {cluster.cluster_id})")
        for atom in cluster.atoms:
            parsed = _parse_single_atom(atom)
            pred = str(parsed[0])
            for ident in _argument_identifier_tokens(parsed):
                add(f"(MM-index-id {ident} {cluster.cluster_id})")
            if pred in _ID_DECLARING_PREDICATES and len(parsed) >= 2:
                ident = _render_sexpr(parsed[1])
                add(f"(MM-index-type {pred} {ident} {cluster.cluster_id})")
            elif pred == "About" and len(parsed) >= 3:
                subject = _render_sexpr(parsed[1])
                entity = _render_sexpr(parsed[2])
                add(f"(MM-index-about {entity} {subject} {cluster.cluster_id})")
            elif pred in {"ClusterStatus", "HasStatus"} and len(parsed) >= 3:
                subject = _render_sexpr(parsed[1])
                status = _render_sexpr(parsed[2])
                add(f"(MM-index-status {status} {subject} {cluster.cluster_id})")
            elif pred == "StatusValue" and len(parsed) >= 3:
                event_id = _render_sexpr(parsed[1])
                if event_id not in superseded_event_ids:
                    status = _render_sexpr(parsed[2])
                    subjects = _second_objects_for_subject(cluster.atoms, "StatusSubject", event_id)
                    subject = subjects[-1] if subjects else event_id
                    add(f"(MM-index-status {status} {subject} {cluster.cluster_id})")
            elif pred == "EpistemicRole" and len(parsed) >= 3:
                subject = _render_sexpr(parsed[1])
                role = _render_sexpr(parsed[2])
                add(f"(MM-index-role {role} {subject} {cluster.cluster_id})")
    return atoms


def _join_bounded_atom_lines(atoms: Iterable[str], limit_chars: int) -> str:
    """Join complete atom lines without exceeding a character budget.

    Bounded prompt/index views should remain parseable MeTTa snippets. If the
    next atom would cross the budget, omit it rather than returning a partial
    atom that a caller might misparse as corrupted memory.
    """
    if limit_chars <= 0:
        return ""
    out: list[str] = []
    used = 0
    for atom in atoms:
        addition = len(atom) + 1  # every emitted atom is newline-terminated
        if used + addition > limit_chars:
            break
        out.append(atom)
        used += addition
    return "\n".join(out) + ("\n" if out else "")


def _atom_symbol_tokens(atom: str) -> set[str]:
    tokens: set[str] = set()

    def visit(expr: SExpr) -> None:
        if isinstance(expr, tuple):
            for item in expr:
                visit(item)
        elif isinstance(expr, str):
            tokens.add(expr)

    visit(_parse_single_atom(atom))
    return tokens


def _argument_identifier_tokens(parsed_atom: tuple[SExpr, ...]) -> set[str]:
    tokens: set[str] = set()

    def visit(expr: SExpr) -> None:
        if isinstance(expr, tuple):
            for item in expr:
                visit(item)
        elif isinstance(expr, str) and _ID_RE.match(expr):
            tokens.add(expr)

    for arg in parsed_atom[1:]:
        visit(arg)
    return tokens


def _salience_points(value: str) -> int:
    if not value:
        return 0
    named = {"critical": 40, "high": 30, "medium": 20, "low": 10}
    if value in named:
        return named[value]
    try:
        numeric = float(value)
    except ValueError:
        return 0
    return max(0, min(40, int(numeric * 40)))


def _split_cluster_chunks(text: str) -> list[str]:
    records: list[str] = []
    current: list[str] = []
    current_id: Optional[str] = None
    saw_delimiter = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(_BEGIN_PREFIX):
            if current:
                raise ValidationError("nested or unterminated cluster record")
            saw_delimiter = True
            current_id = line[len(_BEGIN_PREFIX) :].strip()
            current = []
            continue
        if line.startswith(_END_PREFIX):
            end_id = line[len(_END_PREFIX) :].strip()
            if current_id is None or end_id != current_id:
                raise ValidationError("cluster delimiter id mismatch")
            chunk = "\n".join(current)
            declared_ids = _declared_cluster_ids_in_chunk(chunk)
            if len(declared_ids) == 1 and declared_ids[0] != current_id:
                raise ValidationError("cluster delimiter id mismatch")
            records.append(chunk)
            current = []
            current_id = None
            continue
        if saw_delimiter:
            if current_id is None:
                raise ValidationError("content outside cluster record")
            current.append(line)
        else:
            if line.startswith("(MemoryCluster ") and current:
                records.append("\n".join(current))
                current = [line]
            elif not line.startswith(";"):
                current.append(line)
    if current_id is not None:
        raise ValidationError("unterminated cluster record")
    if not saw_delimiter and current:
        records.append("\n".join(current))
    return records


def _declared_cluster_ids_in_chunk(text: str) -> list[str]:
    """Return MemoryCluster ids declared in a raw delimited record body.

    Delimited journals carry the cluster id both in the BEGIN/END markers and in
    the MemoryCluster atom.  Checking the redundant id during read prevents audit
    and query views from silently normalizing a manually corrupted record whose
    envelope points at a different cluster.
    """
    return _objects_for_predicate(tuple(_parse_atom_texts(text)), "MemoryCluster")


def _pettachainer_stv(truth_value: str, *, promotion_trust: str) -> Optional[str]:
    parsed = _parse_single_atom(truth_value)
    if len(parsed) != 3 or str(parsed[0]).lower() != "stv":
        return None
    strength = _render_sexpr(parsed[1])
    confidence = _render_sexpr(parsed[2])
    if not (_is_bounded_probability(strength) and _is_bounded_probability(confidence)):
        return None
    confidence_number = float(confidence)
    capped_confidence = min(confidence_number, float(promotion_trust))
    rendered_confidence = confidence if capped_confidence == confidence_number else _format_probability(capped_confidence)
    return f"(STV {strength} {rendered_confidence})"


def _format_probability(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _is_bounded_probability(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return 0.0 <= number <= 1.0


def _is_non_negative_number(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return math.isfinite(number) and number >= 0.0
