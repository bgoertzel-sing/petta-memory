from __future__ import annotations

from dataclasses import dataclass
import fcntl
from pathlib import Path
import os
import re
import tempfile
from typing import Iterable, Optional

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
    ) -> None:
        self.path = Path(path)
        self.max_atom_chars = max_atom_chars
        self.max_cluster_chars = max_cluster_chars
        self.max_query_chars = max_query_chars

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
        ids = _declared_ids(atoms)
        duplicates = sorted({ident for ident in ids if ids.count(ident) > 1})
        if duplicates:
            raise ValidationError(f"duplicate declared ids in cluster: {', '.join(duplicates)}")
        return MemoryCluster(cluster_id=cluster_id, atoms=atoms)

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
        text = "\n".join(pieces)
        return text[:limit_chars]

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

    def pln_view(self, *, excluded_predicates: Optional[set[str]] = None) -> str:
        excluded = _DEFAULT_PLN_EXCLUDED | (excluded_predicates or set())
        safe_atoms: list[str] = []
        promoted_beliefs = self._promoted_belief_ids()
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
                if pred in {"DerivedBelief", "BeliefContent"}:
                    if subject not in promoted_beliefs:
                        continue
                safe_atoms.append(atom)
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

    def _promoted_belief_ids(self) -> set[str]:
        promoted: set[str] = set()
        truth_subjects: set[str] = set()
        evidence_subjects: set[str] = set()
        for cluster in self.clusters():
            for atom in cluster.atoms:
                pred = _predicate(atom)
                if pred == "PromotionEvent":
                    promoted.update(_second_objects_for_subject(cluster.atoms, "PromotesTo", _first_arg(atom)))
                elif pred == "TruthValue":
                    truth_subjects.add(_first_arg(atom))
                elif pred == "EvidenceFor":
                    evidence_subjects.add(_first_arg(atom))
        return promoted & truth_subjects & evidence_subjects

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
            records.append("\n".join(current))
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
