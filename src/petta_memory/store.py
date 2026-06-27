from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
from typing import Iterable, Optional


class ValidationError(ValueError):
    """Raised when a memory atom or cluster is invalid."""


SCHEMA_VERSION = "medium-memory-v1"
_BEGIN_PREFIX = ";;; BEGIN MemoryCluster "
_END_PREFIX = ";;; END MemoryCluster "
_ATOM_RE = re.compile(r"^\(([^\s()]+)(?:\s+.*)?\)$")
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
    "QuotedClaim",
    "ClaimStatus",
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
        return cluster

    def validate_cluster(self, text: str) -> MemoryCluster:
        if len(text.encode("utf-8")) > self.max_cluster_chars:
            raise ValidationError("cluster exceeds max_cluster_chars")
        atoms = tuple(_clean_atom_lines(text))
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
        return MemoryCluster(cluster_id=cluster_id, atoms=atoms)

    def validate_atom(self, atom: str) -> None:
        if len(atom.encode("utf-8")) > self.max_atom_chars:
            raise ValidationError("atom exceeds max_atom_chars")
        if not atom.startswith("(") or not atom.endswith(")"):
            raise ValidationError(f"not an atom: {atom}")
        if not _balanced_parentheses(atom):
            raise ValidationError(f"unbalanced parentheses: {atom}")
        if not _ATOM_RE.match(atom):
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
        for cluster in self.clusters():
            subjects = _objects_for_predicate(cluster.atoms, "StatusSubject")
            if subjects:
                if any(self.current_status(subject) == status for subject in subjects):
                    matches.append(cluster)
            elif re.search(rf"\((?:ClusterStatus|HasStatus)\s+[^\s()]+\s+{re.escape(status)}\)", cluster.text):
                matches.append(cluster)
        return self._bounded(matches, limit)

    def query_role(self, role: str, *, limit: int = 20) -> list[MemoryCluster]:
        pat = re.compile(rf"^\(EpistemicRole\s+[^\s()]+\s+{re.escape(role)}\)", re.MULTILINE)
        return self._bounded([c for c in self.clusters() if pat.search(c.text)], limit)

    def current_status(self, subject_id: str) -> Optional[str]:
        current: Optional[str] = None
        superseded_events: set[str] = set()
        events: list[tuple[str, str]] = []
        for cluster in self.clusters():
            atoms = cluster.atoms
            superseded_events.update(_second_objects_for_predicate(atoms, "Supersedes"))
            for event_id in _objects_for_predicate(atoms, "StatusEvent"):
                if _second_objects_for_subject(atoms, "StatusSubject", event_id) == [subject_id]:
                    values = _second_objects_for_subject(atoms, "StatusValue", event_id)
                    if values:
                        events.append((event_id, values[-1]))
        for event_id, value in events:
            if event_id not in superseded_events:
                current = value
        return current

    def prompt_view(self, *, limit_chars: int = 4000) -> str:
        pieces: list[str] = []
        allowed = {"CommitmentText", "QuestionText", "BoundaryName", "ArtifactPath", "HasStatus", "ClusterStatus", "StatusValue", "About"}
        for cluster in self.clusters():
            for atom in cluster.atoms:
                if _predicate(atom) in allowed:
                    pieces.append(atom)
        text = "\n".join(pieces)
        return text[:limit_chars]

    def pln_view(self, *, excluded_predicates: Optional[set[str]] = None) -> str:
        excluded = excluded_predicates or _DEFAULT_PLN_EXCLUDED
        safe_atoms: list[str] = []
        promoted_beliefs = self._promoted_belief_ids()
        for cluster in self.clusters():
            for atom in cluster.atoms:
                pred = _predicate(atom)
                if pred in excluded:
                    continue
                if pred == "EpistemicRole" and "quoted-utterance" in atom:
                    continue
                if pred in {"DerivedBelief", "BeliefContent"}:
                    subject = _first_arg(atom)
                    if subject not in promoted_beliefs:
                        continue
                safe_atoms.append(atom)
        return "\n".join(safe_atoms) + ("\n" if safe_atoms else "")

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


def _clean_atom_lines(text: str) -> Iterable[str]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        yield line


def _balanced_parentheses(text: str) -> bool:
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_string


def _predicate(atom: str) -> str:
    m = _ATOM_RE.match(atom)
    if not m:
        raise ValidationError(f"malformed atom: {atom}")
    return m.group(1)


def _first_arg(atom: str) -> str:
    rest = atom[len(_predicate(atom)) + 2 : -1].strip()
    return rest.split(maxsplit=1)[0] if rest else ""


def _has_predicate(atoms: Iterable[str], predicate: str) -> bool:
    return any(_predicate(atom) == predicate for atom in atoms)


def _objects_for_predicate(atoms: Iterable[str], predicate: str) -> list[str]:
    out: list[str] = []
    prefix = f"({predicate} "
    for atom in atoms:
        if atom.startswith(prefix):
            rest = atom[len(prefix) : -1].strip()
            first = rest.split(maxsplit=1)[0] if rest else ""
            out.append(first)
    return out


def _second_objects_for_predicate(atoms: Iterable[str], predicate: str) -> list[str]:
    out: list[str] = []
    prefix = f"({predicate} "
    for atom in atoms:
        if atom.startswith(prefix):
            parts = atom[len(prefix) : -1].strip().split(maxsplit=2)
            if len(parts) >= 2:
                out.append(parts[1])
    return out


def _second_objects_for_subject(atoms: Iterable[str], predicate: str, subject: str) -> list[str]:
    out: list[str] = []
    prefix = f"({predicate} {subject} "
    for atom in atoms:
        if atom.startswith(prefix):
            rest = atom[len(prefix) : -1].strip()
            out.append(rest.split(maxsplit=1)[0] if rest else "")
    return out


def _cluster_mentions_id(cluster: MemoryCluster, identifier: str) -> bool:
    return any(identifier in atom.split() for atom in cluster.atoms)


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
