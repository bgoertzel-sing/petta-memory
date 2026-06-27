from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Optional


class ValidationError(ValueError):
    """Raised when a memory atom or cluster is invalid."""


_ATOM_RE = re.compile(r"^\(([^\s()]+)(?:\s+.*)?\)$")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_REQUIRED_CLUSTER_PREDICATES = ("MemoryCluster", "ClusterType", "ClusterOpenedAt", "Contains")
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


class MediumMemoryStore:
    """Append-only `.metta` cluster journal with bounded deterministic queries.

    This is intentionally conservative: it treats the journal as auditable text,
    validates cluster envelopes, and leaves full AtomSpace/PLN loading to later
    integration code.
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
        with self.path.open("a", encoding="utf-8") as f:
            if self.path.exists() and self.path.stat().st_size > 0:
                f.write("\n")
            f.write(cluster.text)
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
        return self._bounded([c for c in self.clusters() if identifier in c.text], limit)

    def query_type(self, type_name: str, *, limit: int = 20) -> list[MemoryCluster]:
        pat = re.compile(rf"^\({re.escape(type_name)}\s+", re.MULTILINE)
        return self._bounded([c for c in self.clusters() if pat.search(c.text)], limit)

    def query_about(self, entity: str, *, limit: int = 20) -> list[MemoryCluster]:
        pat = re.compile(rf"^\(About\s+[^\s()]+\s+{re.escape(entity)}\)", re.MULTILINE)
        return self._bounded([c for c in self.clusters() if pat.search(c.text)], limit)

    def query_status(self, status: str, *, limit: int = 20) -> list[MemoryCluster]:
        pat = re.compile(rf"\((?:HasStatus|ClusterStatus|StatusValue|CommitmentStatus|QuestionStatus|HypothesisStatus)\s+[^\s()]+\s+{re.escape(status)}\)")
        return self._bounded([c for c in self.clusters() if pat.search(c.text)], limit)

    def query_role(self, role: str, *, limit: int = 20) -> list[MemoryCluster]:
        pat = re.compile(rf"^\(EpistemicRole\s+[^\s()]+\s+{re.escape(role)}\)", re.MULTILINE)
        return self._bounded([c for c in self.clusters() if pat.search(c.text)], limit)

    def prompt_view(self, *, limit_chars: int = 4000) -> str:
        pieces: list[str] = []
        for cluster in self.clusters():
            for atom in cluster.atoms:
                if _predicate(atom) in {"CommitmentText", "QuestionText", "BoundaryName", "ArtifactPath", "HasStatus", "CommitmentStatus", "QuestionStatus", "HypothesisStatus", "About"}:
                    pieces.append(atom)
        text = "\n".join(pieces)
        return text[:limit_chars]

    def pln_view(self, *, excluded_predicates: Optional[set[str]] = None) -> str:
        excluded = excluded_predicates or _DEFAULT_PLN_EXCLUDED
        safe_atoms: list[str] = []
        for cluster in self.clusters():
            for atom in cluster.atoms:
                pred = _predicate(atom)
                if pred in excluded:
                    continue
                if pred == "EpistemicRole" and "quoted-utterance" in atom:
                    continue
                safe_atoms.append(atom)
        return "\n".join(safe_atoms) + ("\n" if safe_atoms else "")

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


def _split_cluster_chunks(text: str) -> list[str]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in _clean_atom_lines(text):
        if line.startswith("(MemoryCluster ") and current:
            chunks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append(current)
    return ["\n".join(chunk) for chunk in chunks]
