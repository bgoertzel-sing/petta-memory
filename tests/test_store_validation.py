"""Direct tests for MediumMemoryStore.validate_atom and validate_cluster.

These two public methods were not directly exercised by the existing
test suite (only indirectly via append_cluster / round-trip tests).
"""
import pytest
from pathlib import Path
import tempfile
from petta_memory.store import MediumMemoryStore, ValidationError, MemoryCluster, SCHEMA_VERSION


@pytest.fixture
def store():
    td = Path(tempfile.mkdtemp())
    return MediumMemoryStore(td / "journal.metta")


def _valid_cluster_text():
    return (
        f"(MemoryCluster tc-001)\n"
        f"(SchemaVersion tc-001 {SCHEMA_VERSION})\n"
        f"(ClusterType tc-001 Observation)\n"
        f"(ClusterOpenedAt tc-001 2026-09-04T16:00:00Z)\n"
        f"(ClusterSource tc-001 test)\n"
        f"(Contains tc-001 e1)\n"
        f"(ClusterStatus tc-001 active)\n"
        f"(ObservedEvent e1)\n"
        f"(EpistemicRole e1 observed-event)\n"
        f"(About e1 test-topic)\n"
        f"(HasStatus e1 recorded)"
    )


class TestValidateAtom:
    def test_valid_atom_passes(self, store):
        store.validate_atom("(HasStatus e1 recorded)")

    def test_malformed_atom_raises(self, store):
        with pytest.raises(ValidationError, match="malformed"):
            store.validate_atom("not an atom")

    def test_oversize_atom_raises(self, store):
        big = "(TooLong " + "x" * (store.max_atom_chars + 1) + ")"
        with pytest.raises(ValidationError, match="max_atom_chars"):
            store.validate_atom(big)


class TestValidateCluster:
    def test_valid_cluster_returns_memorycluster(self, store):
        result = store.validate_cluster(_valid_cluster_text())
        assert isinstance(result, MemoryCluster)
        assert result.cluster_id == "tc-001"

    def test_empty_cluster_raises(self, store):
        with pytest.raises(ValidationError, match="empty"):
            store.validate_cluster("")

    def test_no_memorycluster_atom_raises(self, store):
        text = (
            f"(SchemaVersion tc-001 {SCHEMA_VERSION})\n"
            f"(ClusterType tc-001 Observation)\n"
            f"(ClusterOpenedAt tc-001 2026-09-04T16:00:00Z)\n"
            f"(ClusterSource tc-001 test)\n"
            f"(Contains tc-001 e1)\n"
            f"(ObservedEvent e1)"
        )
        with pytest.raises(ValidationError, match="exactly one"):
            store.validate_cluster(text)

    def test_oversize_cluster_raises(self, store):
        big_line = "(Big " + "x" * (store.max_cluster_chars + 1) + ")"
        with pytest.raises(ValidationError, match="max_cluster_chars"):
            store.validate_cluster(big_line)

    def test_wrong_schema_version_raises(self, store):
        text = (
            f"(MemoryCluster tc-001)\n"
            f"(SchemaVersion tc-001 wrong-version)\n"
            f"(ClusterType tc-001 Observation)\n"
            f"(ClusterOpenedAt tc-001 2026-09-04T16:00:00Z)\n"
            f"(ClusterSource tc-001 test)\n"
            f"(Contains tc-001 e1)\n"
            f"(ObservedEvent e1)"
        )
        with pytest.raises(ValidationError, match="SchemaVersion"):
            store.validate_cluster(text)

    def test_missing_provenance_raises(self, store):
        text = (
            f"(MemoryCluster tc-001)\n"
            f"(SchemaVersion tc-001 {SCHEMA_VERSION})\n"
            f"(ClusterType tc-001 Observation)\n"
            f"(ClusterOpenedAt tc-001 2026-09-04T16:00:00Z)\n"
            f"(Contains tc-001 e1)\n"
            f"(ObservedEvent e1)"
        )
        with pytest.raises(ValidationError, match="ClusterSource or HasProvenance"):
            store.validate_cluster(text)

    def test_missing_required_predicate_raises(self, store):
        """Remove ClusterType to trigger required-predicate check."""
        text = (
            f"(MemoryCluster tc-001)\n"
            f"(SchemaVersion tc-001 {SCHEMA_VERSION})\n"
            f"(ClusterOpenedAt tc-001 2026-09-04T16:00:00Z)\n"
            f"(ClusterSource tc-001 test)\n"
            f"(Contains tc-001 e1)\n"
            f"(ObservedEvent e1)"
        )
        with pytest.raises(ValidationError, match="ClusterType"):
            store.validate_cluster(text)

    def test_undeclared_contains_target_raises(self, store):
        """Contains target must be a locally declared id."""
        text = (
            f"(MemoryCluster tc-001)\n"
            f"(SchemaVersion tc-001 {SCHEMA_VERSION})\n"
            f"(ClusterType tc-001 Observation)\n"
            f"(ClusterOpenedAt tc-001 2026-09-04T16:00:00Z)\n"
            f"(ClusterSource tc-001 test)\n"
            f"(Contains tc-001 e1)\n"
            f"(ObservedEvent e1)\n"
            f"(Contains tc-001 missing-id)"
        )
        with pytest.raises(ValidationError, match="not declared in cluster"):
            store.validate_cluster(text)

    def test_duplicate_declared_ids_raises(self, store):
        """Same id declared twice via different predicates."""
        text = (
            f"(MemoryCluster tc-001)\n"
            f"(SchemaVersion tc-001 {SCHEMA_VERSION})\n"
            f"(ClusterType tc-001 Observation)\n"
            f"(ClusterOpenedAt tc-001 2026-09-04T16:00:00Z)\n"
            f"(ClusterSource tc-001 test)\n"
            f"(Contains tc-001 e1)\n"
            f"(ObservedEvent e1)\n"
            f"(ObservedEvent e1)"
        )
        with pytest.raises(ValidationError, match="duplicate declared ids"):
            store.validate_cluster(text)
