"""
Object identity index for the cognitive-memory system.

Maps declared cognitive objects (beliefs, events, hypotheses, etc.) to
their containing clusters.  Provides typed lookup independent of
query_cluster(), so callers can resolve beliefs, events, and evidence
by object ID rather than by cluster ID.

Part of C02: Establish object identity and repair recall.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .store import _objects_for_predicate, MemoryCluster


_OBJECT_PREDICATES = frozenset({
    "DerivedBelief", "ObservedEvent", "SpeechEvent", "QuotedClaim",
    "Decision", "Hypothesis", "OpenQuestion", "Commitment",
    "Boundary", "Artifact", "StatusEvent", "SalienceEvent",
    "PromotionEvent", "TruthValueEvent",
})


class ObjectIndex:
    """Index from cognitive object IDs to their containing clusters."""

    def __init__(self):
        self._by_object: dict[str, tuple[str, str]] = {}
        self._by_cluster: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def index_cluster(self, cluster: MemoryCluster) -> None:
        cid = cluster.cluster_id
        atoms = cluster.atoms
        for pred in _OBJECT_PREDICATES:
            for obj_id in _objects_for_predicate(atoms, pred):
                self._by_object[obj_id] = (cid, pred)
                self._by_cluster[cid].append((obj_id, pred))

    def build_from_store(self, store) -> None:
        for cluster in store.clusters():
            self.index_cluster(cluster)

    def lookup(self, object_id: str) -> Optional[tuple[str, str]]:
        return self._by_object.get(object_id)

    def objects_in_cluster(self, cluster_id: str) -> list[tuple[str, str]]:
        return list(self._by_cluster.get(cluster_id, []))

    def cluster_of(self, object_id: str) -> Optional[str]:
        result = self._by_object.get(object_id)
        return result[0] if result else None

    def object_type(self, object_id: str) -> Optional[str]:
        result = self._by_object.get(object_id)
        return result[1] if result else None

    def all_objects(self) -> dict[str, tuple[str, str]]:
        return dict(self._by_object)

    def __len__(self) -> int:
        return len(self._by_object)
