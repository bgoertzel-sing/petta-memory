"""
WMTM Utility — utility scoring and writeback to LTM.

Scores items by usefulness, and writes high-utility items back to
MediumMemoryStore for persistence in long-term memory.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from petta_memory.wmtm import WMTMStore, WMTMItem
    from petta_memory.store import MediumMemoryStore
    from petta_memory.ecan_bridge import ECANBridge

log = logging.getLogger(__name__)


class WMTMUtility:
    """Computes utility scores and manages writeback to LTM."""

    def __init__(
        self,
        wmtm: "WMTMStore",
        store: "MediumMemoryStore",
        ecan_bridge: "ECANBridge",
        writeback_threshold: float = 5.0,
        sti_writeback_boost: float = 3.0,
    ):
        self.wmtm = wmtm
        self.store = store
        self.ecan = ecan_bridge
        self.writeback_threshold = writeback_threshold
        self.sti_writeback_boost = sti_writeback_boost

    def compute_utility(self, item: "WMTMItem") -> float:
        """Compute utility score for an item.

        utility = use_count * recency_factor + sti * 0.3 - age_penalty
        """
        recency = 0.9 ** max(0, item.age - item.last_used)
        age_penalty = 0.1 * item.age
        return item.use_count * recency + item.sti * 0.3 - age_penalty

    def score_all(self) -> dict[str, float]:
        """Compute utility scores for all items."""
        return {item.id: self.compute_utility(item) for item in self.wmtm.all_items()}

    def get_writeback_candidates(self) -> list["WMTMItem"]:
        """Get items that should be written back to LTM."""
        scored = [(self.compute_utility(i), i) for i in self.wmtm.all_items()]
        scored.sort(key=lambda x: -x[0])
        return [item for score, item in scored if score >= self.writeback_threshold]

    def _build_cluster_text(self, item) -> str:
        """Build a valid s-expression cluster for a derived WMTM item.

        F02 FIX: Uses DerivedBelief (not ObservedEvent) for derived items.
        F08 FIX: About text is NOT quoted so query_about regex can match.
        """
        import time
        import uuid

        short_uuid = uuid.uuid4().hex[:8]
        cluster_id = f"wmtm-{short_uuid}"
        belief_id = f"wmtm-bl-{short_uuid}"
        event_id = f"wmtm-ev-{short_uuid}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # F08 FIX: Use unquoted text for About so query_about can match it.
        # If text contains spaces, use it as-is (the store regex matches unquoted tokens).
        safe_text = item.text.replace('"', '').replace('\n', ' ').strip()
        # Truncate to reasonable length
        safe_text = safe_text[:200] if len(safe_text) > 200 else safe_text

        source_desc = "wmtm-derived"
        if item.derived_from:
            source_desc = f"wmtm-derived-from:{','.join(item.derived_from[:3])}"

        atoms = [
            f"(MemoryCluster {cluster_id})",
            f"(SchemaVersion {cluster_id} medium-memory-v1)",
            f"(ClusterType {cluster_id} belief-record)",
            f"(ClusterOpenedAt {cluster_id} \"{timestamp}\")",
            f"(ClusterSource {cluster_id} {source_desc})",
            f"(Contains {cluster_id} {belief_id})",
            f"(Contains {cluster_id} {event_id})",
            # F02 FIX: Use DerivedBelief for derived items, not ObservedEvent
            f"(DerivedBelief {belief_id})",
            f"(BeliefContent {belief_id} ({safe_text}))",
            f'(About {belief_id} "{safe_text}")',
            f"(ObservedEvent {event_id})",
            f"(EvidenceFor {belief_id} {event_id})",
        ]
        return "\n".join(atoms) + "\n"

    def writeback(self, item_ids: Optional[list[str]] = None) -> list[str]:
        """Write high-utility items back to LTM.

        For recalled items: boost STI via ECAN stimulate.
        For derived items: append as new cluster via store.append_cluster
        using a properly formatted s-expression cluster.

        Returns list of successfully written item IDs.
        """
        if item_ids is None:
            candidates = self.get_writeback_candidates()
        else:
            candidates = [self.wmtm.get(i) for i in item_ids if i in self.wmtm]
            candidates = [c for c in candidates if c is not None]

        written = []
        stimuli = {}

        for item in candidates:
            if item.source_type == "recalled" and item.origin_cluster:
                # Boost STI in ECAN for recalled items
                stimuli[item.origin_cluster] = self.sti_writeback_boost
                written.append(item.id)
            elif item.source_type == "derived":
                # Persist derived item to store as a valid s-expression cluster
                try:
                    cluster_text = self._build_cluster_text(item)
                    if hasattr(self.store, "append_cluster"):
                        self.store.append_cluster(cluster_text)
                    elif hasattr(self.store, "append"):
                        self.store.append(cluster_text)
                    written.append(item.id)
                except Exception as e:
                    log.warning("Writeback failed for %s: %s", item.id, e)

        # Batch stimulate ECAN
        if stimuli:
            try:
                self.ecan.stimulate_beliefs(stimuli)
            except Exception as e:
                log.warning("ECAN stimulation failed: %s", e)

        log.debug("Utility: wrote back %d items", len(written))
        return written


    def summary(self) -> dict:
        scores = self.score_all()
        values = list(scores.values())
        return {
            "total_items": len(scores),
            "avg_utility": sum(values) / max(1, len(values)),
            "max_utility": max(values) if values else 0.0,
            "writeback_candidates": sum(1 for v in values if v >= self.writeback_threshold),
        }
