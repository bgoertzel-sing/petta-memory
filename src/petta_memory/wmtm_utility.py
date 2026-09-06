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
        """Build a valid s-expression cluster for a derived WMTM item."""
        import time
        import uuid

        short_uuid = uuid.uuid4().hex[:8]
        cluster_id = f"wmtm-{short_uuid}"
        episode_id = f"wmtm-ep-{short_uuid}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        safe_text = item.text.replace('"', '\\"').replace('\n', ' ')

        source_desc = "wmtm-derived"
        if item.derived_from:
            source_desc = f"wmtm-derived-from:{','.join(item.derived_from[:3])}"

        cluster_text = (
            f"(MemoryCluster {cluster_id})\n"
            f"(SchemaVersion {cluster_id} medium-memory-v1)\n"
            f"(ClusterType {cluster_id} episode-record)\n"
            f"(ClusterOpenedAt {cluster_id} \"{timestamp}\")\n"
            f"(ClusterSource {cluster_id} {source_desc})\n"
            f"(Contains {cluster_id} {episode_id})\n"
            f"(ClusterStatus {cluster_id} active)\n"
            f"(ObservedEvent {episode_id})\n"
            f"(EpistemicRole {episode_id} observed-event)\n"
            f"(About {episode_id} \"{safe_text}\")\n"
            f"(HasStatus {episode_id} recorded)\n"
        )
        return cluster_text

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
