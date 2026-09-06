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

    def writeback(self, item_ids: Optional[list[str]] = None) -> list[str]:
        """Write high-utility items back to LTM via petta_append.

        For recalled items: boost STI via ECAN stimulate.
        For derived items: append as new cluster via store.append_cluster.

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
                # Try to persist derived item to store
                try:
                    note = f"[WMTM-derived] {item.text}"
                    # Use store's append method if available
                    if hasattr(self.store, "append_cluster"):
                        self.store.append_cluster(note)
                    elif hasattr(self.store, "append"):
                        self.store.append(note)
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
