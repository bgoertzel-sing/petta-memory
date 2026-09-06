"""
WMTM Coordinator — orchestrates the full working memory cycle.

Ties together RecallBridge, WMTMInferenceEngine, WMTMUtility, and WMTMStore
into a single callable cycle: recall → spread → tick → derive → utility → writeback.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from petta_memory.store import MediumMemoryStore
    from petta_memory.ecan_bridge import ECANBridge

from petta_memory.wmtm import WMTMStore, ForgettingPolicy
from petta_memory.wmtm_recall import RecallBridge
from petta_memory.wmtm_inference import WMTMInferenceEngine
from petta_memory.wmtm_utility import WMTMUtility

log = logging.getLogger(__name__)


class WMTMCoordinator:
    """Top-level coordinator for the Working Memory subsystem.

    Usage:
        coord = WMTMCoordinator(store, ecan_bridge)
        coord.on_context("user asked about ECAN tuning")
        # ... agent processes ...
        coord.on_tick()
        # ... derive new insights ...
        coord.on_derive("conclusion text", ["source1", "source2"])
        # ... at end of turn ...
        coord.on_turn_end()
    """

    def __init__(
        self,
        store: "MediumMemoryStore",
        ecan_bridge: "ECANBridge",
        capacity: int = 60,
        decay_rate: float = 0.85,
        recall_top_k: int = 15,
        spread_depth: int = 2,
        writeback_threshold: float = 5.0,
        auto_writeback: bool = True,
    ):
        self.wmtm = WMTMStore(
            capacity=capacity,
            decay_rate=decay_rate,
            forgetting=ForgettingPolicy(max_items=capacity),
        )
        self.recall = RecallBridge(store, ecan_bridge, self.wmtm)
        self.inference = WMTMInferenceEngine(self.wmtm)
        self.utility = WMTMUtility(
            self.wmtm, store, ecan_bridge,
            writeback_threshold=writeback_threshold,
        )

        self.recall_top_k = recall_top_k
        self.spread_depth = spread_depth
        self.auto_writeback = auto_writeback
        self._turn_count = 0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def on_context(self, context: str) -> list[str]:
        """Called when new context/query arrives. Recalls + spreads activation."""
        recalled = self.recall.recall(context, top_k=self.recall_top_k)
        if recalled:
            self.recall.spreading_activation(
                recalled, depth=self.spread_depth
            )
        log.info("WMTM on_context: %d recalled + spread items", len(recalled))
        return recalled

    def on_tick(self) -> list[str]:
        """Called periodically to maintain working memory (decay + forget)."""
        return self.wmtm.tick()

    def on_derive(
        self,
        text: str,
        source_ids: list[str],
        sti: Optional[float] = None,
    ) -> Optional[str]:
        """Called when the agent derives new knowledge."""
        item = self.inference.derive(text, source_ids, sti=sti)
        if item:
            log.info("WMTM on_derive: '%s' from %d sources", text[:50], len(source_ids))
            return item.id
        return None

    def on_touch(self, item_id: str) -> None:
        """Mark an item as used (boosts its utility)."""
        self.wmtm.touch(item_id)

    def on_turn_end(self) -> list[str]:
        """Called at end of turn: final tick + writeback to LTM."""
        self.wmtm.tick()
        self._turn_count += 1
        written = []
        if self.auto_writeback:
            written = self.utility.writeback()
        log.info(
            "WMTM on_turn_end (turn %d): %d items written back, %d remaining",
            self._turn_count, len(written), len(self.wmtm),
        )
        return written

    def get_active_context(self, limit: int = 20) -> list[dict]:
        """Get the current active working memory items as context dicts."""
        items = self.wmtm.get_active_set(limit=limit)
        return [
            {
                "id": i.id,
                "text": i.text,
                "sti": round(i.sti, 2),
                "source_type": i.source_type,
                "use_count": i.use_count,
                "derived_from": i.derived_from if i.is_derived else None,
            }
            for i in items
        ]

    def full_summary(self) -> dict:
        """Comprehensive summary of all WMTM subsystems."""
        return {
            "turn": self._turn_count,
            "wmtm": self.wmtm.summary(),
            "inference": self.inference.summary(),
            "utility": self.utility.summary(),
        }
