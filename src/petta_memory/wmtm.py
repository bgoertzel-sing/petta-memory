"""
WMTM — Working Memory with Tick-driven Maintenance.

Foundational data structures: WMTMItem, WMTMStore, ForgettingPolicy.
Design: /home/openclaw/research-agent/memory/wmtm_design_doc.txt
Integration map: /home/openclaw/research-agent/memory/wmtm_integration_map.txt
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import logging

log = logging.getLogger(__name__)


@dataclass
class WMTMItem:
    """A single item held in working memory."""

    id: str
    text: str
    source_type: str = "recalled"       # "recalled" | "derived"
    origin_cluster: Optional[str] = None
    derived_from: list[str] = field(default_factory=list)
    sti: float = 10.0
    lti: float = 0.0
    age: int = 0
    last_used: int = 0
    use_count: int = 0
    utility: float = 0.0

    @property
    def is_derived(self) -> bool:
        return self.source_type == "derived"

    def touch(self, cycle: int) -> None:
        self.last_used = cycle
        self.use_count += 1
        self.utility += 1.0

    def decay_sti(self, rate: float = 0.85) -> None:
        self.sti *= rate
        self.age += 1
        recency_factor = 0.9 ** max(0, self.age - self.last_used)
        self.utility = self.use_count * recency_factor - 0.1 * self.age


class ForgettingPolicy:
    """Decides which items should be evicted from WMTM."""

    def __init__(
        self,
        sti_threshold: float = 5.0,
        max_age: int = 50,
        max_items: int = 60,
        derived_eviction_mult: float = 1.5,
    ):
        self.sti_threshold = sti_threshold
        self.max_age = max_age
        self.max_items = max_items
        self.derived_eviction_mult = derived_eviction_mult

    def should_evict(self, item: WMTMItem) -> bool:
        if item.sti < self.sti_threshold:
            return True
        if item.age >= self.max_age:
            return True
        return False

    def evict_candidates(self, items: list[WMTMItem]) -> list[str]:
        must_evict = []
        soft_candidates = []
        for item in items:
            if self.should_evict(item):
                must_evict.append(item.id)
            else:
                penalty = self.derived_eviction_mult if item.is_derived else 1.0
                score = item.sti - penalty * item.age * 0.5
                soft_candidates.append((score, item.id))
        soft_candidates.sort()
        excess = len(items) - self.max_items
        if excess > 0:
            for _ in range(excess):
                if soft_candidates:
                    _, cid = soft_candidates.pop(0)
                    must_evict.append(cid)
        return must_evict


class WMTMStore:
    """In-memory working memory with tick-driven maintenance."""

    def __init__(
        self,
        capacity: int = 60,
        decay_rate: float = 0.85,
        forgetting: Optional[ForgettingPolicy] = None,
    ):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.forgetting = forgetting or ForgettingPolicy(max_items=capacity)
        self._items: dict[str, WMTMItem] = {}
        self._cycle: int = 0

    @property
    def cycle(self) -> int:
        return self._cycle

    def admit(
        self,
        item_id: str,
        text: str,
        source_type: str = "recalled",
        origin_cluster: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
        sti: float = 10.0,
    ) -> WMTMItem:
        if item_id in self._items:
            existing = self._items[item_id]
            existing.sti = max(existing.sti, sti)
            existing.touch(self._cycle)
            return existing
        item = WMTMItem(
            id=item_id,
            text=text,
            source_type=source_type,
            origin_cluster=origin_cluster,
            derived_from=derived_from or [],
            sti=sti,
            last_used=self._cycle,
        )
        self._items[item_id] = item
        self._enforce_capacity()
        return item

    def get(self, item_id: str) -> Optional[WMTMItem]:
        return self._items.get(item_id)

    def touch(self, item_id: str) -> None:
        item = self._items.get(item_id)
        if item:
            item.touch(self._cycle)

    def remove(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def get_active_set(self, limit: int = 20) -> list[WMTMItem]:
        return sorted(self._items.values(), key=lambda x: -x.sti)[:limit]

    def all_items(self) -> list[WMTMItem]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def tick(self) -> list[str]:
        """Run one maintenance cycle: decay STI, age items, evict forget candidates.

        Returns list of evicted item IDs.
        """
        self._cycle += 1
        for item in self._items.values():
            item.decay_sti(self.decay_rate)
        evict_ids = self.forgetting.evict_candidates(list(self._items.values()))
        for eid in evict_ids:
            self._items.pop(eid, None)
        if evict_ids:
            log.debug("WMTM tick %d: evicted %d items", self._cycle, len(evict_ids))
        return evict_ids

    def _enforce_capacity(self) -> None:
        if len(self._items) <= self.capacity:
            return
        evict_ids = self.forgetting.evict_candidates(list(self._items.values()))
        for eid in evict_ids:
            self._items.pop(eid, None)
            if len(self._items) <= self.capacity:
                break

    def summary(self) -> dict:
        items = list(self._items.values())
        return {
            "cycle": self._cycle,
            "count": len(items),
            "capacity": self.capacity,
            "recalled": sum(1 for i in items if i.source_type == "recalled"),
            "derived": sum(1 for i in items if i.source_type == "derived"),
            "avg_sti": sum(i.sti for i in items) / max(1, len(items)),
            "max_sti": max((i.sti for i in items), default=0.0),
            "min_sti": min((i.sti for i in items), default=0.0),
        }
