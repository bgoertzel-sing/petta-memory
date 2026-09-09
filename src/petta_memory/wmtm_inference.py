"""
WMTM InferenceEngine — derivation tracking and inference over working memory items.

Tracks which items were derived from which, and provides simple inference
operations (conjunction, implication) that produce new WMTMItems.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from petta_memory.wmtm import WMTMStore, WMTMItem

log = logging.getLogger(__name__)


class WMTMInferenceEngine:
    """Derives new items from active working memory set."""

    def __init__(self, wmtm: "WMTMStore"):
        self.wmtm = wmtm
        self._derivation_count = 0

    @property
    def derivation_count(self) -> int:
        return self._derivation_count

    # F03 FIX: Valid rule names for derivation
    _VALID_RULES = frozenset({"conjunction", "implication", "abduction",
                              "induction", "deduction", "analogy"})

    def derive(
        self,
        text: str,
        source_ids: list[str],
        sti: Optional[float] = None,
        rule: str = "conjunction",
    ) -> Optional["WMTMItem"]:
        """Derive a new item from source items.

        The derived item's STI is the average of source STIs * 0.8 (slight discount).
        If sti is provided explicitly, it overrides.

        F03 FIX: Rejects missing parent IDs and unknown rule names.
        """
        # F03 FIX: Validate rule
        if rule not in self._VALID_RULES:
            log.warning("Inference: unknown rule '%s'", rule)
            return None

        # F03 FIX: All source_ids must be present in WMTM
        missing = [sid for sid in source_ids if sid not in self.wmtm]
        if missing:
            log.warning("Inference: missing source IDs: %s", missing)
            return None

        sources = [self.wmtm.get(sid) for sid in source_ids if sid in self.wmtm]
        if not sources:
            log.debug("Inference: no valid sources for derivation")
            return None

        if sti is None:
            avg_sti = sum(s.sti for s in sources) / len(sources)
            sti = avg_sti * 0.8

        self._derivation_count += 1
        deriv_id = f"deriv-{self._derivation_count}"

        item = self.wmtm.admit(
            item_id=deriv_id,
            text=text,
            source_type="derived",
            derived_from=source_ids,
            sti=sti,
        )
        log.debug("Inference: derived '%s' from %d sources (STI=%.1f)", text[:40], len(sources), sti)
        return item

    def derive_conjunction(self, id_a: str, id_b: str) -> Optional["WMTMItem"]:
        """Derive a conjunction: A AND B."""
        a = self.wmtm.get(id_a)
        b = self.wmtm.get(id_b)
        if not a or not b:
            return None
        text = f"{a.text} AND {b.text}"
        return self.derive(text, [id_a, id_b], rule="conjunction")

    def derive_implication(self, antecedent_id: str, consequent_id: str) -> Optional["WMTMItem"]:
        """Derive an implication: A -> B."""
        ant = self.wmtm.get(antecedent_id)
        con = self.wmtm.get(consequent_id)
        if not ant or not con:
            return None
        text = f"{ant.text} -> {con.text}"
        return self.derive(text, [antecedent_id, consequent_id], rule="implication")

    def get_provenance(self, item_id: str, depth: int = 5) -> dict:
        """Trace the provenance chain of a derived item.

        Returns nested dict: {id, text, source_type, derived_from: [...]}

        F04 FIX: When a source item has been evicted, returns a placeholder
        dict with id and evicted=True instead of None, preserving the chain.
        """
        visited = set()

        def _trace(cid: str, d: int) -> Optional[dict]:
            if cid in visited or d <= 0:
                return None
            visited.add(cid)
            item = self.wmtm.get(cid)
            if not item:
                # F04 FIX: Return placeholder for evicted items
                return {"id": cid, "evicted": True}
            result = {
                "id": item.id,
                "text": item.text[:80],
                "source_type": item.source_type,
                "sti": item.sti,
            }
            if item.derived_from:
                result["derived_from"] = [
                    _trace(sid, d - 1) for sid in item.derived_from
                ]
                result["derived_from"] = [x for x in result["derived_from"] if x]
            return result

        return _trace(item_id, depth)

    def summary(self) -> dict:
        items = self.wmtm.all_items()
        derived = [i for i in items if i.is_derived]
        return {
            "total_derivations": self._derivation_count,
            "active_derived": len(derived),
            "avg_derived_sti": sum(i.sti for i in derived) / max(1, len(derived)),
        }
