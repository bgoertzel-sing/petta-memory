"""
WMTM RecallBridge — keyword-based recall and spreading activation.

Bridges MediumMemoryStore query methods + ECANBridge attention data
to populate WMTMStore with relevant items.
"""
from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING
import logging
import re

if TYPE_CHECKING:
    from petta_memory.wmtm import WMTMStore
    from petta_memory.store import MediumMemoryStore
    from petta_memory.ecan_bridge import ECANBridge

log = logging.getLogger(__name__)

# Keywords to ignore during extraction
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "or", "but", "if", "because", "as", "until", "while", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "whose", "it's", "its",
})


def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    """Extract meaningful keywords from text, filtering stop words."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in _STOP_WORDS]


class RecallBridge:
    """Recalls items from LTM into WMTM using keyword matching + ECAN attention."""

    def __init__(
        self,
        store: "MediumMemoryStore",
        ecan_bridge: "ECANBridge",
        wmtm: "WMTMStore",
    ):
        self.store = store
        self.ecan = ecan_bridge
        self.wmtm = wmtm

    def recall(
        self,
        query_context: str,
        top_k: int = 20,
        sti_boost: float = 5.0,
    ) -> list[str]:
        """Recall items from LTM matching the query context.

        1. Extract keywords from query_context
        2. For each keyword: store.query_about(keyword)
        3. Score by keyword overlap x ECAN STI x recency
        4. Admit top_k into WMTM

        Returns list of admitted item IDs.
        """
        keywords = extract_keywords(query_context)
        if not keywords:
            log.debug("RecallBridge: no keywords extracted from context")
            return []

        # Get ECAN prioritized beliefs for STI scoring
        try:
            prioritized = dict(self.ecan.get_prioritized_beliefs(limit=500))
        except Exception:
            prioritized = {}

        # Collect candidates from keyword queries
        candidates: dict[str, dict] = {}  # id -> {score, text, keywords_matched}
        for kw in keywords:
            try:
                clusters = self.store.query_about(kw, limit=20)
            except Exception:
                clusters = []
            for cluster in clusters:
                cid = getattr(cluster, "id", None) or str(cluster)
                if cid in candidates:
                    candidates[cid]["keywords_matched"] += 1
                else:
                    text = getattr(cluster, "text", str(cluster))
                    candidates[cid] = {
                        "text": text,
                        "keywords_matched": 1,
                        "sti": prioritized.get(cid, 0.0),
                    }

        if not candidates:
            log.debug("RecallBridge: no candidates found for keywords: %s", keywords)
            return []

        # Score: keyword_overlap * max(1, sti) * recency
        scored = []
        for cid, info in candidates.items():
            kw_score = info["keywords_matched"] / len(keywords)
            sti_score = max(1.0, info["sti"])
            combined = kw_score * sti_score
            scored.append((combined, cid, info["text"]))

        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        admitted = []
        for score, cid, text in top:
            initial_sti = 10.0 + min(score * 20.0, 30.0)  # cap boost
            self.wmtm.admit(
                item_id=cid,
                text=text,
                source_type="recalled",
                origin_cluster=cid,
                sti=initial_sti,
            )
            admitted.append(cid)

        log.debug("RecallBridge: admitted %d items from %d candidates", len(admitted), len(candidates))
        return admitted

    def spreading_activation(
        self,
        seed_ids: list[str],
        depth: int = 2,
        decay_factor: float = 0.7,
        threshold: float = 0.1,
        max_new: int = 15,
    ) -> list[str]:
        """Spread activation from seed items through EvidenceFor links.

        1. Start with seed_ids (already in WMTM)
        2. Follow evidence links via ECANBridge.get_evidence_map()
        3. For each neighbor: parent_sti * decay_factor^depth
        4. Collect items above threshold
        5. Admit into WMTM

        Returns list of newly admitted item IDs.
        """
        try:
            evidence_map = self.ecan.get_evidence_map()
        except Exception:
            log.debug("RecallBridge: no evidence map available")
            return []

        # Build adjacency from evidence map
        # evidence_map is typically {belief_id: [evidence_ids]} or similar
        adjacency: dict[str, list[str]] = {}
        if isinstance(evidence_map, dict):
            for bid, evidences in evidence_map.items():
                if isinstance(evidences, (list, tuple)):
                    adjacency.setdefault(bid, [])
                    for ev in evidences:
                        ev_id = ev if isinstance(ev, str) else getattr(ev, "id", str(ev))
                        adjacency[bid].append(ev_id)
                        adjacency.setdefault(ev_id, []).append(bid)

        if not adjacency:
            log.debug("RecallBridge: no adjacency from evidence map")
            return []

        # BFS with decay
        visited = set(seed_ids)
        frontier = list(seed_ids)
        new_items: list[tuple[float, str]] = []

        for hop in range(1, depth + 1):
            next_frontier = []
            for node_id in frontier:
                seed_item = self.wmtm.get(node_id)
                seed_sti = seed_item.sti if seed_item else 10.0
                neighbors = adjacency.get(node_id, [])
                for nbr_id in neighbors:
                    if nbr_id in visited:
                        continue
                    visited.add(nbr_id)
                    activation = seed_sti * (decay_factor ** hop)
                    if activation >= threshold:
                        new_items.append((activation, nbr_id))
                        next_frontier.append(nbr_id)
            frontier = next_frontier
            if not frontier:
                break

        # Sort by activation, admit top max_new
        new_items.sort(key=lambda x: -x[0])
        admitted = []
        for activation, nbr_id in new_items[:max_new]:
            if nbr_id in self.wmtm:
                # Already in WMTM - just boost STI
                item = self.wmtm.get(nbr_id)
                if item:
                    item.sti += activation * 0.5
                continue
            # Try to get text from store
            try:
                cluster = self.store.query_cluster(nbr_id)
                text = getattr(cluster, "text", nbr_id) if cluster else nbr_id
            except Exception:
                text = nbr_id
            self.wmtm.admit(
                item_id=nbr_id,
                text=text,
                source_type="recalled",
                origin_cluster=nbr_id,
                sti=activation,
            )
            admitted.append(nbr_id)

        log.debug("RecallBridge: spreading activation admitted %d new items", len(admitted))
        return admitted

    def recall_and_spread(
        self,
        query_context: str,
        top_k: int = 15,
        spread_depth: int = 2,
    ) -> list[str]:
        """Convenience: recall then spread from recalled items."""
        recalled = self.recall(query_context, top_k=top_k)
        if recalled:
            spread = self.spreading_activation(recalled, depth=spread_depth)
            return recalled + spread
        return recalled
