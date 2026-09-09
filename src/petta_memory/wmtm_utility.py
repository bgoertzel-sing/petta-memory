"""
WMTM Utility — utility scoring and writeback to LTM.

Scores items by usefulness, and writes high-utility items back to
MediumMemoryStore for persistence in long-term memory.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import logging
import hashlib
import time

if TYPE_CHECKING:
    from petta_memory.wmtm import WMTMStore, WMTMItem
    from petta_memory.store import MediumMemoryStore
    from petta_memory.ecan_bridge import ECANBridge

log = logging.getLogger(__name__)


def _sexpr_escape(text):
    """Escape text for safe S-expression embedding."""
    text = text.replace(chr(92), chr(92)+chr(92))
    text = text.replace(chr(34), chr(92)+chr(34))
    text = text.replace(chr(10), chr(92)+chr(110))
    return text


def _stable_derivation_id(text, derived_from):
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    for pid in sorted(derived_from):
        h.update(pid.encode("utf-8"))
    return "wmtm-" + h.hexdigest()[:12]


class WMTMUtility:
    def __init__(self, wmtm, store, ecan_bridge, writeback_threshold=5.0, sti_writeback_boost=3.0):
        self.wmtm = wmtm
        self.store = store
        self.ecan = ecan_bridge
        self.writeback_threshold = writeback_threshold
        self.sti_writeback_boost = sti_writeback_boost
        self._writeback_receipts = {}

    def compute_utility(self, item):
        recency = 0.9 ** max(0, item.age - item.last_used)
        age_penalty = 0.1 * item.age
        return item.use_count * recency + item.sti * 0.3 - age_penalty

    def score_all(self):
        return {item.id: self.compute_utility(item) for item in self.wmtm.all_items()}

    def get_writeback_candidates(self):
        scored = [(self.compute_utility(i), i) for i in self.wmtm.all_items()]
        scored.sort(key=lambda x: -x[0])
        return [item for score, item in scored if score >= self.writeback_threshold]

    def _build_cluster_text(self, item):
        cluster_id = _stable_derivation_id(item.text, item.derived_from)
        belief_id = cluster_id + "-bl"
        reasoning_event_id = cluster_id + "-re"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        escaped_text = _sexpr_escape(item.text)
        source_desc = "wmtm-derived"
        if item.derived_from:
            joined = ",".join(item.derived_from[:3])
            source_desc = "wmtm-derived-from:" + joined
        # Use .format() with single-quoted format strings to avoid quote escaping
        atoms = [
            '(MemoryCluster {})'.format(cluster_id),
            '(SchemaVersion {} medium-memory-v1)'.format(cluster_id),
            '(ClusterType {} belief-record)'.format(cluster_id),
            '(ClusterOpenedAt {} "{}")'.format(cluster_id, timestamp),
            '(ClusterSource {} {})'.format(cluster_id, source_desc),
            '(Contains {} {})'.format(cluster_id, belief_id),
            '(DerivedBelief {})'.format(belief_id),
            '(BeliefContent {} "{}")'.format(belief_id, escaped_text),
            '(About {} "{}")'.format(belief_id, escaped_text),
            '(Contains {} {})'.format(cluster_id, reasoning_event_id),
            '(ObservedEvent {})'.format(reasoning_event_id),
            '(DerivedAt {} cycle:{})'.format(reasoning_event_id, item.age),
            '(Produced {} {})'.format(reasoning_event_id, belief_id),
        ]
        for parent_id in item.derived_from:
            atoms.append('(EvidenceFor {} {})'.format(belief_id, parent_id))
        return chr(10).join(atoms) + chr(10)

    def writeback(self, item_ids=None):
        if item_ids is None:
            candidates = self.get_writeback_candidates()
        else:
            candidates = [self.wmtm.get(i) for i in item_ids if i in self.wmtm]
            candidates = [c for c in candidates if c is not None]
        written = []
        stimuli = {}
        for item in candidates:
            if item.written_back:
                continue
            if item.id in self._writeback_receipts:
                item.written_back = True
                continue
            if item.source_type == "recalled" and item.origin_cluster:
                belief_ids = self._resolve_belief_ids(item.origin_cluster)
                for bid in belief_ids:
                    stimuli[bid] = self.sti_writeback_boost
                item.written_back = True
                self._writeback_receipts[item.id] = item.origin_cluster
                written.append(item.id)
            elif item.source_type == "derived":
                try:
                    cluster_text = self._build_cluster_text(item)
                    cluster_id = _stable_derivation_id(item.text, item.derived_from)
                    if hasattr(self.store, "append_cluster"):
                        self.store.append_cluster(cluster_text)
                    elif hasattr(self.store, "append"):
                        self.store.append(cluster_text)
                    item.written_back = True
                    self._writeback_receipts[item.id] = cluster_id
                    written.append(item.id)
                except Exception as e:
                    log.warning("Writeback failed for %s: %s", item.id, e)
        if stimuli:
            try:
                self.ecan.stimulate_beliefs(stimuli)
            except Exception as e:
                log.warning("ECAN stimulation failed: %s", e)
        log.debug("Utility: wrote back %d items", len(written))
        return written

    def _resolve_belief_ids(self, cluster_id):
        try:
            from petta_memory.object_index import ObjectIndex
            idx = ObjectIndex()
            idx.build_from_store(self.store)
            objects = idx.objects_in_cluster(cluster_id)
            belief_ids = [oid for oid, otype in objects if otype == "DerivedBelief"]
            if belief_ids:
                return belief_ids
        except Exception:
            pass
        return [cluster_id]

    def summary(self):
        scores = self.score_all()
        values = list(scores.values())
        return {
            "total_items": len(scores),
            "avg_utility": sum(values) / max(1, len(values)),
            "max_utility": max(values) if values else 0.0,
            "writeback_candidates": sum(1 for v in values if v >= self.writeback_threshold),
        }
