WMTM Code Review - Defect Report and Fix Plan
=============================================
Reviewer: Iter Agent (autonomous review pass)
Date: 2026-09-08T22:00Z
Branch: wmtm-review-fixes (from main @ f90c848)
Files: wmtm.py, wmtm_inference.py, wmtm_utility.py,
       wmtm_coordinator.py, wmtm_recall.py, ecan_bridge.py,
       object_index.py, store.py
Tests: 864 pass (28 files), 0 fail

======================================================
CONFIRMED DEFECTS (all reproduced with test scripts)
======================================================

D01 - Writeback is NOT idempotent (duplicate clusters)
  Module: wmtm_utility.py:_build_cluster_text()
  Severity: HIGH
  Repro: writeback(['d1']) called twice -> 2 clusters appended
  Root cause: _build_cluster_text() generates a new random UUID
    each call. No tracking of which items have already been
    persisted. The store's _reject_duplicate_ids only catches
    identical cluster IDs, but each call produces a different UUID.
  Impact: In production, on_turn_end() auto-writeback runs every
    turn. A derived item that stays in WMTM for 10 turns gets
    written back 10 times, creating 10 duplicate clusters in LTM.

D02 - Cluster ID vs Belief ID mismatch (STI boost silently dropped)
  Module: wmtm_utility.py:writeback() + ecan_bridge.py:stimulate_beliefs()
  Severity: HIGH
  Repro: Admit 'mc-a' as recalled item -> writeback -> ecan.stimulate
    passes 'mc-a' but _belief_ids = {'belief-a'} -> STI stays 0.0
  Root cause: RecallBridge.recall() admits items with
    item_id = cluster_id and origin_cluster = cluster_id.
    But ECAN tracks BELIEF IDs (e.g. 'belief-a'), not cluster IDs.
    stimulate_beliefs() checks `if belief_id in self._belief_ids`
    and silently skips unmatched IDs.
  Impact: Recalled items NEVER get their STI boosted in ECAN.
    The entire writeback-for-recalled-items pathway is dead code.

D03 - Spreading activation uses cluster IDs but evidence map
      uses belief IDs (finds 0 neighbors)
  Module: wmtm_recall.py:spreading_activation()
  Severity: HIGH
  Repro: recall returns ['mc-a'] -> spreading_activation(['mc-a'])
    -> adjacency built from evidence_map (keyed by belief IDs)
    -> adjacency.get('mc-a', []) = [] -> 0 new items
  Root cause: Same ID-domain mismatch as D02. The seed IDs from
    recall are cluster IDs, but the evidence map is keyed by
    belief IDs. BFS finds no neighbors.
  Impact: Spreading activation is completely non-functional.
    Related beliefs connected via EvidenceFor edges are never
    discovered through graph traversal.

D04 - Double-tick per turn (2x decay per turn)
  Module: wmtm_coordinator.py:on_tick() + on_turn_end()
  Severity: MEDIUM
  Repro: on_tick() calls wmtm.tick(), on_turn_end() calls
    wmtm.tick() again -> 2 ticks per turn, _turn_count += 1
  Impact: STI decays 2x faster than intended. An item with
    sti=100 after 10 turns (with both calls) reaches 100*0.85^20
    = 3.9, but should be 100*0.85^10 = 19.7. Items are evicted
    ~2x too fast.

D05 - ECAN cycles never run in coordinator
  Module: wmtm_coordinator.py:on_tick()
  Severity: MEDIUM
  Root cause: on_tick() calls ecan.sync_from_store() but never
    calls ecan.run_cycle(). ECAN attention values are populated
    but diffusion, rent collection, and decay never execute.
  Impact: ECAN attention is static - no spreading, no rent-based
    forgetting. The WMTM has its own decay but ECAN's graph-aware
    dynamics are completely bypassed.

D06 - Derivation counter collision across engine instances
  Module: wmtm_inference.py:derive()
  Severity: LOW
  Repro: Two WMTMInferenceEngine instances sharing same WMTMStore
    both produce deriv-1 -> second overwrites first in admit()
  Root cause: _derivation_count starts at 0 per instance. No
    coordination with the shared store.
  Note: The test test_local_counter expects this behavior
    (separate engine = separate counter). This is by design
    for separate engines, but fragile if engines share a store.

D07 - decay_sti increments age before computing recency
  Module: wmtm.py:decay_sti()
  Severity: LOW
  Root cause: self.age += 1 runs before recency_factor =
    0.9 ** max(0, self.age - self.last_used). On the first tick
    after admission (age=0->1, last_used=0), recency = 0.9^1 = 0.9
    instead of 1.0.
  Impact: Newly admitted items decay slightly faster than intended.
    Over 50 ticks, the cumulative effect is ~10% extra decay.

======================================================
FIX PLAN
======================================================

FIX D01: Writeback idempotency
  - Add _written_back: set[str] to WMTMUtility
  - Track item IDs that have been successfully persisted
  - Skip items already in _written_back
  - Clear tracking on explicit request or when item is evicted
  - New test: test_writeback_idempotent

FIX D02+D03: Cluster ID -> Belief ID mapping
  - Add origin_belief_ids: list[str] field to WMTMItem
  - RecallBridge uses ObjectIndex to find belief IDs in each cluster
  - recall() stores belief_ids in WMTMItem.origin_belief_ids
  - spreading_activation uses belief_ids for adjacency lookup
  - writeback uses belief_ids for ECAN stimulate
  - New tests: test_recall_tracks_belief_ids,
    test_spreading_finds_neighbors, test_writeback_stimulates_ecan

FIX D04: Single tick per turn
  - Add _ticked_this_turn: bool flag to coordinator
  - Set in on_tick(), clear in on_turn_end()
  - on_turn_end() only ticks if not already ticked this turn
  - New test: test_no_double_tick

FIX D05: ECAN cycle in coordinator
  - on_tick() calls ecan.run_cycle() after sync_from_store()
  - New test: test_ecan_cycle_runs

FIX D06: Unique derivation IDs
  - Move derivation counter to WMTMStore (shared across engines)
  - derive() asks store for next derivation ID
  - New test: test_unique_derivation_ids

FIX D07: Recency off-by-one
  - Compute recency BEFORE incrementing age
  - New test: test_recency_first_tick
