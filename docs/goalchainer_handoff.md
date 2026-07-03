# GoalChainer evidence handoff (non-live)

Status: exploratory contract only. No live OmegaClaw/Protomegabot skill is loaded, no task is claimed, and no memory writes are enabled.

## Provenance

- GoalChainer source inspected in `projects/omegaclaw/repos/OmegaClaw-GoalChainer` at commit `23f49515b1556ce04981f74bde4b56ee0a4375c6`.
- Existing OmegaClaw map: `projects/omegaclaw/GOALCHAINER_INTEGRATION_MAP.md`.
- PeTTaChainer handoff source: `MediumMemoryStore.pettachainer_handoff_cache(...)`, which emits promoted STV statements and explicit-EC `EvidencePacket` atoms while keeping `compileadd`/query gated.

## Contract

`MediumMemoryStore.goalchainer_handoff_cache(...)` and CLI command `goalchainer-handoff-cache` repackage promoted memory evidence as JSON items for a future non-live GoalChainer gate:

- `acceptability-belief-evidence`: promoted PeTTaChainer STV statements, suitable as belief-strength inputs for action acceptability appraisal.
- `contextual-appraisal-evidence`: promoted `EvidencePacket` atoms with explicit support/opposition counts, suitable as contextual evidence inputs.

Each item preserves `belief_id`, `cluster_id`, `promotion_event`, `promotion_rule`, `promotion_domain`, and `promotion_trust` so GoalChainer can attribute its appraisal back to the memory journal.

## Boundaries

- The cache is read-only and derived; do not append it as canonical memory.
- Items are PLN-ready inputs, not inferred beliefs.
- GoalChainer must not claim tasks, write memory, or load a live OmegaClaw skill from this cache.
- The current PeTTaChainer `compileadd` bottleneck remains gated; this handoff supports a heuristic/non-live appraisal gate first.

## Smallest next non-live gate

1. Generate a `goalchainer-handoff-cache` from a hand-picked promoted-memory fixture.
2. Feed one `acceptability-belief-evidence` and, if present, one `contextual-appraisal-evidence` item into a local GoalChainer incident/request harness under timeout.
3. Require a decision payload with ranked actions, a motivation/deontic summary, and provenance pointers, but no live directive/task claim.
4. Archive the input/output artifact before any discussion of live OmegaClaw integration.
