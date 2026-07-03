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

## Current non-live gate

`goalchainer-smoke` defaults to a precompiled handoff-cache gate while the external GoalChainer CLI path remains blocked by PeTTaChainer `compileadd`. The precompiled gate imports only GoalChainer scenario/scoring/explanation modules, ranks actions from promoted `Acceptable` STV evidence, and now folds matching contextual `EvidencePacket` EC support/opposition into the appraisal as bounded derived strength/confidence. The EC path is provenance-preserving and non-live: it does not call PeTTaChainer `compileadd`/query, claim directives/tasks, load OmegaClaw skills, or write memory.

A valid gate artifact must show ranked actions, a recommended action, selected handoff provenance, `compileadd_not_invoked`, `no_live_directive_or_task_claim`, and `no_memory_write` before any discussion of live OmegaClaw integration.
