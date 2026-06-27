# petta-memory

Prototype intermediate PeTTa/MeTTa memory store for ProtomegaTron/OmegaClaw.

The store is a bounded append-only `.metta` journal of `MemoryCluster` records.
It is designed to sit between volatile working memory/history and broad vector or
Markdown long-term memory.

Design source:
`projects/hyperseed-formalizations/repos/hyperseed-formalizations/papers/0003-medium-petta-memory-plan/medium_petta_memory_plan.tex`

## v0 goals

- Append complete `MemoryCluster` records only, serialized with explicit begin/end delimiters.
- Require `(SchemaVersion <cluster-id> medium-memory-v1)` in each cluster.
- Validate basic MeTTa-like syntax, required metadata, and size limits.
- Query by cluster/id, type, `About`, status, and epistemic role, returning whole clusters.
- Generate bounded prompt context.
- Export a PLN-safe view that excludes raw quoted utterance text and unpromoted quoted claims.
- Compute current status from append-only `StatusEvent` plus `Supersedes` atoms.

## Non-goals for v0

- No live OmegaClaw integration.
- No autonomous external actions.
- No database service.
- No raw transcript mirroring.

## Canonical record format

Each journal record is one cluster:

```text
;;; BEGIN MemoryCluster mc-example
(MemoryCluster mc-example)
(SchemaVersion mc-example medium-memory-v1)
...
;;; END MemoryCluster mc-example
```

The implementation validates the full cluster before writing, then writes through a
temporary file replacement. This is conservative and local-first; a later OmegaClaw
integration can replace it with file locking or an AtomSpace-backed journal.
