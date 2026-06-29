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

## OmegaClaw integration sketch: feature flags and boundary

`petta_memory.omegaclaw` contains a local-only wrapper sketch for future OmegaClaw
prompt assembly. It is not imported by OmegaClaw and does not touch any live agent
state.

Feature flags are explicit and default-safe:

- `prompt_view_reads_enabled=False` by default. When false, the wrapper returns an
  empty prompt fragment. When true, it returns only the bounded `prompt_view` atoms
  from a caller-supplied local `MediumMemoryStore`, wrapped in a read-only MeTTa
  envelope.
- `autonomous_writes_enabled=False` is enforced. Setting it to true raises
  `LiveWriteDisabled`, and `OmegaClawMemoryBridge.append_from_omegaclaw(...)`
  always raises in v0.

Intended read/write boundary:

1. **Prompt-view reads:** OmegaClaw may later read a bounded read-only fragment via
   `OmegaClawMemoryBridge.prompt_view_metta()` after an integration review. These
   atoms are prompt context, not new authority.
2. **Manual/local writes:** repository tests and reviewed migration scripts may use
   `MediumMemoryStore.append_cluster(...)` directly against local files.
3. **Autonomous memory writes:** disabled until a separate design review defines
   validation, provenance, failure handling, audit logging, and rollback semantics.

Example wrapper shape:

```text
;;; BEGIN OmegaClawPromptView oc-prompt-memory-view
(OmegaClawPromptView oc-prompt-memory-view)
(PromptViewSource oc-prompt-memory-view petta-memory)
(PromptViewMode oc-prompt-memory-view read-only)
(PromptViewGeneratedAt oc-prompt-memory-view "2026-06-29T18:10:00+00:00")
...
;;; END OmegaClawPromptView oc-prompt-memory-view
```

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
