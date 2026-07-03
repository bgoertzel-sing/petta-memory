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
- Validate basic MeTTa-like syntax, required metadata, delimited record envelope/atom id consistency, unary ID-declaration and binary metadata/retrieval relation arity, symbol IDs, local `Contains` boundaries including self-containment rejection, and size limits.
- Allow a caller-supplied parse-check hook for external PeTTa/MeTTa runtime validation; `make_petta_parse_checker(...)` wires this to a local PeTTa runtime when explicitly requested.
- Query by cluster/id, type, `About`, status, and epistemic role, returning whole clusters.
- Generate a bounded audit view of complete canonical `MemoryCluster` records for human/review tooling, preserving begin/end delimiters instead of slicing through records.
- Generate a bounded `MM-index` view for id/type/about/status/role retrieval edges, with id edges for valid identifier arguments so generated index recall can match direct `query_id` recall; bounded index output preserves complete atom lines.
- Generate bounded prompt context, with optional topic/status preferences and
  salience/recency ordering; fixture tests cover relevance under a tight prompt
  character budget, negative bounds are rejected, and bounded output preserves
  complete atom lines.
- Export a PLN-safe view that excludes raw quoted utterance text and unpromoted quoted claims; optional PLN-view character bounds preserve complete atom lines.
- Export promoted beliefs as PeTTaChainer-compatible `(: proof-id statement (STV strength confidence))` statements via `pettachainer-view`; confidence is capped by `PromotionTrust`.
- Export promoted beliefs with explicit `EvidenceSupportCount`/`EvidenceOppositionCount` atoms as PeTTaChainer `EvidencePacket` atoms via `pettachainer-packets-view`; EC counts are never inferred from truth values.
- Emit a non-live JSON handoff cache via `pettachainer-handoff-cache`, packaging promoted STV statements and EvidencePackets as PLN-ready inputs for review/OmegaClaw/GoalChainer mapping while explicitly labeling them as not inferred beliefs and keeping PeTTaChainer `compileadd`/query gated.
- Emit a GoalChainer-facing non-live JSON handoff via `goalchainer-handoff-cache`, mapping promoted evidence into appraisal/acceptability input slots with explicit no-task-claim/no-live-skill boundaries; see `docs/goalchainer_handoff.md`.
- Generate narrow PeTTaChainer profile workloads with `python -m petta_memory.pettachainer_profile`, covering promoted-belief STV proof statements and EvidencePacket exports; opt-in runtime constructor, direct-vs-eval-control internal `compileadd` probes, proof/contextual add-only, and add+query stages run in bounded subprocesses via `--stage-timeout-sec` because they are noisy/slow locally.
- Compute current status from append-only `StatusEvent` plus `Supersedes` atoms.
- Require explicit promotion rule, bounded trust, and domain metadata before derived beliefs are exported as PLN premises; `pln-view --normalized` adds normalized `MM-PLN*` mapping atoms for eligible beliefs.

## Non-goals for v0

- No live OmegaClaw integration.
- No autonomous external actions.
- No database service.
- No raw transcript mirroring.

## OmegaClaw integration sketch: feature flags and boundary

See also `docs/omegaclaw_migration.md` for proposed migration/API names.

`petta_memory.omegaclaw` contains a local-only wrapper sketch for future OmegaClaw
prompt assembly. It is not imported by OmegaClaw and does not touch any live agent
state.

Feature flags are explicit and default-safe:

- `prompt_view_reads_enabled=False` by default. When false, the wrapper returns an
  empty prompt fragment. When true, it returns only the bounded `prompt_view` atoms
  from a caller-supplied local `MediumMemoryStore`, wrapped in a read-only MeTTa
  envelope with a validated symbol id and escaped generated-at string.
- `index_view_reads_enabled=False` by default. When true, the wrapper returns a
  separately bounded, read-only-derived `MM-index` envelope for id/type/about/status/role
  retrieval checks; the generated index is never appended back into the journal.
- `autonomous_writes_enabled=False` is enforced. Setting it to true raises
  `LiveWriteDisabled`, and `OmegaClawMemoryBridge.append_from_omegaclaw(...)`
  always raises in v0.

Intended read/write boundary:

1. **Prompt-view reads:** OmegaClaw may later read a bounded read-only fragment via
   `OmegaClawMemoryBridge.prompt_view_metta()` after an integration review. These
   atoms are prompt context, not new authority.
2. **Generated-index reads:** OmegaClaw may later read a bounded derived retrieval
   fragment via `OmegaClawMemoryBridge.index_view_metta()` after the same review;
   these atoms are lookup hints, not canonical memory.
3. **Manual/local writes:** repository tests and reviewed migration scripts may use
   `MediumMemoryStore.append_cluster(...)` directly against local files.
4. **Autonomous memory writes:** disabled until a separate design review defines
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

The implementation validates the full cluster before writing, optionally runs a
caller-supplied parse-check hook over the canonicalized cluster, then writes through
a temporary file replacement. `petta_memory.make_petta_parse_checker(...)` can be
passed as that hook to check the canonical cluster with an explicitly configured
local PeTTa runtime; it is opt-in and does not enable live OmegaClaw writes. This
is conservative and local-first; a later OmegaClaw integration can replace it
with an AtomSpace-backed journal.
