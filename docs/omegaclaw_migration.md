# OmegaClaw migration/API sketch

This is a non-live migration sketch for eventual OmegaClaw Core upstreaming. It documents stable v0 names only; no OmegaClaw runtime, secrets, remotes, or live agent state are changed by this repository.

## Proposed API names

- `petta_memory.store.MediumMemoryStore.append_cluster(text)`: reviewed local/manual write path for complete `MemoryCluster` records.
- `MediumMemoryStore.prompt_view(limit_chars, topics, statuses)`: bounded prompt-context atom view. `limit_chars` is validated as non-negative.
- `MediumMemoryStore.index_view(limit_chars)`: derived `MM-index` retrieval view for id/type/about/status/role edges.
- `MediumMemoryStore.pln_view(normalized=True)`: PLN-safe view with explicit promoted-premise metadata.
- `petta_memory.omegaclaw.OmegaClawMemoryBridge.prompt_view_metta()`: future read-only OmegaClaw prompt fragment wrapper.
- `OmegaClawMemoryBridge.index_view_metta()`: future read-only-derived OmegaClaw `MM-index` fragment wrapper for retrieval checks.
- `OmegaClawMemoryBridge.append_from_omegaclaw(...)`: intentionally rejected in v0.

## Migration sequence

1. Keep `OmegaClawMemoryPolicy.prompt_view_reads_enabled=False` and `index_view_reads_enabled=False` by default.
2. Add an OmegaClaw Core adapter that can construct `MediumMemoryStore` from a reviewed local journal path, but only returns an empty fragment while the flag is false.
3. Enable read-only prompt fragments for a local test agent with a small `prompt_view_limit_chars` budget and topic/status preferences.
4. Enable read-only-derived `MM-index` fragments only for local retrieval diagnostics, with a separate `index_view_limit_chars` budget.
5. Compare prompt fragments against `index_view`/direct query tests before any broader rollout.
6. Defer autonomous writes until a separate review defines provenance, validation, audit logging, rollback, and operator controls.

## Boundary rules

- Prompt-view atoms are context, not authority.
- Generated `MM-index` atoms are derived lookup hints and must not be appended back into the journal.
- Raw quoted claims remain excluded from PLN-safe premises unless explicitly promoted with rule/trust/domain metadata.
- Negative bounds are rejected rather than interpreted by Python slicing.
- Prompt and index read wrappers have separate feature flags and validated wrapper ids so an integration can enable/disable them independently.

## Live OmegaClaw integration review gate

Before any live OmegaClaw runtime wiring is enabled, a reviewer should verify the
following boundary in the OmegaClaw-side adapter:

1. The journal path is an explicit local allowlisted path, not inferred from chat
   input, model output, environment secrets, or a remote URL.
2. Prompt and index reads are separately feature-flagged and default off in live
   configs; enabling one does not enable the other.
3. Prompt/index budgets are small fixed integers in config, validated as
   non-negative, and cannot be raised by model output.
4. Prompt fragments are labeled as read-only context; generated `MM-index` atoms
   are labeled as derived lookup hints and are never appended to the journal.
5. All writes from OmegaClaw are rejected unless a later design adds reviewed
   provenance, validation, operator controls, audit logging, failure handling, and
   rollback semantics.
6. Optional PeTTa runtime parse checks run before append on reviewed local/manual
   write paths only; parse-check failure leaves the journal unchanged.
7. Integration tests compare the live adapter output to `prompt_view`,
   `index_view`, and `audit_view` on a fixture journal before any broader rollout.
