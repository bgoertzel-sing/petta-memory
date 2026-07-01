# OmegaClaw migration/API sketch

This is a non-live migration sketch for eventual OmegaClaw Core upstreaming. It documents stable v0 names only; no OmegaClaw runtime, secrets, remotes, or live agent state are changed by this repository.

## Proposed API names

- `petta_memory.store.MediumMemoryStore.append_cluster(text)`: reviewed local/manual write path for complete `MemoryCluster` records.
- `MediumMemoryStore.prompt_view(limit_chars, topics, statuses)`: bounded prompt-context atom view. `limit_chars` is validated as non-negative.
- `MediumMemoryStore.index_view(limit_chars)`: derived `MM-index` retrieval view for id/type/about/status/role edges.
- `MediumMemoryStore.pln_view(normalized=True)`: PLN-safe view with explicit promoted-premise metadata.
- `petta_memory.omegaclaw.OmegaClawMemoryBridge.prompt_view_metta()`: future read-only OmegaClaw prompt fragment wrapper.
- `OmegaClawMemoryBridge.append_from_omegaclaw(...)`: intentionally rejected in v0.

## Migration sequence

1. Keep `OmegaClawMemoryPolicy.prompt_view_reads_enabled=False` by default.
2. Add an OmegaClaw Core adapter that can construct `MediumMemoryStore` from a reviewed local journal path, but only returns an empty fragment while the flag is false.
3. Enable read-only prompt fragments for a local test agent with a small `prompt_view_limit_chars` budget and topic/status preferences.
4. Compare prompt fragments against `index_view`/direct query tests before any broader rollout.
5. Defer autonomous writes until a separate review defines provenance, validation, audit logging, rollback, and operator controls.

## Boundary rules

- Prompt-view atoms are context, not authority.
- Generated `MM-index` atoms are derived and should not be appended back into the journal.
- Raw quoted claims remain excluded from PLN-safe premises unless explicitly promoted with rule/trust/domain metadata.
- Negative bounds are rejected rather than interpreted by Python slicing.
