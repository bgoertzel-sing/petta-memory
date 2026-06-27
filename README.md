# petta-memory

Prototype intermediate PeTTa/MeTTa memory store for ProtomegaTron/OmegaClaw.

The store is a bounded append-only `.metta` journal of `MemoryCluster` records.
It is designed to sit between volatile working memory/history and broad vector or
Markdown long-term memory.

Design source:
`projects/hyperseed-formalizations/repos/hyperseed-formalizations/papers/0003-medium-petta-memory-plan/medium_petta_memory_plan.tex`

## v0 goals

- Append complete `MemoryCluster` records only.
- Validate basic MeTTa-like syntax, required metadata, and size limits.
- Query by id, type, `About`, status, and epistemic role.
- Generate bounded prompt context.
- Export a PLN-safe view that excludes raw quoted utterance text and unpromoted quoted claims.

## Non-goals for v0

- No live OmegaClaw integration.
- No autonomous external actions.
- No database service.
- No raw transcript mirroring.
