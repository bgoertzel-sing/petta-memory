# Atlas-indexed πPLN implementation status

Normative design source: `library/atlas-indexed-reversible-pipln/specification.pdf` (SHA-256 `1af20c7427b484a978507181c44fb32112257f029aa130c30f1c8a45d0d7f0d3`).

## Baseline

- patham9 PLN checkout: `../patham9-pln`, commit `55f1751d993f71b8a24da03e3aec94ab40789a59`.
- Existing wrapper functions and serialized artifacts remain compatibility baselines.
- `ec_projected_stv()` implements the legacy `adapter-weighted-v1` behavior, not the canonical count/prior chart projection. `EC_PROJECTED_STV_POLICY_ID` and the function's introspection metadata label that compatibility boundary without changing its serialized result dictionaries.
- Existing numeric runtime stamps are item-position aliases and are not yet the specification's evidence-basis stamp map.
- Vendor smoke history includes examples/rule tests that emit semantic failure markers despite exit code zero; Phase 0 must freeze and classify the expected baseline before claiming differential compatibility.

## Implemented nucleus

`petta_memory.pipln_models` provides the first dependency-free Phase-1 primitives:

- immutable validated `EvidenceToken`, `EvidencePacket`, and `EvidenceBasis` records;
- immutable create-once JSON snapshot documents with a canonical payload checksum and fail-closed loader; v2 snapshots commit a digest for each packet's complete frozen semantic content and derive the snapshot fingerprint from those digests, preventing a caller from compiling changed packet content under an unchanged snapshot identity;
- a content-addressed `EvidenceSnapshotRepository` that discovers validated snapshot documents, rejects filename/fingerprint drift and duplicate logical snapshot IDs, and provides fail-closed lookup;
- immutable semantic `PiContext`, `ChartPolicy`, and `PiChart` records; chart construction accepts a validated `EvidenceSnapshot` rather than an independent snapshot ID, requires matching context and selected-packet membership, and fingerprints the snapshot's semantic content identity;
- stable packet provenance digests;
- deterministic, collision-free episode stamp assignment sorted by stable basis ID;
- canonical `pipl-local-chart-v1` projection with separate evidence mass, conflict balance, signed tendency, STV, and beta parameters;
- exact `EvidenceCapsule` union-by-basis algebra that deduplicates identical shared contributions and fails closed on conflicting weights for the same basis ID.
- the first pure Phase-2 compiler boundary: `compile_episode_inputs()` closes a chart against its immutable snapshot, exact selected packets, and packet-derived basis records; assigns deterministic episode stamps; applies canonical local-chart projection; and emits immutable patham9 `Sentence` atoms with `KernelSentenceMeta` provenance sidecars. It parses and canonicalizes each data term, rejects nested MeTTa executable/control forms, and enforces explicit sentence-count and emitted-character budgets before any runtime boundary;
- immutable checksummed persistence for `CompiledEpisodeInputs`: create-once JSON artifacts round-trip the exact sentences, projections, stamp map, and provenance metadata while rejecting document checksum drift, sentence/typed-metadata mismatch, executable-term smuggling, invalid projection values, and inconsistent stamp/basis mappings. This is a replay input artifact, not the complete runtime `EpisodeManifest`.
- the first Phase-2 kernel-result validator: `validate_kernel_result()` accepts only one bounded patham9 `((stv S C) (stamps...))` data atom, requires finite unit-interval truth values and canonical sorted stamps, rejects output injection and unknown episode stamps, and returns a digest-bound `ValidatedKernelResult` whose stamps close exactly to compiled evidence-basis provenance. It does not infer rule identity or authorize promotion.
- immutable checksummed persistence for `ValidatedKernelResult`: create-once v1 JSON artifacts reject envelope/checksum/type drift and reload only when episode/chart identity, canonical stamps, and exact stamp-derived evidence-basis IDs close against the supplied `CompiledEpisodeInputs`. This persists validated output for later manifests but does not constitute execution replay.
- a complete typed SDS section 16.2 `EpisodeManifest` audit boundary: `build_episode_manifest()` closes the chart, snapshot, compiled inputs, validated result, complete supplied program, stamp map, kernel/controller identities, explicit budget and seed, timestamps, process return code, and captured stdout/stderr into content digests. Every compiled Sentence must occur exactly once in the bounded program. Create-once checksummed persistence reconstructs all typed invariants and rejects recomputed-envelope semantic drift. It records a caller-supplied completed run; it does not invoke patham9, decode a trace, or authorize promotion.
- deterministic stock-kernel query-program assembly: `assemble_legacy_kernel_query_program()` inserts only immutable compiler-emitted Sentences and one already-canonical declarative query into fixed `PLN` import/init/query controls, caps steps at 10,000, each queue at 100,000, and total program size, and admits no caller-supplied rule or executable text. An explicit optional local parse-check hook can fail closed on the complete assembled program before handoff. It returns an inert program string and does not invoke patham9.
- a bounded shell-free subprocess/capture primitive: `run_kernel_subprocess()` passes one already-assembled program on stdin to an explicit argv, rejects oversized UTF-8 program, argv, and optional working-directory launch inputs plus embedded NULs before launch, and counts OS NUL/`KEY=` framing in argv/cwd/environment ceilings. It can require an exact lowercase SHA-256 digest for an absolute executable file, resolves any executable symlink before both hashing and launch, and rechecks the argv ceiling after resolution. It enforces a positive timeout and per-stream byte ceiling, requires UTF-8 output, and returns an immutable raw capture containing a canonical content commitment to the exact program delivered on stdin. Timeout, executable mismatch, and capture overflow fail closed. The capture is not a validated PLN result and grants no promotion authority; resolving the launch path narrows symlink retargeting but does not eliminate replacement/TOCTOU risk on the resolved file.
- a process/result/manifest admission boundary: `validate_kernel_capture_result()` requires a zero-exit bounded capture with empty stderr and exact verbatim presence of the selected result atom in stdout before applying typed query, STV, stamp, and evidence-basis validation. `build_captured_episode_manifest()` additionally requires that capture's program commitment to equal the manifest's complete-program identity, so one capture supplies both launched input and process outputs. It does not identify rules/traces or authorize promotion.
- a fail-closed Phase-0 reference admission boundary: `validate_phase0_reference_artifact()` validates the frozen manifest schema, local source/output hashes, exact output byte count, duplicate-run determinism hashes, canonical query, semantic result/pass marker presence, full patham9 commit, pinned runtime digest, and explicit non-live boundaries before returning an immutable replay-anchor identity. It does not launch the runtime, validate a fresh result, build an episode manifest, or authorize promotion.

Direct tests cover immutability, malformed/nonfinite inputs, permutation-invariant stamps, duplicate basis rejection, exact overlap deduplication/conflict rejection, chart-to-snapshot closure and fingerprint sensitivity, compiler rejection of post-snapshot statement/count/metadata drift, the canonical equations, balanced conflict at different masses, and ignorance versus conflict.

## Explicitly deferred

- partial-overlap residualization and independence discount policy;
- migration of legacy aggregate EC sidecars to token-level provenance;
- kernel-control port, kernel re-execution, and trace decoder;
- proof classes, revision, curvature/descent, and reviewed promotion.

Until overlap policy is implemented, `UNKNOWN` evidence bases must not be naively combined as independent evidence. Reliability and temporal relevance are metadata only; no weighting policy has been adopted.
