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
- immutable create-once JSON snapshot documents with a canonical payload checksum and fail-closed loader; snapshot records require at least one canonical packet ID and a lowercase SHA-256 fingerprint, including after loading a separately checksummed document;
- a content-addressed `EvidenceSnapshotRepository` that discovers validated snapshot documents, rejects filename/fingerprint drift and duplicate logical snapshot IDs, and provides fail-closed lookup;
- stable packet provenance digests;
- deterministic, collision-free episode stamp assignment sorted by stable basis ID;
- canonical `pipl-local-chart-v1` projection with separate evidence mass, conflict balance, signed tendency, STV, and beta parameters;
- exact `EvidenceCapsule` union-by-basis algebra that deduplicates identical shared contributions and fails closed on conflicting weights for the same basis ID.

Direct tests cover immutability, malformed/nonfinite inputs, permutation-invariant stamps, duplicate basis rejection, exact overlap deduplication/conflict rejection, the canonical equations, balanced conflict at different masses, and ignorance versus conflict.

## Explicitly deferred

- packet-to-basis construction, partial-overlap residualization, and independence discount policy;
- migration of legacy aggregate EC sidecars to token-level provenance;
- episode compiler, kernel-control port, replay manifests, and trace decoder;
- proof classes, revision, curvature/descent, and reviewed promotion.

Until overlap policy is implemented, `UNKNOWN` evidence bases must not be naively combined as independent evidence. Reliability and temporal relevance are metadata only; no weighting policy has been adopted.
