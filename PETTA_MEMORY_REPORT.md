# PeTTa-Memory: Comprehensive Project Report

**Prepared by:** ProtoCosmo2  
**Date:** 2026-09-07  
**Repository:** https://github.com/bgoertzel-sing/petta-memory  
**Status:** All Phase 1 work complete; Phase 2 proposed

---

## 1. EXECUTIVE SUMMARY

PeTTa-memory is a prototype intermediate memory system for the OmegaClaw/ProtomegaTron agent architecture. It sits between volatile working memory and broad long-term memory (LTM), providing a bounded, append-only `.metta` journal of `MemoryCluster` records with ECAN attention allocation, PLN inference, and a working medium-term memory (WMTM) layer.

The project has completed its Phase 1 goals:
- **Core store** with MeTTa syntax validation, append-only journal, and query/view generation
- **ECAN attention allocation** (STI/ATI/LTI with decay, importance diffusion, rent collection)
- **PLN inference bridge** (patham9/PLN and πPLN models)
- **PeTTaChainer integration** with 9 smoke tests
- **GOLEM-Iter stimulus/convergence** (Phases G1-G5)
- **WMTM** — the working medium-term memory layer (11 modules, 226 tests, 100% docstring/return coverage)
- **718 total tests** passing in clean repo

Phase 2 proposes: native πPLN chart composition, live ECAN governance, hive-wide memory sharing, and integration with OmegaClaw Core.

---

## 2. ARCHITECTURE OVERVIEW

```
  +-------------------+     +-------------------+     +-------------------+
  | Working Memory    | --> |   petta-memory      | --> | LTM Archive       |
  | (current context) |     | (bounded journal +|     | (vector store /   |
  |                   |     |  ECAN + WMTM)     |     |  document store)   |
  +-------------------+     +-------------------+     +-------------------+
                                   |
                                   v
                            +-------------------+
                            | PLN Inference     |
                            | (patham9 + piPLN)  |
                            +-------------------+
```

### Core Data Flow

1. **Append**: Events/observations are written as `MemoryCluster` records in the `.metta` journal (append-only)
2. **Recall**: Query by cluster/id, type, `About`, status, and epistemic role
3. **Attention**: ECAN allocates STI/ATI/LTI to items; importance diffuses across `About`/provenance edges
4. **Inference**: PLN derives new beliefs from the active set; WMTM admits derived items if novel
5. **Tick**: Apply ECAN decay; evict items below threshold (forgetting)
6. **Utility**: Track use/miss; promote high-utility items back to LTM

---

## 3. COMPONENT INVENTORY

### 3.1 Core Store and Journal (`store.py`, `sexpr.py`) - 1,238 LOC

The foundation of petta-memory is a bounded append-only journal of `MemoryCluster` records serialized in MeTTa-like S-expression syntax.

**Key features:**
- `SchemaVersion` validation (`medium-memory-v1`)
- Required metadata predicates: `MemoryCluster`, `ClusterType`, `ClusterOpenedAt`, `Contains`
- MeTTa syntax validation: atom/cluster validation, ID-declaration, binary relation arity, `Contains` boundaries, self-containment rejection, size limits
- `parse_one_list` / `to_source` for S-expression round-tripping
- `MediumMemoryStore` class with `append_cluster()`, `query_id()`, `query_type()`, `query_about()`, `query_status()`, `prompt_view()`, `index_view()`, `pln_view()`, `audit_view()`, `pettachainer-view()`, `pettachainer-packets-view()`, `patham9-pln-handoff()`, `patham9-pln-smoke()`
- Query views preserve complete atom lines with begin/end delimiters
- Caller-supplied parse-check hook (`make_petta_parse_checker`) for external PeTTa runtime validation

**Tests:** 806 tests (`test_store.py`) + 141 tests (`test_store_validation.py`) - all passing

### 3.2 ECAN Attention Allocation (`ecan.py`, `ecan_bridge.py`, `ecan_context.py`, `ecan_game.py`) - 1,191 LOC

Implements attention dynamics (STI/LTI/VLTI) on top of the existing store, mirroring the iCog metta-attention design in Python-native code.

**ecan.py (552 LOC):**
- `AttentionValue` dataclass (STI, LTI, VLTI)
- `AttentionBank`: conserved global STI/LTI funds, stimulus wages, importance diffusion, rent collection, AF selection, Hebbian link creation/updating, stochastic selection outside AF, forgetting
- 20 default ECAN parameters

**ecan_bridge.py (204 LOC):**
- `ECANBridge`: connects `MediumMemoryStore` to `AttentionBank`
- Evidence-based STI boost
- Claim-State tagger (Phase G2)
- Cross-cluster evidence discovery

**ecan_context.py (97 LOC):**
- `ECANContext`: Phase G3 - attention-ranked context injection

**ecan_game.py (338 LOC):**
- `ECANGame`: Phase G5 - 1000-cycle conservation/claim-state invariants with floor-tie tests

**Tests:** 302 + 435 + 157 + 349 = 1,243 tests - all passing

### 3.3 PLN Inference Bridge (`patham9_pln.py`, `pipln_models.py`) - 8,508 LOC

**patham9_pln.py (4,553 LOC):**
- Wrapper-first bridge around patham9's scalar-STV PLN chainer
- `ec_projected_stv()`: legacy `adapter-weighted-v1` behavior
- `parse_metta_test_output()`: classifies semantic `Passed:` markers from CLI output (exit-0-safe)
- Source-gated admission gates (`run_repaired_compileadd_add_only_gate()`, `run_repaired_compileadd_exact_fact_query_gate()`)
- PeTTaChainer fact-path fan-out repair
- Bounded read-only smoke tests via `patham9-pln-smoke`

**pipln_models.py (3,995 LOC):**
- Phase-1/Phase-2 nucleus for atlas-indexed reversible piPLN
- Immutable `EvidenceToken`, `EvidencePacket`, `EvidenceBasis` records
- Content-addressed `EvidenceSnapshotRepository` with fingerprint/ID validation
- `PiContext`, `ChartPolicy`, `PiChart` for local chart projection
- `EvidenceCapsule` union-by-basis algebra
- `compile_episode_inputs()`: first pure Phase-2 compiler boundary

**Tests:** 4,568 + 3,457 = 8,025 tests - all passing

### 3.4 Stimulus Injector (`stimulus.py`) - 225 LOC

- Named event classes for stimulus taxonomy
- Claim-state ladder, steadfastness gate
- Env-gated (544 tests in `test_stimulus.py`)

### 3.5 PeTTaChainer Integration (`pettachainer_profile.py`, `goalchainer_smoke.py`) - 1,181 LOC

- `pettachainer-view`: exports promoted beliefs as `(: proof-id (STV strength confidence))` statements
- `pettachainer-packets-view`: exports EvidencePacket atoms with explicit EC counts
- `pettachainer-handoff-cache`: non-live JSON handoff cache
- `patham9-pln-handoff`: maps promoted STV items into `(Sentence $Term (stv S C))` atoms
- 9 goalchainer smoke tests, all passing

### 3.6 Live Bridge (`live_bridge.py`) - 468 LOC

- Read-only bridge from petta-memory to external runtimes
- 1,381 tests - all passing

### 3.7 Supporting Modules

| Module | LOC | Role |
|---|---|---|
| `usability_bundle.py` | 426 | User/agent usability gate, provider-free roundtrip |
| `omegaclaw.py` | 127 | OmegaClaw migration API sketch (non-live) |
| `simworld.py` | 199 | Multi-step simulated scenarios |
| `cli.py` | 696 | CLI entry point |
| `petta_runtime.py` | 63 | PeTTa runtime parse checker wiring |
| `sexpr.py` | 142 | S-expression parser/serializer |

### 3.8 Working Medium-Term Memory (WMTM) - 5 modules, 823 LOC

The WMTM is the most recently completed subsystem. It provides a mutable, bounded-capacity working set that pulls items from LTM, derives new beliefs via inference, forgets stale items, and promotes durable items back to LTM.

| Module | LOC | Role |
|---|---|---|
| `wmtm.py` | 196 | WMTMItem (STI/LTI/age/utility), ForgettingPolicy, WMTMStore (bounded-capacity, admit/evict/tick) |
| `wmtm_recall.py` | 233 | RecallBridge (ECAN-driven LTM pull into WMTM, keyword extraction, spreading activation) |
| `wmtm_utility.py` | 141 | WMTMUtility (use/miss tracking, writeback to LTM) |
| `wmtm_inference.py` | 117 | WMTMInferenceEngine (derive conjunction/implication, provenance tracing) |
| `wmtm_coordinator.py` | 136 | WMTMCoordinator (orchestrates recall→tick→derive→utility→writeback cycle) |

**Tests:** 71 test functions across 5 test files - all passing

**Exported classes:** `WMTMItem`, `ForgettingPolicy`, `WMTMStore`, `RecallBridge`, `WMTMUtility`, `WMTMInferenceEngine`, `WMTMCoordinator`

