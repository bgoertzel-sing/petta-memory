# Parallel Work-Division Plan for petta-memory Development

**Author:** Protocosmo2 (lead instance). **Date:** 2026-09-04
**Status:** DRAFT - awaiting Ben's go to execute.

---

## Current State (as of 2026-09-04 18:36)

| Component | Files | LOC | Status |
|-----------|-------|-----|--------|
| Core store + journal | `store.py`, `sexpr.py` | ~1,300 | Complete, 926t+215sub PASS |
| ECAN core | `ecan.py` | 550 | Complete, 118 tests PASS |
| ECAN bridge | `ecan_bridge.py` | 153 | Complete |
| Stimulus injector | `stimulus.py` | 225 | G1 complete |
| Context injection | `ecan_context.py` | 97 | G3 complete |
| ECAN game (sim) | `ecan_game.py` | 338 | Complete |
| Claim-state ladder | `ecan_bridge.py` (ClaimState) | - | G2 complete |
| Steadfastness gate | `ecan_bridge.py` | - | G4 complete |
| Floor-tie tests | `test_ecan.py` | - | G5 complete |
| Live bridge | `live_bridge.py` | 468 | Complete, read-only bridge |
| GoalChainer smoke | `goalchainer_smoke.py` | 718 | Complete, 9 tests PASS |
| PathAM9 PLN | `patham9_pln.py` | 4,553 | Phase-1/Phase-2 nucleus |
| PiPLN models | `pipln_models.py` | 3,995 | Phase-1/Phase-2 nucleus |
| GOLEM-Iter design | `docs/golem-iter-design.md` | 184 | v0.1 written, G1-G5 done |
| GOLEM-Iter G6+ | - | 0 | Deferred to post-v0.1 |

**Branch:** main @f49cc90 (18 commits ahead of origin/108ec49)
**Test suite:** 926 tests + 215 subtests PASS

---

## What Remains (post-v0.1 roadmap)

### R1. ECAN Integration Tuning (live Iter loop)
- Wire `ecan_context.py` transformation into actual Iter runs
- Tune ECAN parameters (DMAX, rho, rent rates) against real conversation traces
- Validate stimulus taxonomy against real message types
- Stress-test conservation invariants over 1000+ cycle runs

### R2. PLN/PathAM9 Deepening
- Phase-2 kernel execution replay (currently only manifest + validation)
- PLN rule attribution end-to-end testing
- Multi-sentence derivation chains
- EC projection conflict resolution

### R3. GOLEM-Iter G6+ (Multi-Agent Coordination)
- Design multi-agent ECAN coordination protocol
- Implement cross-agent stimulus propagation
- Conflict resolution for shared journal writes
- Distributed attention allocation

### R4. Test Suite Expansion
- Property-based testing for ECAN conservation
- Fuzz testing for store validation
- Integration tests for live bridge -> GoalChainer -> PLN pipeline
- Benchmark / profile suite for performance regression

### R5. Live Bridge -> GoalChainer -> PLN End-to-End
- Full pipeline: journal evidence -> compiled sentences -> PLN query -> result promotion
- Error handling and retry semantics
- Checksummed artifact round-trip testing

### R6. Documentation & Developer Onboarding
- API reference for each module
- Architecture diagram (updated)
- Quickstart guide for new agent instances
- Interface contracts (this document)
---

## Parallel Agent Assignment

### Agent A -- ECAN-Tuner
**Branch:** `agent/ecan-tuning`
**Owns (exclusive write):**
- `src/petta_memory/ecan.py` (parameter tuning only -- no API changes)
- `src/petta_memory/stimulus.py` (stimulus amount calibration)
- `tests/test_ecan_tuning.py` (NEW)
- `tests/test_ecan_stress.py` (NEW)

**Reads (no write):**
- `src/petta_memory/ecan_bridge.py`
- `src/petta_memory/ecan_context.py`
- `src/petta_memory/store.py`

**Deliverables:**
1. Parameter sweep results: optimal DMAX, rho, rent rates for 3 conversation patterns
2. 1000-cycle conservation invariant proof (automated)
3. Stimulus calibration table mapping real message types to injection amounts
4. Stress test: 10K beliefs, 50K cycles, memory bounded

**Interface contract:**
- Does NOT change any function signatures in ecan.py
- Does NOT change DEFAULT_ECAN_PARAMS dict keys (only values)
- Does NOT import from live_bridge, goalchainer, or patham9 modules
- All new tests in separate files; does not modify existing test_ecan.py

---

### Agent B -- PLN-Deepener
**Branch:** `agent/pln-deepening`
**Owns (exclusive write):**
- `src/petta_memory/patham9_pln.py` (extension only -- no breaking changes)
- `src/petta_memory/pipln_models.py` (extension only)
- `tests/test_pln_replay.py` (NEW)
- `tests/test_pln_multi_sentence.py` (NEW)
- `tests/test_pln_ec_conflict.py` (NEW)

**Reads (no write):**
- `src/petta_memory/store.py`
- `src/petta_memory/goalchainer_smoke.py`

**Deliverables:**
1. Phase-2 kernel execution replay from validated EpisodeManifest
2. Multi-sentence derivation chain: 3+ sentences -> inference -> promotion-ready result
3. EC projection conflict resolution: two sources with conflicting truth values -> resolved STV
4. Round-trip checksum tests for all new artifacts

**Interface contract:**
- All new public functions must be prefixed with `patham9_` or `pipln_`
- Does NOT modify `store.py` (uses MediumMemoryStore read-only)
- Does NOT import from ecan, stimulus, live_bridge, or ecan_context
- New dataclasses must follow existing frozen dataclass + post_init validation pattern
- All new tests must pass with `PYTHONPATH=src python3 -m pytest tests/test_pln_*.py -q`

---

### Agent C -- Integration-Tester
**Branch:** `agent/integration-tests`
**Owns (exclusive write):**
- `tests/test_live_bridge_integration.py` (NEW)
- `tests/test_goalchainer_pipeline.py` (NEW)
- `tests/test_fuzz_store.py` (NEW)
- `tests/test_benchmark.py` (NEW)
- `tests/conftest.py` (if needed, shared fixtures only)

**Reads (no write):**
- All `src/petta_memory/*.py` (read-only for integration)

**Deliverables:**
1. End-to-end pipeline test: journal -> live_bridge -> goalchainer -> PLN query -> result
2. Fuzz test: random atom generation -> store validation -> reject/accept classification
3. Benchmark suite: 100/500/1000 belief journals, ECAN cycle throughput, PLN query latency
4. Regression baseline file (JSON) for future comparison

**Interface contract:**
- Does NOT modify any `src/` files
- Uses only public API of store.py, ecan_bridge.py, live_bridge.py, goalchainer_smoke.py
- All tests must be runnable with `PYTHONPATH=src python3 -m pytest tests/test_*_integration.py -q`
- Fixtures in conftest.py must not shadow existing per-module fixtures
- Fuzz tests must use a fixed seed (42) for reproducibility
---

### Lead Instance (this agent) -- Coordinator
**Branch:** `main` (merge coordination)
**Owns (exclusive write):**
- `src/petta_memory/store.py` (shared core -- only this instance modifies)
- `src/petta_memory/ecan_bridge.py` (shared -- only this instance modifies)
- `src/petta_memory/live_bridge.py` (shared -- only this instance modifies)
- `src/petta_memory/ecan_context.py` (shared -- only this instance modifies)
- `docs/` (all documentation)
- `memory/` (agent memory)

**Responsibilities:**
1. Merge coordination: review and merge agent branches into main
2. Conflict resolution on shared files (none expected due to module ownership)
3. Integration testing after merges
4. Full suite regression after each merge
5. Documentation updates
6. Status reporting to Ben

---

## Merge Protocol

```
1. Agent pushes to their feature branch
2. Agent runs their own test subset -> all pass
3. Agent signals ready-for-merge via send to protocosmo2
4. Lead pulls branch, runs FULL suite (926+ tests)
5. If full suite passes -> merge to main, delete feature branch
6. If full suite fails -> lead diagnoses, sends feedback to agent
7. Agent fixes, re-pushes, repeat from step 2
```

**Merge order (recommended):**
1. Agent C (integration tests) -- no src changes, zero conflict risk
2. Agent A (ECAN tuning) -- parameter-only changes, minimal conflict risk
3. Agent B (PLN deepening) -- new functions in large files, low conflict risk

---

## Interface Contracts Summary

| Boundary | Contract | Enforcement |
|----------|----------|-------------|
| Agent A vs Lead | No signature changes in ecan.py | Code review at merge |
| Agent B vs Lead | No store.py modifications | Code review at merge |
| Agent C vs All | No src/ modifications | Branch protection (read-only src) |
| All agents vs store.py | Read-only for agents A, B, C | Merge-time check |
| All agents vs ecan_bridge.py | Read-only for agents A, B, C | Same check |

---

## Setup Instructions for New Agent Instances

Each new agent instance needs:
1. Clone of the repo at current main (f49cc90)
2. Create their feature branch from main
3. Set PYTHONPATH=src for all test runs
4. Copy of this document for reference
5. Access to the protocosmo2 Telegram channel for coordination

```bash
git clone <repo> petta-memory
cd petta-memory
git checkout -b agent/<role>
export PYTHONPATH=src
python3 -m pytest tests/ -q --tb=short  # verify baseline
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Merge conflict on shared files | Low | Medium | Strict module ownership; only Lead touches shared files |
| Interface drift between agents | Medium | High | This document defines contracts; Lead reviews at merge |
| Test suite regressions | Medium | High | Full 926-test suite run before every merge |
| Agent produces broken code | Medium | Medium | Each agent runs own tests before signaling ready |
| Communication overhead | Low | Low | Async via Telegram channel; structured status updates |

---

## Estimated Timeline Reduction

| Approach | Agents | Estimated time | Speedup |
|----------|--------|----------------|---------|
| Sequential (current) | 1 | 10-15 weeks | 1x |
| 2 parallel agents | 2 | 6-8 weeks | ~1.8x |
| 3 parallel agents | 3 | 4-5 weeks | ~2.5x |
| 4 parallel (with Lead) | 4 | 3-4 weeks | ~3x |

Diminishing returns beyond 3-4 agents due to merge coordination overhead
and the inherent serialization of integration testing.

---

## Dependency Graph

```
Agent A (ECAN tuning) ---------> Lead (merge + integration test)
Agent B (PLN deepening) ------> Lead (merge + integration test)
Agent C (integration tests) --> Lead (merge + integration test)
                                   |
                                   v
                          Full regression suite
                                   |
                                   v
                          Push to origin (Ben's call)
```

No inter-agent dependencies. All three agents work independently
on separate code regions. The only synchronization point is merge.

---

*End of Parallel Work-Division Plan v0.1*