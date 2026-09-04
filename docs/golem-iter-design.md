# GOLEM-Iter Design Document v0.1
**Status: proposal for review — nothing implemented yet.**
Author: Iter agent (ProtoCosmo2 VM). Date: 2026-09-04.
Inputs: Ben AtomSpace/Iter brainstorming (msg 7805), STI Economy + Claim-State Ladder spec v0.3 (msg 7811/7812), ECAN implementation in petta-memory, MeTTaClaw/Iter architecture review.

---

## 1. Purpose

GOLEM-Iter connects the ECAN attention allocation system (already implemented and tuned in petta-memory) to the Iter agent's goal/action selection loop. The core idea: **attention allocates context and computation, not action authority.** The LLM remains the deliberative authority; ECAN proposes what the LLM should inspect.

This directly addresses the "Next Move" failure documented in the August 19 review: STI had collapsed toward a near-uniform floor, and a derived "Next Move" field manufactured authority from an unreliable signal. GOLEM-Iter ensures this can never recur by enforcing named-stimulus-only activation, conservation invariants, and a claim-state ladder that distinguishes observed state from ranked candidates from recommended actions.

## 2. Architecture

```text
                    +-------------------------+
                    |   Iter Agent (iter.py)   |
                    |                         |
                    |  +-------------------+  |
                    |  |  Context Assembly  |  |
                    |  |  (transformations) |  |
                    |  +--------+----------+  |
                    |           |              |
                    |  +--------v----------+  |
                    |  |  LLM (deliberative) |  |
                    |  |  action authority   |  |
                    |  +--------+----------+  |
                    |           |              |
                    |  +--------v----------+  |
                    |  |  Tool Execution    |  |
                    |  +--------+----------+  |
                    |           |              |
                    |  +--------v----------+  |
                    |  |  Experience Log    |  |
                    |  +--------+----------+  |
                    +-----------+-------------+
                                |
                    +-----------v-------------+
                    |  petta-memory Journal   |
                    |  (MediumMemoryStore)   |
                    |  +-------------------+  |
                    |  |  ECANBridge        |  |
                    |  |  +---------------+ |  |
                    |  |  | AttentionBank | |  |
                    |  |  | ImportanceDiff | |  |
                    |  |  | RentCollection | |  |
                    |  |  | ECANCycle      | |  |
                    |  |  +---------------+ |  |
                    |  +-------------------+  |
                    +-------------------------+
```

**Data flow:**
1. Events (human messages, tool results, goal writes) -> stimulus injection -> ECAN cycles
2. ECAN produces attentional focus (high-STI beliefs) -> context transformation injects ranked beliefs into LLM context
3. LLM deliberates with attention-ranked context -> selects action (NOT ECAN)
4. Tool execution -> new events -> new stimuli -> next cycle

## 3. Stimulus Taxonomy (from spec v0.4)

Every stimulus is **named** and carries **provenance**. No anonymous activation.

| event class              | bundle        | inj size  | provenance required                    |
|--------------------------|---------------|-----------|----------------------------------------|
| human message received   | INTERLOCUTION | 1.00 DMAX | witnessed CONVERSATION_WINDOW IN line  |
| receipt landed (msg id)  | RECEIPT       | 0.50 DMAX | witnessed send/OUT with message_id    |
| goal verified / solved   | MILESTONE     | 0.75 DMAX | goal-drop with last-verified set       |
| formal/dry-run green     | PROOF         | 0.75 DMAX | EXIT-0 log + axiom list on disk        |
| erratum caught+logged    | ERRATUM       | 0.25 DMAX | correction note with provenance        |
| heartbeat-nop cycle      | RHYTHM        | 0.00      | sustains rhythm; injects nothing       |

DMAX = 0.30 (pinned constant, sweep-validated). Self-reported events inject at half size until externally witnessed.

## 4. Claim-State Ladder

Belief atoms carry a claim-state that can never be silently upgraded:

- **SelfReported**: agent-written; field is a claim only
- **KernelChecked**: kernel re-derives from goals.metta + compost log each wake
- **ExternallyVerified**: independent probe (git log / test run / file:line read)
- **Unavailable != Absent**: a source that cannot be read is typed unavailable, never silently zero

Rule: labels never upgrade a state; only linted evidence does. This replaces the removed "Next Move" field and the 8ff5c00 relabel patch.

## 5. Formal Invariants

### Invariant A (Bounded Decrease)
Bundles B_1..B_n with salience s_i >= 0. Injection map I(stimulus) -> {(i, delta_i)} with each delta_i in [0, DMAX].
Decay: s_i(t+1) = rho * s_i(t), rho in (0,1) pinned.

Lyapunov candidate: V(t) = sum s_i(t).
V(t+1) - V(t) <= -(1-rho)*V(t) + sum_I delta_i.
With zero injections: V(t+1) = rho*V(t) < V(t). Mass decays geometrically.

**Already verified** in petta-memory ECAN: conservation holds to <1 unit across 400 cycles.

### Invariant B (No-Exclusion at Floor)
Scalar STI may hit a rounding floor f > 0 and tie. Ranking at the floor MUST NOT use s_i.
Tiebreak key: (last_injection_time, bundle_creation_time) -- both facts of record, not claims.
A bundle with zero injection since boot promotes nothing (Steadfastness analogue).

## 6. Implementation Plan

### Phase G1: Stimulus Injector (petta-memory side)
- New module `stimulus.py` in petta-memory: classifies events into the taxonomy, injects bounded stimuli into ECANBridge.
- Gate: environment variable `ITER_PETTA_ECAN=1`, default OFF.
- No `iter.py` edit needed for this phase -- stimulus injection happens in the petta_hook already present.

### Phase G2: Claim-State Tagger (petta-memory side)
- Extend MediumMemoryStore cluster schema with optional `(ClaimState belief_id state)` atom.
- Kernel re-derivation on each `sync_from_store()` call.
- No automatic upgrade; only explicit evidence-linked writes change state.

### Phase G3: Context Injection (Iter transformation, no iter.py edit)
- New transformation `ecan_context.py` drops into `transformations/`.
- Injects a bounded (<=500 char) attention-ranked belief index into system context.
- Renders: belief_id, claim-state, STI rank, evidence count, last-stimulus event class.
- No-op when ITER_PETTA_ECAN unset or on any exception.
- **Does NOT inject a "Next Move" or recommended action.** Only ranked observations.

### Phase G4: Steadfastness Gate (petta-memory side)
- A belief with zero injection since boot has STI = 0 and is excluded from context injection.
- Tiebreak by (last_injection_time, creation_time) implemented in `get_prioritized_beliefs()`.
- Test: a journal with only RHYTHM events produces an empty attentional focus.

### Phase G5: Floor-Tie Test Suite
- Property-based test: given DMAX=0.30 and rho=0.99, verify that 1000 cycles of RHYTHM-only events produce zero beliefs above FORGET_THRESHOLD.
- Test: conservation holds across mixed stimulus patterns.
- Test: claim-state never auto-upgrades.
- Test: Unavailable sources are typed, not zeroed.

## 7. What GOLEM-Iter Does NOT Do

1. **Does not select actions.** The LLM remains the sole action authority. ECAN ranks context; the LLM decides.
2. **Does not auto-promote beliefs.** Promotion requires explicit evidence with provenance, not STI threshold crossing.
3. **Does not modify iter.py.** All integration is via the existing petta_hook and transformation system.
4. **Does not amplify without stimulus.** No spontaneous activation. Absent named stimuli, mass decays geometrically.
5. **Does not collapse to a scalar.** Each belief's context entry shows (rank, claim-state, evidence-count, last-stimulus-class) as a vector, not one number.

## 8. Relationship to Existing Work

| Component          | Status      | Role in GOLEM-Iter                        |
|-------------------|-------------|-------------------------------------------|
| ECAN (ecan.py)    | DONE        | Attention dynamics, conservation, decay   |
| ECANBridge        | DONE        | Store <-> bank sync, evidence graph        |
| ecan_game.py      | DONE        | Simulation harness for tuning/validation  |
| live_bridge.py    | DONE        | GoalChainer decision pipeline integration |
| petta_hook.py     | DRAFT       | Runtime hook (env-gated) for journal append|
| petta_context.py  | DRAFT       | Transformation for journal index injection|
| stimulus.py       | PROPOSED    | Event classification + bounded injection  |
| ecan_context.py   | PROPOSED    | Transformation for attention-ranked context|

## 9. Key Differences from the Failed "Next Move" Approach

| Property              | Failed Next Move        | GOLEM-Iter                     |
|----------------------|------------------------|--------------------------------|
| What it outputs       | Recommended action     | Ranked observations only       |
| Authority             | ECAN decides           | LLM decides; ECAN proposes     |
| Stimulus source       | Implicit/unknown        | Named, provenanced, bounded    |
| Conservation          | Not enforced            | Lyapunov-verified              |
| Claim states          | Flat (all "verified")   | Ladder (SelfReported->External) |
| Floor ties            | Arbitrary               | (injection_time, creation_time) |
| Context content       | Single scalar (STI)     | Vector (rank, state, evidence)  |

## 10. Open Questions for Ben

1. Should stimulus injection happen inside the existing petta_hook, or as a separate hook? (My recommendation: same hook, since both fire after each agent step.)
2. Should the claim-state ladder be persisted in the petta-memory journal itself (as claim-state atoms), or in a separate sidecar file? (My recommendation: journal, so it shares provenance and lifecycle.)
3. Is DMAX=0.30 the right value for the petta-memory scale (~120 beliefs), or should it scale with belief count?
4. Should the context transformation render belief text (compact) or only IDs+metadata? (My recommendation: IDs+metadata only for v0.1; the LLM can request full text via tool calls.)
5. What constitutes "externally witnessed" for a RECEIPT stimulus — is the send tool's return value sufficient, or does a human ack count?

## 11. Non-Goals (v0.1)

- No automatic goal creation from attention patterns.
- No multi-agent coordination warnings (Phase G6+, post-v0.1).
- No embedding-based retrieval (separate roadmap item).
- No modification to the existing 9 smoke tests or ECAN test suite.
- No push to origin without explicit instruction.

---

*End of GOLEM-Iter Design Document v0.1*
