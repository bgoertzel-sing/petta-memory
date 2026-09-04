# ECAN-in-petta Integration Design

## Goal
Integrate iCog metta-attention ECAN (Economic Attention Allocation) into petta-memory,
adding attention dynamics (STI/LTI/VLTI) on top of existing STV/evidence system.

## Current petta-memory Architecture
- **Journal**: append-only .metta clusters with MemoryCluster, BeliefContent, TruthValue (STV s c)
- **Store**: MediumMemoryStore with resolved_view(), clusters(), promotion gates
- **Evidence**: EvidenceFor/EvidenceSupportCount/EvidenceOppositionCount per belief
- **SalienceValue**: static scoring for belief ranking
- **Goalchainer**: smoke tests + live_bridge (patham9 PLN integration)
- **SimWorld**: scenario tests for evidence-driven goal dynamics

## ECAN Components to Integrate (from metta-attention)

### 1. AttentionValue (AV)
- STI (Short-Term Importance): decays quickly, drives attentional focus
- LTI (Long-Term Importance): decays slowly, drives survival/forgetting
- VLTI (Very Long-Term Importance): binary flag for permanent retention
- Added to each belief/fact atom alongside existing STV

### 2. AttentionBank
- Global STI/LTI funds (economic model: fixed budget, redistributed)
- Attentional Focus (AF): top-N atoms by STI above MIN_AF_STI threshold
- Importance Index: bin-based O(1) retrieval by STI range

### 3. ImportanceDiffusionAgent
- Spreads STI from high-importance atoms to neighbors via evidence links
- Incident atoms (EvidenceFor targets) receive diffused STI
- Hebbian links (ASYMMETRIC_HEBBIAN_LINK) for learned associations
- Two modes: AF diffusion (from attentional focus) and WA diffusion (whole atomspace)

### 4. RentCollectionAgent
- Charges STI/LTI rent from atoms in attentional focus
- Atoms below FORGET_THRESHOLD become candidates for forgetting
- Rent redistributed to global funds (economic cycle)

### 5. AttentionParameters
- FUNDS_STI/LTI: global economy budget (starts 100000 each)
- TARGET_STI/LTI: equilibrium target (10000 each)
- STI_ATOM_WAGE: per-cycle stimulus (10)
- MAX_AF_SIZE: attentional focus cap (1000)
- MIN_AF_STI: AF admission threshold (100000 initially, updates dynamically)
- AFB_DECAY: attentional focus boundary decay (0.05)
- FORGET_THRESHOLD: removal cutoff (0.05)

## Integration Plan

### Phase 1: AttentionValue Data Layer
- Add AV (sti, lti, vlti) to resolved belief atoms in journal
- Extend MediumMemoryStore to track AV per atom
- Backwards-compatible: atoms without AV get default (STI=0, LTI=0, VLTI=0)

### Phase 2: AttentionBank Core
- Implement AttentionBank class in Python (mirrors metta-attention bank.metta)
- Global funds, AF management, importance index binning
- stimulate(atom, stimulus): wage-based STI/LTI adjustment
- get_af_atoms(): return current attentional focus set

### Phase 3: Importance Diffusion
- Diffuse STI across EvidenceFor links (petta-memory's evidence graph)
- Probability vector: normalized STI of incident/neighbor atoms
- Diffusion amount: percentage of source STI (MAX_SPREAD_PERCENTAGE=0.4)
- Tournament selection (DIFFUSION_TOURNAMENT_SIZE=5) for efficiency

### Phase 4: Rent Collection
- Per-cycle rent from AF atoms (calculateStiRent/calculateLtiRent)
- Rent scales with fund deficit (economic equilibrium seeking)
- Atoms below FORGET_THRESHOLD → forgetting candidates

### Phase 5: ECAN Cycle Integration
- Wire ECAN cycle into petta_runtime: stimulate → diffuse → collect_rent → update_AF
- Bridge with goalchainer: high-STI beliefs get priority for goal processing
- Bridge with live_bridge: patham9 receives high-STI beliefs as PLN input

### Phase 6: Testing
- Unit tests for each ECAN component
- SimWorld scenarios: attention-driven evidence prioritization
- Integration test: full ECAN cycle on journal beliefs

## Key Design Decisions
1. **Python-native**: ECAN logic in Python, not metta — petta-memory store is Python
2. **Evidence links = diffusion channels**: EvidenceFor links serve as Hebbian-like connections
3. **STV remains primary truth**: AV modulates priority/access, doesn't override truth values
4. **Bounded**: MAX_AF_SIZE caps working set; importance index for O(1) bin retrieval
5. **Economic**: Fixed STI/LTI funds redistributed via wages/rent — self-regulating

## Parameters (initial, to be tuned)
```python
ECAN_PARAMS = {
    "STARTING_FUNDS_STI": 100000,
    "STARTING_FUNDS_LTI": 100000,
    "TARGET_STI": 10000,
    "TARGET_LTI": 10000,
    "STI_ATOM_WAGE": 10,
    "LTI_ATOM_WAGE": 10,
    "STI_FUNDS_BUFFER": 10000,
    "LTI_FUNDS_BUFFER": 10000,
    "MAX_AF_SIZE": 1000,
    "MIN_AF_STI": 100,  # lowered from metta default for petta scale
    "AFB_DECAY": 0.05,
    "AFB_BOTTOM": 50.0,
    "FORGET_THRESHOLD": 0.05,
    "MAX_SPREAD_PERCENTAGE": 0.4,
    "DIFFUSION_TOURNAMENT_SIZE": 5,
    "RENT_TOURNAMENT_SIZE": 5,
    "AFRentFrequency": 5.0,
}
```
