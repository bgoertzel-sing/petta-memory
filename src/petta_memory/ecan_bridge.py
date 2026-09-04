"""ECAN bridge: connects petta-memory store beliefs to ECAN attention dynamics.

Builds the evidence-link graph from store clusters, feeds it into
ImportanceDiffusion, and provides attention-prioritizedized views of beliefs
for the goalchainer and live_bridge pipelines.
"""

from __future__ import annotations

from typing import Optional

from .ecan import AttentionBank, AttentionValue, ECANCycle, ImportanceDiffusion, RentCollection
from .store import MediumMemoryStore, _second_objects_for_subject, _objects_for_predicate


class ECANBridge:
    """Bridge between MediumMemoryStore and the ECAN attention system.

    Extracts belief IDs and evidence links from the journal, populates
    the AttentionBank, builds the diffusion graph, and runs ECAN cycles.
    High-STI beliefs can then be prioritized for goalchainer/live_bridge.
    """

    def __init__(
        self,
        store: MediumMemoryStore,
        params: Optional[dict[str, float]] = None,
    ):
        self.store = store
        self.bank = AttentionBank(params)
        self.diffusion = ImportanceDiffusion(self.bank)
        self.rent = RentCollection(self.bank)
        self.cycle = ECANCycle(self.bank, self.diffusion, self.rent)
        self._belief_ids: set[str] = set()
        self._evidence_map: dict[str, list[str]] = {}
        self._claim_states: dict[str, str] = {}

    def sync_from_store(self) -> dict[str, int]:
        """Extract beliefs and evidence links from the store journal.

        Returns summary: {beliefs, evidence_links, new_atoms}.
        """
        self._belief_ids.clear()
        self._evidence_map.clear()
        self._claim_states.clear()

        for cluster in self.store.clusters():
            atoms = cluster.atoms
            # Extract DerivedBelief IDs
            for belief_id in _objects_for_predicate(atoms, "DerivedBelief"):
                self._belief_ids.add(belief_id)
                # Register atom in bank if not already present
                if self.bank.get_sti(belief_id) == 0.0 and self.bank.get_lti(belief_id) == 0.0:
                    self.bank.set_av(belief_id, AttentionValue(sti=0, lti=0, vlti=0))

                # Extract claim state: (ClaimState belief_id state_value)
                state_values = _second_objects_for_subject(atoms, "ClaimState", belief_id)
                if state_values:
                    # Last write wins (append-only journal; latest state is authoritative)
                    self._claim_states[belief_id] = state_values[-1]

                # Extract evidence links: EvidenceFor belief_id source_id
                evidence_sources = _second_objects_for_subject(atoms, "EvidenceFor", belief_id)
                if evidence_sources:
                    self._evidence_map[belief_id] = evidence_sources
                    # Register evidence source atoms in bank
                    for src in evidence_sources:
                        if self.bank.get_sti(src) == 0.0 and self.bank.get_lti(src) == 0.0:
                            self.bank.set_av(src, AttentionValue(sti=0, lti=0, vlti=0))

        # Build diffusion graph from evidence map
        self.diffusion.build_from_evidence_map(self._evidence_map)

        return {
            "beliefs": len(self._belief_ids),
            "evidence_links": sum(len(v) for v in self._evidence_map.values()),
            "new_atoms": self.bank.num_atoms,
        }

    def stimulate_beliefs(self, stimuli: dict[str, float]) -> None:
        """Apply STI stimulus to specific belief IDs."""
        for belief_id, amount in stimuli.items():
            if belief_id in self._belief_ids:
                self.bank.stimulate(belief_id, amount)

    def run_cycle(self, stimuli: Optional[dict[str, float]] = None) -> object:
        """Run one ECAN cycle with optional stimuli.

        Returns ECANCycleResult.
        """
        return self.cycle.step(stimuli=stimuli)

    def run_cycles(self, num_cycles: int, stimuli_fn=None) -> list[object]:
        """Run multiple ECAN cycles."""
        return self.cycle.run(num_cycles, stimuli_fn=stimuli_fn)

    def get_prioritized_beliefs(self, limit: int = 20) -> list[tuple[str, float]]:
        """Return beliefs sorted by STI descending (attention-prioritized).

        Args:
            limit: maximum number of beliefs to return.

        Returns:
            List of (belief_id, sti) tuples, highest STI first.
        """
        scored = [
            (bid, self.bank.get_sti(bid))
            for bid in self._belief_ids
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def get_attentional_focus_beliefs(self) -> list[str]:
        """Return belief IDs currently in the attentional focus."""
        af = set(self.bank.get_af_atoms())
        return [bid for bid in self._belief_ids if bid in af]

    def get_forget_candidates(self) -> list[str]:
        """Return belief IDs that are forget candidates (below threshold)."""
        return list(self.rent.forget_candidates)

    def get_evidence_map(self) -> dict[str, list[str]]:
        """Return the evidence link map {belief_id: [source_ids]}."""
        return dict(self._evidence_map)

    def get_claim_states(self) -> dict[str, str]:
        """Return the claim-state map {belief_id: state_string}.

        State values: 'SelfReported', 'KernelChecked', 'ExternallyVerified', 'Unavailable'.
        Beliefs without an explicit ClaimState atom default to 'SelfReported'.
        """
        result = {}
        for bid in self._belief_ids:
            result[bid] = self._claim_states.get(bid, "SelfReported")
        return result

    def get_claim_state(self, belief_id: str) -> str:
        """Return the claim state for a single belief, or 'SelfReported' if unset."""
        return self._claim_states.get(belief_id, "SelfReported")

    def summary(self) -> dict[str, object]:
        """Return a summary of the ECAN bridge state."""
        return {
            "total_beliefs": len(self._belief_ids),
            "total_atoms": self.bank.num_atoms,
            "af_size": self.bank.af_size,
            "funds_sti": self.bank.funds_sti,
            "funds_lti": self.bank.funds_lti,
            "cycle_count": self.cycle.cycle_count,
            "evidence_links": sum(len(v) for v in self._evidence_map.values()),
            "min_af_sti": self.bank.min_af_sti,
            "claim_states": dict(self._claim_states),
        }
