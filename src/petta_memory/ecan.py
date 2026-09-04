"""ECAN (Economic Attention Allocation) for petta-memory.

Implements attention dynamics (STI/LTI/VLTI) on top of the existing
STV/evidence system, mirroring the iCog metta-attention design but
in Python-native code that operates directly on the MediumMemoryStore.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

DEFAULT_ECAN_PARAMS: dict[str, float] = {
    "STARTING_FUNDS_STI": 100_000.0,
    "STARTING_FUNDS_LTI": 100_000.0,
    "TARGET_STI": 10_000.0,
    "TARGET_LTI": 10_000.0,
    "STI_ATOM_WAGE": 10.0,
    "LTI_ATOM_WAGE": 10.0,
    "STI_FUNDS_BUFFER": 90_000.0,
    "LTI_FUNDS_BUFFER": 10_000.0,
    "MAX_AF_SIZE": 1000.0,
    "MIN_AF_STI": 100.0,
    "AFB_DECAY": 0.05,
    "AFB_BOTTOM": 50.0,
    "FORGET_THRESHOLD": 0.05,
    "NON_AF_DECAY_RATE": 0.15,
    "MAX_SPREAD_PERCENTAGE": 0.08,
    "DIFFUSION_TOURNAMENT_SIZE": 5.0,
    "RENT_TOURNAMENT_SIZE": 5.0,
    "STI_RENT_RATE": 0.01,
    "LTI_RENT_RATE": 0.005,
    "AFRentFrequency": 5.0,
    "HEBBIAN_MAX_ALLOCATION_PERCENTAGE": 0.05,
}


@dataclass(frozen=True)
class AttentionValue:
    """STI / LTI / VLTI triple attached to each atom."""
    sti: float = 0.0
    lti: float = 0.0
    vlti: int = 0

    def with_sti(self, new_sti: float) -> "AttentionValue":
        return AttentionValue(sti=new_sti, lti=self.lti, vlti=self.vlti)

    def with_lti(self, new_lti: float) -> "AttentionValue":
        return AttentionValue(sti=self.sti, lti=new_lti, vlti=self.vlti)

    def with_av(self, sti: float, lti: float, vlti: Optional[int] = None) -> "AttentionValue":
        return AttentionValue(sti=sti, lti=lti, vlti=self.vlti if vlti is None else vlti)

    def as_tuple(self) -> tuple[float, float, int]:
        return (self.sti, self.lti, self.vlti)

    @classmethod
    def default(cls) -> "AttentionValue":
        return cls()


_GROUP_SIZE = 8.0
_GROUP_NUM = 12.0


def importance_bin(sti: float) -> int:
    """Compute bin index for a given STI value."""
    impo_int = int(sti)
    if impo_int < 0:
        return 0
    if impo_int < 2 * _GROUP_SIZE:
        return impo_int
    imp = int((impo_int - _GROUP_SIZE) / _GROUP_SIZE)
    if imp <= 0:
        i = 0
    else:
        log_val = math.ceil(math.log2(imp + 1))
        i = min(log_val, int(_GROUP_NUM))
    if i > 0:
        temp = math.ceil(impo_int / (2.0 ** (i - 1)))
    else:
        temp = float(impo_int)
    ad = _GROUP_SIZE - temp
    return max(0, int(i * _GROUP_SIZE - ad))


class AttentionBank:
    """Global attention economy: funds, attentional focus, importance index."""

    def __init__(self, params: Optional[dict[str, float]] = None):
        self.params = dict(DEFAULT_ECAN_PARAMS)
        if params:
            self.params.update(params)
        self._funds_sti: float = self.params["STARTING_FUNDS_STI"]
        self._funds_lti: float = self.params["STARTING_FUNDS_LTI"]
        self._min_af_sti: float = self.params["MIN_AF_STI"]
        self._max_sti_seen: float = 0.0
        self._min_sti_seen: float = 0.0
        self._av: dict[str, AttentionValue] = {}
        self._af: set[str] = set()
        self._bins: dict[int, set[str]] = defaultdict(set)

    def get_param(self, name: str) -> float:
        return self.params[name]

    def set_param(self, name: str, value: float) -> None:
        self.params[name] = value

    @property
    def funds_sti(self) -> float:
        return self._funds_sti

    @property
    def funds_lti(self) -> float:
        return self._funds_lti

    @property
    def min_af_sti(self) -> float:
        return self._min_af_sti

    @property
    def af_size(self) -> int:
        return len(self._af)

    @property
    def num_atoms(self) -> int:
        return len(self._av)

    def get_av(self, atom_id: str) -> AttentionValue:
        return self._av.get(atom_id, AttentionValue.default())

    def get_sti(self, atom_id: str) -> float:
        return self.get_av(atom_id).sti

    def get_lti(self, atom_id: str) -> float:
        return self.get_av(atom_id).lti

    def get_vlti(self, atom_id: str) -> int:
        return self.get_av(atom_id).vlti

    def set_av(self, atom_id: str, av: AttentionValue) -> None:
        """Set attention value, updating funds, AF, and bins."""
        old = self._av.get(atom_id)
        old_sti = old.sti if old else 0.0
        old_lti = old.lti if old else 0.0
        self._funds_sti += old_sti - av.sti
        self._funds_lti += old_lti - av.lti
        self._av[atom_id] = av
        old_bin = importance_bin(old_sti)
        new_bin = importance_bin(av.sti)
        if old_bin != new_bin:
            self._bins[old_bin].discard(atom_id)
            if not self._bins[old_bin]:
                del self._bins[old_bin]
        self._bins[new_bin].add(atom_id)
        if av.sti > self._max_sti_seen:
            self._max_sti_seen = av.sti
        if av.sti < self._min_sti_seen:
            self._min_sti_seen = av.sti
        self._update_af(atom_id, av.sti)

    def apply_non_af_decay(self) -> None:
        """Apply percentage decay to non-AF atoms each cycle.

        Atoms outside the attentional focus gradually lose STI,
        eventually becoming forget candidates.
        """
        decay_rate = self.params["NON_AF_DECAY_RATE"]
        for atom_id, av in list(self._av.items()):
            if atom_id not in self._af:
                new_sti = av.sti * (1.0 - decay_rate)
                self.set_av(atom_id, av.with_sti(new_sti))

    def _update_af(self, atom_id: str, sti: float) -> None:
        """Add/remove atom from attentional focus based on STI threshold."""
        max_af = int(self.params["MAX_AF_SIZE"])
        if sti >= self._min_af_sti:
            if atom_id not in self._af:
                if len(self._af) < max_af:
                    self._af.add(atom_id)
                else:
                    af_atoms = [(a, self.get_sti(a)) for a in self._af]
                    af_atoms.sort(key=lambda x: x[1])
                    if af_atoms and sti > af_atoms[0][1]:
                        evicted = af_atoms[0][0]
                        self._af.discard(evicted)
                        self._af.add(atom_id)
                        remaining = [self.get_sti(a) for a in self._af]
                        self._min_af_sti = min(remaining) if remaining else self.params["MIN_AF_STI"]
        else:
            if atom_id in self._af:
                self._af.discard(atom_id)
                if self._af:
                    self._min_af_sti = min(self.get_sti(a) for a in self._af)
                else:
                    self._min_af_sti = self.params["MIN_AF_STI"]

    def get_af_atoms(self) -> list[str]:
        """Return atoms in attentional focus, sorted by STI descending."""
        af_list = [(a, self.get_sti(a)) for a in self._af]
        af_list.sort(key=lambda x: -x[1])
        return [a for a, _ in af_list]

    def is_in_af(self, atom_id: str) -> bool:
        return atom_id in self._af

    def get_atoms_in_sti_range(self, lower: float, upper: float) -> list[str]:
        """Return atoms whose STI falls within [lower, upper] using bin index."""
        lower_bin = importance_bin(lower)
        upper_bin = importance_bin(upper)
        result: list[str] = []
        for b in range(lower_bin, upper_bin + 1):
            for atom_id in self._bins.get(b, set()):
                sti = self.get_sti(atom_id)
                if lower <= sti <= upper:
                    result.append(atom_id)
        return result

    def get_random_atom_in_af(self) -> Optional[str]:
        af = list(self._af)
        return random.choice(af) if af else None

    def get_random_atom_not_in_af(self) -> Optional[str]:
        non_af = [a for a in self._av if a not in self._af]
        return random.choice(non_af) if non_af else None

    def calculate_sti_wage(self) -> float:
        """Calculate STI wage based on fund equilibrium."""
        funds = self._funds_sti
        target = self.params["TARGET_STI"]
        buffer = self.params["STI_FUNDS_BUFFER"]
        base_wage = self.params["STI_ATOM_WAGE"]
        diff = funds - target
        ndiff_raw = diff / buffer
        ndiff = max(-1.0, min(1.0, ndiff_raw))
        return base_wage + base_wage * ndiff

    def calculate_lti_wage(self) -> float:
        """Calculate LTI wage based on fund equilibrium."""
        funds = self._funds_lti
        target = self.params["TARGET_LTI"]
        buffer = self.params["LTI_FUNDS_BUFFER"]
        base_wage = self.params["LTI_ATOM_WAGE"]
        diff = funds - target
        ndiff_raw = diff / buffer
        ndiff = max(-1.0, min(1.0, ndiff_raw))
        return base_wage + base_wage * ndiff

    def stimulate(self, atom_id: str, stimulus: float) -> None:
        """Adjust STI and LTI of an atom based on wages and stimulus."""
        av = self.get_av(atom_id)
        sti_wage = self.calculate_sti_wage() * stimulus
        lti_wage = self.calculate_lti_wage() * stimulus
        new_sti = av.sti + sti_wage
        new_lti = av.lti + lti_wage
        self.set_av(atom_id, av.with_av(new_sti, new_lti))

    def calculate_sti_rent(self, base_rent: float = 1.0) -> float:
        """Calculate STI rent rate based on fund equilibrium.

        Returns a multiplier applied to each atom's STI to determine rent.
        When funds are above target, rent is low (atoms keep more STI).
        When funds are below target, rent increases (drains atoms to refill funds).
        Always returns > 0 to ensure dominant atoms are gradually drained.
        """
        funds = self._funds_sti
        target = self.params["TARGET_STI"]
        buffer = self.params["STI_FUNDS_BUFFER"]
        diff = target - funds
        ndiff = diff / buffer
        ndiff = max(-0.99, min(1.0, ndiff))
        return base_rent + base_rent * ndiff

    def calculate_lti_rent(self, base_rent: float = 1.0) -> float:
        """Calculate LTI rent rate based on fund equilibrium.

        Returns a multiplier for LTI rent. Always positive.
        """
        funds = self._funds_lti
        target = self.params["TARGET_LTI"]
        buffer = self.params["LTI_FUNDS_BUFFER"]
        diff = target - funds
        ndiff = diff / buffer
        ndiff = max(-1.0, min(1.0, ndiff))
        return base_rent + base_rent * ndiff


# ---------------------------------------------------------------------------
# ImportanceDiffusion
# ---------------------------------------------------------------------------


class ImportanceDiffusion:
    """Spread STI from high-importance atoms to their evidence-linked neighbors.

    Mirrors ImportanceDiffusionAgent from metta-attention. In petta-memory,
    the evidence graph (EvidenceFor links) serves as the diffusion channel:
    when a belief has high STI, a fraction of its STI spreads to the beliefs
    it provides evidence for, and vice versa.
    """

    def __init__(self, bank: AttentionBank):
        self.bank = bank
        # adjacency: atom_id -> set of neighbor atom_ids (via evidence links)
        self._evidence_links: dict[str, set[str]] = defaultdict(set)

    def add_evidence_link(self, source: str, target: str) -> None:
        """Add a directed evidence link: source provides evidence for target."""
        self._evidence_links[source].add(target)

    def add_bidirectional_link(self, a: str, b: str) -> None:
        """Add bidirectional evidence link between two atoms."""
        self._evidence_links[a].add(b)
        self._evidence_links[b].add(a)

    def build_from_evidence_map(self, evidence_map: dict[str, list[str]]) -> None:
        """Build diffusion graph from an evidence map.

        Args:
            evidence_map: {belief_id: [evidence_source_ids]}
        """
        for belief_id, sources in evidence_map.items():
            for src in sources:
                self.add_bidirectional_link(src, belief_id)

    def diffuse_atom(self, atom_id: str, af_mode: bool = True) -> list[tuple[str, float]]:
        """Diffuse STI from one atom to its neighbors.

        Returns list of (neighbor_id, sti_received) tuples.
        """
        neighbors = self._evidence_links.get(atom_id, set())
        if not neighbors:
            return []

        source_sti = self.bank.get_sti(atom_id)
        max_spread = self.bank.get_param("MAX_SPREAD_PERCENTAGE")
        total_diffusion = source_sti * max_spread

        if total_diffusion <= 0:
            return []

        # Probability vector: normalize by neighbor STI (heavier neighbors get more)
        neighbor_stis = [(n, max(1.0, self.bank.get_sti(n))) for n in neighbors]
        total_weight = sum(s for _, s in neighbor_stis)

        results: list[tuple[str, float]] = []
        for neighbor_id, weight in neighbor_stis:
            share = total_diffusion * (weight / total_weight)
            if share > 0:
                av = self.bank.get_av(neighbor_id)
                self.bank.set_av(neighbor_id, av.with_sti(av.sti + share))
                results.append((neighbor_id, share))

        # Deduct diffused STI from source
        av = self.bank.get_av(atom_id)
        self.bank.set_av(atom_id, av.with_sti(av.sti - total_diffusion))

        return results

    def diffuse_af(self) -> dict[str, list[tuple[str, float]]]:
        """Diffuse STI from all atoms in the attentional focus.

        Returns {atom_id: [(neighbor, sti_received), ...]} for each diffused atom.
        """
        results: dict[str, list[tuple[str, float]]] = {}
        tournament_size = int(self.bank.get_param("DIFFUSION_TOURNAMENT_SIZE"))
        af_atoms = self.bank.get_af_atoms()

        # Tournament selection: pick subset of AF atoms to diffuse
        if len(af_atoms) > tournament_size:
            # Select top tournament_size by STI
            selected = af_atoms[:tournament_size]
        else:
            selected = af_atoms

        for atom_id in selected:
            results[atom_id] = self.diffuse_atom(atom_id, af_mode=True)

        return results


# ---------------------------------------------------------------------------
# RentCollection
# ---------------------------------------------------------------------------


class RentCollection:
    """Charge STI/LTI rent from atoms in attentional focus.

    Atoms that fall below FORGET_THRESHOLD after rent collection become
    forgetting candidates. Rent is redistributed to global funds.
    """

    def __init__(self, bank: AttentionBank):
        self.bank = bank
        self._forget_candidates: list[str] = []

    @property
    def forget_candidates(self) -> list[str]:
        return self._forget_candidates

    def collect_rent(self) -> dict[str, dict[str, float]]:
        """Collect rent from all AF atoms.

        Returns {atom_id: {sti_rent, lti_rent, new_sti, new_lti}}.
        """
        self._forget_candidates = []
        results: dict[str, dict[str, float]] = {}
        af_atoms = self.bank.get_af_atoms()
        rent_freq = self.bank.get_param("AFRentFrequency")
        forget_threshold = self.bank.get_param("FORGET_THRESHOLD")

        rent_multiplier = self.bank.calculate_sti_rent()
        lti_rent_multiplier = self.bank.calculate_lti_rent()
        sti_rate = self.bank.get_param("STI_RENT_RATE")
        lti_rate = self.bank.get_param("LTI_RENT_RATE")

        for atom_id in af_atoms:
            av = self.bank.get_av(atom_id)
            # Proportional rent: percentage of atom's own STI/LTI
            # Multiplied by equilibrium factor (higher when funds low)
            sti_rent = av.sti * sti_rate * rent_multiplier
            lti_rent = av.lti * lti_rate * lti_rent_multiplier

            new_sti = av.sti - sti_rent
            new_lti = av.lti - lti_rent

            self.bank.set_av(atom_id, av.with_av(new_sti, new_lti))

            results[atom_id] = {
                "sti_rent": sti_rent,
                "lti_rent": lti_rent,
                "new_sti": new_sti,
                "new_lti": new_lti,
            }

            # Check for forgetting candidacy
            if new_sti < forget_threshold:
                self._forget_candidates.append(atom_id)

        # Also check all non-AF atoms for forget candidacy
        for atom_id, av in self.bank._av.items():
            if atom_id not in self._forget_candidates and av.sti < forget_threshold:
                self._forget_candidates.append(atom_id)

        return results


# ---------------------------------------------------------------------------
# ECAN Cycle
# ---------------------------------------------------------------------------


@dataclass
class ECANCycleResult:
    """Result of one ECAN cycle."""
    cycle: int
    stimulated: int
    diffused: dict[str, list[tuple[str, float]]]
    rent_collected: dict[str, dict[str, float]]
    forget_candidates: list[str]
    af_size: int
    funds_sti: float
    funds_lti: float


class ECANCycle:
    """Orchestrates the full ECAN attention cycle.

    Cycle: stimulate -> diffuse -> collect_rent -> update_AF
    """

    def __init__(
        self,
        bank: AttentionBank,
        diffusion: ImportanceDiffusion,
        rent: RentCollection,
    ):
        self.bank = bank
        self.diffusion = diffusion
        self.rent = rent
        self._cycle_count = 0

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def step(
        self,
        stimuli: Optional[dict[str, float]] = None,
    ) -> ECANCycleResult:
        """Execute one ECAN cycle.

        Args:
            stimuli: {atom_id: stimulus_amount} for this cycle.
                     If None, no stimulation is applied.

        Returns:
            ECANCycleResult with details of this cycle.
        """
        self._cycle_count += 1

        # 1. Stimulate
        stimulated = 0
        if stimuli:
            for atom_id, stimulus in stimuli.items():
                self.bank.stimulate(atom_id, stimulus)
                stimulated += 1

        # 2. Diffuse
        diffused = self.diffusion.diffuse_af()

        # 3. Decay non-AF atoms
        self.bank.apply_non_af_decay()

        # 4. Collect rent
        rent_results = self.rent.collect_rent()

        # 4. Build result
        result = ECANCycleResult(
            cycle=self._cycle_count,
            stimulated=stimulated,
            diffused=diffused,
            rent_collected=rent_results,
            forget_candidates=list(self.rent.forget_candidates),
            af_size=self.bank.af_size,
            funds_sti=self.bank.funds_sti,
            funds_lti=self.bank.funds_lti,
        )

        return result

    def run(self, num_cycles: int, stimuli_fn=None) -> list[ECANCycleResult]:
        """Run multiple ECAN cycles.

        Args:
            num_cycles: number of cycles to run.
            stimuli_fn: optional callable(cycle_num) -> dict[str, float].

        Returns:
            List of ECANCycleResult for each cycle.
        """
        results: list[ECANCycleResult] = []
        for i in range(num_cycles):
            stimuli = stimuli_fn(i) if stimuli_fn else None
            results.append(self.step(stimuli))
        return results
