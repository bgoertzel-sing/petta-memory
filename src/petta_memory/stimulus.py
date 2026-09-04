"""GOLEM-Iter stimulus injector: classifies agent events into named,
provenance-bearing stimuli and feeds them to ECANBridge.

Design (GOLEM-Iter v0.1):
  - Every stimulus is NAMED (event class from the taxonomy).
  - Every stimulus carries PROVENANCE (witness record).
  - Injection amounts are bounded by DMAX.
  - Self-reported events inject at half size until externally witnessed.
  - RHYTHM events inject zero (sustain rhythm, no amplification).

Activation is gated by environment variable ITER_PETTA_ECAN=1 (default OFF).
When unset, all methods are no-ops.
"""

from __future__ import annotations

import enum
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .ecan_bridge import ECANBridge

DMAX: float = 0.30
"""Pinned maximum stimulus injection. Sweep-validated in ECAN tuning."""

SELF_REPORT_FACTOR: float = 0.50

_ENV_GATE = "ITER_PETTA_ECAN"


def _is_enabled() -> bool:
    return os.environ.get(_ENV_GATE, "").strip() in ("1", "true", "True", "yes", "on")


class EventClass(enum.Enum):
    INTERLOCUTION = "INTERLOCUTION"
    RECEIPT = "RECEIPT"
    MILESTONE = "MILESTONE"
    PROOF = "PROOF"
    ERRATUM = "ERRATUM"
    RHYTHM = "RHYTHM"


_INJECTION_TABLE: dict[EventClass, float] = {
    EventClass.INTERLOCUTION: 1.00,
    EventClass.RECEIPT: 0.50,
    EventClass.MILESTONE: 0.75,
    EventClass.PROOF: 0.75,
    EventClass.ERRATUM: 0.25,
    EventClass.RHYTHM: 0.00,
}


@dataclass(frozen=True)
class Provenance:
    """Witness record for a stimulus event."""
    witness: str
    evidence_ref: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class StimulusEvent:
    """A single classified stimulus event ready for injection."""
    event_class: EventClass
    target_belief_ids: tuple[str, ...]
    provenance: Provenance
    externally_witnessed: bool = False

    def injection_amount(self) -> float:
        base = DMAX * _INJECTION_TABLE.get(self.event_class, 0.0)
        if not self.externally_witnessed:
            base *= SELF_REPORT_FACTOR
        return base


class ClaimState(enum.Enum):
    SELF_REPORTED = "SelfReported"
    KERNEL_CHECKED = "KernelChecked"
    EXTERNALLY_VERIFIED = "ExternallyVerified"
    UNAVAILABLE = "Unavailable"


_LADDER_ORDER = {
    ClaimState.UNAVAILABLE: 0,
    ClaimState.SELF_REPORTED: 1,
    ClaimState.KERNEL_CHECKED: 2,
    ClaimState.EXTERNALLY_VERIFIED: 3,
}


@dataclass
class _SteadfastnessRecord:
    """Per-belief tracking for steadfastness gate and floor-tie resolution."""
    last_injection_time: float = 0.0
    last_event_class: Optional[EventClass] = None
    creation_time: float = field(default_factory=time.time)
    total_injections: int = 0
    claim_state: ClaimState = ClaimState.SELF_REPORTED


class StimulusInjector:
    """Classifies events, applies bounded stimuli to ECANBridge, and tracks
    per-belief steadfastness metadata.

    All methods are no-ops when ITER_PETTA_ECAN is unset.
    """

    def __init__(self, bridge: ECANBridge):
        self._bridge = bridge
        self._enabled = _is_enabled()
        self._records: dict[str, _SteadfastnessRecord] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def inject(self, event: StimulusEvent) -> dict[str, float]:
        """Inject a stimulus event into the ECAN bridge.

        Returns:
            {belief_id: amount_injected} for each belief that received stimulus.
            Empty dict when disabled or RHYTHM event.
        """
        if not self._enabled:
            return {}

        amount = event.injection_amount()
        if amount <= 0.0:
            for bid in event.target_belief_ids:
                rec = self._get_or_create(bid)
                rec.last_event_class = event.event_class
            return {}

        injected: dict[str, float] = {}
        now = time.time()
        for bid in event.target_belief_ids:
            if bid in self._bridge._belief_ids:
                self._bridge.stimulate_beliefs({bid: amount})
                injected[bid] = amount
                rec = self._get_or_create(bid)
                rec.last_injection_time = now
                rec.last_event_class = event.event_class
                rec.total_injections += 1
        return injected

    def run_cycle_with_stimulus(self, event: StimulusEvent) -> Optional[object]:
        """Inject a stimulus and run one ECAN cycle.

        Returns ECANCycleResult when enabled, None when disabled.
        """
        if not self._enabled:
            return None
        self.inject(event)
        return self._bridge.run_cycle()

    def _get_or_create(self, belief_id: str) -> _SteadfastnessRecord:
        if belief_id not in self._records:
            self._records[belief_id] = _SteadfastnessRecord()
        return self._records[belief_id]

    def get_steadfastness(self, belief_id: str) -> Optional[_SteadfastnessRecord]:
        return self._records.get(belief_id)

    def prioritized_with_steadfastness(
        self, limit: int = 20,
    ) -> list[tuple[str, float, float, int, Optional[str]]]:
        """Return beliefs with steadfastness metadata.

        Tuple: (belief_id, sti, last_injection_time, total_injections, last_event_class_name)
        Beliefs with zero injection since boot (STI<=0) are excluded.
        Ties at the STI floor are broken by (last_injection_time, creation_time).
        """
        if not self._enabled:
            return []

        scored = [
            (bid, self._bridge.bank.get_sti(bid))
            for bid in self._bridge._belief_ids
        ]
        scored = [(bid, sti) for bid, sti in scored if sti > 0.0]
        scored.sort(
            key=lambda x: (
                -x[1],
                -(self._records[x[0]].last_injection_time if x[0] in self._records else 0.0),
                -(self._records[x[0]].creation_time if x[0] in self._records else 0.0),
            )
        )
        result = []
        for bid, sti in scored[:limit]:
            rec = self._records.get(bid)
            result.append((
                bid, sti,
                rec.last_injection_time if rec else 0.0,
                rec.total_injections if rec else 0,
                rec.last_event_class.value if rec and rec.last_event_class else None,
            ))
        return result

    def set_claim_state(self, belief_id: str, state: ClaimState, evidence_ref: str) -> None:
        """Set the claim state for a belief. Never auto-upgrades."""
        if not self._enabled:
            return
        rec = self._get_or_create(belief_id)
        if _LADDER_ORDER[state] > _LADDER_ORDER[rec.claim_state]:
            rec.claim_state = state

    def get_claim_state(self, belief_id: str) -> Optional[ClaimState]:
        rec = self._records.get(belief_id)
        return rec.claim_state if rec else None

    def summary(self) -> dict[str, Any]:
        if not self._enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "dmax": DMAX,
            "tracked_beliefs": len(self._records),
            "claim_states": {
                s.value: sum(1 for r in self._records.values() if r.claim_state == s)
                for s in ClaimState
            },
        }
