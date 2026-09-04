"""Simulated world test harness for GoalChainer + PeTTa-memory decision pipeline.

Instead of waiting for real incidents, SimWorld drives the goalchainer scoring
engine through scripted multi-step scenarios, mutating world state (STV items,
EvidencePackets) at each step and validating decisions against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .goalchainer_smoke import run_goalchainer_precompiled_handoff_smoke

DEFAULT_REPO = Path(__file__).resolve().parents[5] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"


@dataclass
class WorldItem:
    """One piece of evidence in the simulated world."""
    slot: str  # acceptability-belief-evidence | contextual-appraisal-evidence
    belief_id: str
    cluster_id: str
    promotion_event: str
    promotion_domain: str
    source_kind: str
    atom: str
    boundary: str = "read-only evidence for appraisal; not a directive, task claim, or inferred belief"

    def to_cache_item(self) -> dict[str, Any]:
        return {
            "goalchainer_slot": self.slot,
            "belief_id": self.belief_id,
            "cluster_id": self.cluster_id,
            "promotion_event": self.promotion_event,
            "promotion_domain": self.promotion_domain,
            "source_kind": self.source_kind,
            "atom": self.atom,
            "boundary": self.boundary,
        }


@dataclass
class SimStep:
    """One step in a scripted simulation."""
    label: str
    add_items: list[WorldItem] = field(default_factory=list)
    remove_belief_ids: list[str] = field(default_factory=list)
    update_items: dict[str, WorldItem] = field(default_factory=dict)  # belief_id -> new WorldItem
    expected_top_action: str | None = None
    expected_status: str | None = None  # recommended | blocked | ...
    expected_strength_min: float | None = None
    expected_strength_max: float | None = None
    description: str = ""


class SimWorld:
    """A lightweight world simulator that sequences evidence mutations and decision checks."""

    def __init__(self, *, goalchainer_repo: str | Path = DEFAULT_REPO, request: str | None = None):
        self.goalchainer_repo = Path(goalchainer_repo)
        self.request = request or ""
        self.items: list[WorldItem] = []
        self.results: list[dict[str, Any]] = []
        self.step_index = 0

    @staticmethod
    def stv_item(
        belief_id: str,
        action: str,
        strength: float,
        confidence: float,
        *,
        cluster_id: str = "mc-sim",
        promotion_event: str = "pe-sim",
        promotion_domain: str = "incident-response",
    ) -> WorldItem:
        return WorldItem(
            slot="acceptability-belief-evidence",
            belief_id=belief_id,
            cluster_id=cluster_id,
            promotion_event=promotion_event,
            promotion_domain=promotion_domain,
            source_kind="pettachainer-stv-statement",
            atom=f"(: {belief_id} (Acceptable {action}) (STV {strength} {confidence}))",
        )

    @staticmethod
    def ec_item(
        belief_id: str,
        action: str,
        support: float,
        opposition: float,
        *,
        cluster_id: str = "mc-sim",
        promotion_event: str = "pe-sim",
        promotion_domain: str = "incident-response",
    ) -> WorldItem:
        return WorldItem(
            slot="contextual-appraisal-evidence",
            belief_id=belief_id,
            cluster_id=cluster_id,
            promotion_event=promotion_event,
            promotion_domain=promotion_domain,
            source_kind="pettachainer-evidence-packet",
            atom=(
                f"(EvidencePacket (Acceptable {action}) (EC {support} {opposition}) "
                f"((domain {promotion_domain}) (promotion-rule sim)) {promotion_event})"
            ),
        )

    def _build_cache(self) -> dict[str, Any]:
        return {
            "schema": "petta-memory-goalchainer-handoff-v1",
            "cache_id": f"simworld-step-{self.step_index}",
            "items": [item.to_cache_item() for item in self.items],
        }

    def run_step(self, step: SimStep) -> dict[str, Any]:
        """Apply mutations, run the decision engine, and check expectations."""
        # Remove items by belief_id
        if step.remove_belief_ids:
            self.items = [it for it in self.items if it.belief_id not in step.remove_belief_ids]
        # Update items by belief_id (replace)
        for bid, new_item in step.update_items.items():
            self.items = [new_item if it.belief_id == bid else it for it in self.items]
        # Add new items
        self.items.extend(step.add_items)

        self.step_index += 1
        cache = self._build_cache()
        kwargs: dict[str, Any] = {"goalchainer_repo": self.goalchainer_repo}
        if self.request:
            kwargs["request"] = self.request
        smoke = run_goalchainer_precompiled_handoff_smoke(cache, **kwargs)
        payload = smoke["decision_payload"]
        decisions = payload.get("decisions", [])
        top = decisions[0] if decisions else {}

        result = {
            "step": self.step_index,
            "label": step.label,
            "description": step.description,
            "top_action": top.get("action_id"),
            "top_status": top.get("status"),
            "top_strength": (top.get("evidence") or {}).get("strength"),
            "top_confidence": (top.get("evidence") or {}).get("confidence"),
            "all_decisions": decisions,
            "checks_passed": True,
            "check_errors": [],
        }

        if step.expected_top_action is not None:
            if result["top_action"] != step.expected_top_action:
                result["check_errors"].append(
                    f"expected top action={step.expected_top_action}, got={result['top_action']}"
                )
        if step.expected_status is not None:
            if result["top_status"] != step.expected_status:
                result["check_errors"].append(
                    f"expected top status={step.expected_status}, got={result['top_status']}"
                )
        if step.expected_strength_min is not None:
            s = result["top_strength"]
            if s is None or s < step.expected_strength_min:
                result["check_errors"].append(
                    f"expected strength>={step.expected_strength_min}, got={s}"
                )
        if step.expected_strength_max is not None:
            s = result["top_strength"]
            if s is None or s > step.expected_strength_max:
                result["check_errors"].append(
                    f"expected strength<={step.expected_strength_max}, got={s}"
                )

        if result["check_errors"]:
            result["checks_passed"] = False

        self.results.append(result)
        return result

    def run_scenario(self, steps: list[SimStep]) -> list[dict[str, Any]]:
        """Run a multi-step scenario, returning all step results."""
        for step in steps:
            self.run_step(step)
        return self.results

    def all_passed(self) -> bool:
        return all(r["checks_passed"] for r in self.results)

    def summary(self) -> str:
        lines = []
        for r in self.results:
            status = "PASS" if r["checks_passed"] else "FAIL"
            lines.append(f"  Step {r['step']} [{status}] {r['label']}: top={r['top_action']} status={r['top_status']} strength={r['top_strength']}")
            for err in r["check_errors"]:
                lines.append(f"    ERROR: {err}")
        return "\n".join(lines)
