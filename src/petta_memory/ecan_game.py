
"""ECAN Game World: simulated world for testing attention allocation.

Instead of waiting for real incidents, the game scripts a sequence of
world events (evidence arriving, incidents firing, attention shifting)
and validates that ECAN attention allocation produces the expected
belief prioritization at each turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .ecan_bridge import ECANBridge
from .ecan import AttentionValue, ECANCycleResult
from .simworld import SimWorld, SimStep, WorldItem
from .store import MediumMemoryStore
from .goalchainer_smoke import run_goalchainer_precompiled_handoff_smoke

DEFAULT_REPO = Path(__file__).resolve().parents[5] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"


@dataclass
class GameEvent:
    """A single event in the simulated world."""
    label: str
    description: str = ""
    add_evidence: list[WorldItem] = field(default_factory=list)
    remove_evidence: list[str] = field(default_factory=list)
    stimulate: dict[str, float] = field(default_factory=dict)
    extra_cycles: int = 0


@dataclass
class GameTurnExpectation:
    """What we expect to observe after a game turn."""
    expected_top_action: Optional[str] = None
    expected_status: Optional[str] = None
    expected_strength_min: Optional[float] = None
    expected_strength_max: Optional[float] = None
    expected_af_contains: list[str] = field(default_factory=list)
    expected_af_excludes: list[str] = field(default_factory=list)
    expected_top_attention_belief: Optional[str] = None
    expected_forget_candidates: list[str] = field(default_factory=list)
    expected_funds_sti_min: Optional[float] = None
    expected_funds_sti_max: Optional[float] = None
    description: str = ""


@dataclass
class GameTurn:
    """One turn of the game: events -> ECAN cycle -> decision -> check."""
    events: list[GameEvent] = field(default_factory=list)
    expectation: GameTurnExpectation = field(default_factory=GameTurnExpectation)
    label: str = ""
    description: str = ""


@dataclass
class GameTurnResult:
    """Result of executing one game turn."""
    turn: int
    label: str
    events_applied: int
    cycle_result: Optional[ECANCycleResult]
    top_action: Optional[str]
    top_status: Optional[str]
    top_strength: Optional[float]
    top_attention_belief: Optional[str]
    af_beliefs: list[str]
    forget_candidates: list[str]
    funds_sti: float
    prioritized_beliefs: list[tuple[str, float]]
    checks_passed: bool
    check_errors: list[str]


class ECANGameWorld:
    """A simulated world that drives ECAN attention allocation through game turns.

    Combines evidence-based decision making (SimWorld) with attention
    dynamics (ECANBridge) to test whether the system correctly prioritizes
    beliefs under changing conditions.
    """

    def __init__(
        self,
        *,
        goalchainer_repo: str | Path = DEFAULT_REPO,
        ecan_params: Optional[dict[str, float]] = None,
        request: str | None = None,
    ):
        self.goalchainer_repo = Path(goalchainer_repo)
        self.request = request or ""
        self.ecan_params = ecan_params or {}
        self.items: list[WorldItem] = []
        self.store = MediumMemoryStore(Path("/dev/null"))
        self.bridge = ECANBridge(self.store, params=ecan_params)
        self._all_belief_ids: set[str] = set()
        self.results: list[GameTurnResult] = []
        self.turn_index = 0

    @staticmethod
    def stv_item(
        belief_id: str,
        action: str,
        strength: float,
        confidence: float,
        *,
        cluster_id: str = "mc-game",
        promotion_event: str = "pe-game",
        promotion_domain: str = "incident-response",
    ) -> WorldItem:
        return SimWorld.stv_item(
            belief_id, action, strength, confidence,
            cluster_id=cluster_id, promotion_event=promotion_event,
            promotion_domain=promotion_domain,
        )

    @staticmethod
    def ec_item(
        belief_id: str,
        action: str,
        support: float,
        opposition: float,
        *,
        cluster_id: str = "mc-game",
        promotion_event: str = "pe-game",
        promotion_domain: str = "incident-response",
    ) -> WorldItem:
        return SimWorld.ec_item(
            belief_id, action, support, opposition,
            cluster_id=cluster_id, promotion_event=promotion_event,
            promotion_domain=promotion_domain,
        )

    def _register_belief(self, belief_id: str) -> None:
        """Register a belief in the ECAN bank if not already present."""
        if belief_id not in self._all_belief_ids:
            self._all_belief_ids.add(belief_id)
            self.bridge._belief_ids.add(belief_id)
            self.bridge.bank.set_av(belief_id, AttentionValue(sti=0, lti=0, vlti=0))

    def _build_evidence_map(self) -> dict[str, list[str]]:
        """Build evidence map from current items (belief_id -> [evidence sources])."""
        evidence_map: dict[str, list[str]] = {}
        for item in self.items:
            if item.belief_id not in evidence_map:
                evidence_map[item.belief_id] = []
            src = f"ev-{item.belief_id}-{item.slot}"
            evidence_map[item.belief_id].append(src)
            self._register_belief(src)
        return evidence_map

    def _rebuild_diffusion_graph(self) -> None:
        """Rebuild the diffusion graph from current evidence."""
        evidence_map = self._build_evidence_map()
        self.bridge.diffusion.build_from_evidence_map(evidence_map)

    def _apply_event(self, event: GameEvent) -> None:
        """Apply a single world event."""
        if event.remove_evidence:
            self.items = [it for it in self.items if it.belief_id not in event.remove_evidence]
        for item in event.add_evidence:
            self._register_belief(item.belief_id)
            self.items.append(item)
        if event.add_evidence or event.remove_evidence:
            self._rebuild_diffusion_graph()
        if event.extra_cycles > 0:
            self.bridge.run_cycles(event.extra_cycles)

    def _run_decision(self) -> dict[str, Any]:
        """Run the goalchainer decision engine on current evidence."""
        cache = {
            "schema": "petta-memory-goalchainer-handoff-v1",
            "cache_id": f"ecan-game-turn-{self.turn_index}",
            "items": [item.to_cache_item() for item in self.items],
        }
        kwargs: dict[str, Any] = {"goalchainer_repo": self.goalchainer_repo}
        if self.request:
            kwargs["request"] = self.request
        smoke = run_goalchainer_precompiled_handoff_smoke(cache, **kwargs)
        payload = smoke.get("decision_payload", {})
        decisions = payload.get("decisions", [])
        top = decisions[0] if decisions else {}
        return {
            "top_action": top.get("action_id"),
            "top_status": top.get("status"),
            "top_strength": (top.get("evidence") or {}).get("strength"),
            "all_decisions": decisions,
        }

    def play_turn(self, turn: GameTurn) -> GameTurnResult:
        """Execute one game turn: apply events, run ECAN, make decision, check."""
        self.turn_index += 1

        # 1. Apply all events
        events_applied = 0
        combined_stimuli: dict[str, float] = {}
        for event in turn.events:
            self._apply_event(event)
            events_applied += 1
            for bid, stim in event.stimulate.items():
                self._register_belief(bid)
                combined_stimuli[bid] = combined_stimuli.get(bid, 0.0) + stim

        # 2. Run ECAN cycle with combined stimuli
        if combined_stimuli:
            cycle_result = self.bridge.run_cycle(stimuli=combined_stimuli)
        else:
            cycle_result = self.bridge.run_cycle()

        # 3. Run decision engine
        decision = self._run_decision()

        # 4. Collect attention state
        prioritized = self.bridge.get_prioritized_beliefs(limit=50)
        top_attn_belief = prioritized[0][0] if prioritized else None
        af_beliefs = self.bridge.get_attentional_focus_beliefs()
        forget_candidates = self.bridge.get_forget_candidates()

        # 5. Check expectations
        check_errors: list[str] = []
        exp = turn.expectation

        if exp.expected_top_action is not None:
            if decision["top_action"] != exp.expected_top_action:
                check_errors.append(
                    f"expected top action={exp.expected_top_action}, "
                    f"got={decision['top_action']}"
                )

        if exp.expected_status is not None:
            if decision["top_status"] != exp.expected_status:
                check_errors.append(
                    f"expected status={exp.expected_status}, "
                    f"got={decision['top_status']}"
                )

        if exp.expected_strength_min is not None:
            if decision["top_strength"] is None or decision["top_strength"] < exp.expected_strength_min:
                check_errors.append(
                    f"expected strength>={exp.expected_strength_min}, "
                    f"got={decision['top_strength']}"
                )

        if exp.expected_strength_max is not None:
            if decision["top_strength"] is None or decision["top_strength"] > exp.expected_strength_max:
                check_errors.append(
                    f"expected strength<={exp.expected_strength_max}, "
                    f"got={decision['top_strength']}"
                )

        if exp.expected_af_contains:
            af_set = set(af_beliefs)
            for bid in exp.expected_af_contains:
                if bid not in af_set:
                    check_errors.append(f"expected {bid} in AF, not present")

        if exp.expected_af_excludes:
            af_set = set(af_beliefs)
            for bid in exp.expected_af_excludes:
                if bid in af_set:
                    check_errors.append(f"expected {bid} NOT in AF, but present")

        if exp.expected_top_attention_belief is not None:
            if top_attn_belief != exp.expected_top_attention_belief:
                check_errors.append(
                    f"expected top attention={exp.expected_top_attention_belief}, "
                    f"got={top_attn_belief}"
                )

        if exp.expected_forget_candidates:
            fc_set = set(forget_candidates)
            for bid in exp.expected_forget_candidates:
                if bid not in fc_set:
                    check_errors.append(f"expected {bid} in forget candidates, not present")

        if exp.expected_funds_sti_min is not None:
            if self.bridge.bank.funds_sti < exp.expected_funds_sti_min:
                check_errors.append(
                    f"expected funds_sti>={exp.expected_funds_sti_min}, "
                    f"got={self.bridge.bank.funds_sti}"
                )

        if exp.expected_funds_sti_max is not None:
            if self.bridge.bank.funds_sti > exp.expected_funds_sti_max:
                check_errors.append(
                    f"expected funds_sti<={exp.expected_funds_sti_max}, "
                    f"got={self.bridge.bank.funds_sti}"
                )

        passed = len(check_errors) == 0

        result = GameTurnResult(
            turn=self.turn_index,
            label=turn.label,
            events_applied=events_applied,
            cycle_result=cycle_result,
            top_action=decision["top_action"],
            top_status=decision["top_status"],
            top_strength=decision["top_strength"],
            top_attention_belief=top_attn_belief,
            af_beliefs=af_beliefs,
            forget_candidates=forget_candidates,
            funds_sti=self.bridge.bank.funds_sti,
            prioritized_beliefs=prioritized,
            checks_passed=passed,
            check_errors=check_errors,
        )
        self.results.append(result)
        return result

    def all_passed(self) -> bool:
        """Check if all turns so far passed their expectations."""
        return all(r.checks_passed for r in self.results)

    def summary(self) -> str:
        """Human-readable summary of all game turns."""
        lines = []
        for r in self.results:
            status = "PASS" if r.checks_passed else "FAIL"
            lines.append(f"Turn {r.turn} [{r.label}] {status}")
            lines.append(f"  events={r.events_applied}, top_action={r.top_action}, status={r.top_status}")
            lines.append(f"  top_attn={r.top_attention_belief}, AF={r.af_beliefs[:5]}")
            lines.append(f"  funds_sti={r.funds_sti:.1f}, forget={r.forget_candidates[:3]}")
            if r.check_errors:
                for e in r.check_errors:
                    lines.append(f"  ERROR: {e}")
        return "\n".join(lines)

    def play_scenarios(self, turns: list[GameTurn]) -> bool:
        """Play a full game (multiple turns). Returns True if all passed."""
        for turn in turns:
            self.play_turn(turn)
        return self.all_passed()
