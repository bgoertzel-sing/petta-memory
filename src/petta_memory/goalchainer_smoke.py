"""Non-live GoalChainer smoke gate for PeTTa memory handoff artifacts.

This module deliberately runs GoalChainer as an external, bounded subprocess and
validates only a decision payload. It does not load an OmegaClaw skill, claim a
task, or write back to memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

from .store import ValidationError

Runner = Callable[..., subprocess.CompletedProcess[str]]


DEFAULT_GOALCHAINER_REPO = Path(__file__).resolve().parents[5] / "omegaclaw" / "repos" / "OmegaClaw-GoalChainer"
DEFAULT_PETTA_DIR = Path(__file__).resolve().parents[3] / "PeTTa"
DEFAULT_PETTACHAINER_DIR = Path(__file__).resolve().parents[3] / "PeTTaChainer"
DEFAULT_SWIPL = Path(__file__).resolve().parents[5] / "omegaclaw" / "local" / "swipl-9.3.36" / "bin" / "swipl"
DEFAULT_REQUEST = (
    "Checkout is down. Engineering wants to paste raw logs into the incident room. "
    "Support says the logs may include customer emails, order IDs, and request payloads. "
    "Use the attached promoted PeTTa memory evidence only as read-only appraisal context."
)

_ACCEPTABLE_STV_RE = re.compile(
    r"\(Acceptable\s+(?P<action>[A-Za-z0-9_:-]+)\)\s+\(STV\s+(?P<strength>[0-9.eE+-]+)\s+(?P<confidence>[0-9.eE+-]+)\)"
)
_REQUIRES_STV_RE = re.compile(
    r"\(Requires\s+(?P<target>[A-Za-z0-9_:-]+)\)\s+\(STV\s+(?P<strength>[0-9.eE+-]+)\s+(?P<confidence>[0-9.eE+-]+)\)"
)
_NOT_APPLICABLE_STV_RE = re.compile(
    r"\(NotApplicable\s+(?P<action>[A-Za-z0-9_:-]+)\).*\(STV\s+(?P<strength>[0-9.eE+-]+)\s+(?P<confidence>[0-9.eE+-]+)\)"
)
_EVIDENCE_PACKET_ACCEPTABLE_RE = re.compile(
    r"\(EvidencePacket\s+\(Acceptable\s+(?P<action>[A-Za-z0-9_:-]+)\)\s+"
    r"\(EC\s+(?P<support>[0-9.eE+-]+)\s+(?P<opposition>[0-9.eE+-]+)\)"
)


def run_goalchainer_handoff_smoke(
    handoff_cache: dict[str, object],
    *,
    goalchainer_repo: str | Path = DEFAULT_GOALCHAINER_REPO,
    request: str = DEFAULT_REQUEST,
    timeout_sec: float = 20.0,
    runner: Runner = subprocess.run,
) -> dict[str, object]:
    """Run and validate a bounded non-live GoalChainer decision smoke.

    The handoff cache supplies provenance and a hand-picked evidence item. The
    GoalChainer command is limited to ``demo --json`` so it produces ranked
    actions/explanation only; directive/task-claim commands are intentionally not
    invoked.
    """
    items = list(handoff_cache.get("items", []))
    if not items:
        raise ValidationError("GoalChainer smoke requires at least one handoff item")
    selected = _select_items(items)
    repo = Path(goalchainer_repo)
    if not (repo / "src" / "goal_chainer" / "cli.py").exists():
        raise ValidationError(f"GoalChainer repo not found or incomplete: {repo}")
    if timeout_sec <= 0:
        raise ValidationError("timeout_sec must be positive")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    if "GOALCHAINER_PETTA_DIR" not in env and (DEFAULT_PETTA_DIR / "src" / "main.pl").exists():
        env["GOALCHAINER_PETTA_DIR"] = str(DEFAULT_PETTA_DIR)
    if (
        "GOALCHAINER_PETTACHAINER_DIR" not in env
        and (DEFAULT_PETTACHAINER_DIR / "pettachainer" / "metta" / "petta_chainer.metta").exists()
    ):
        env["GOALCHAINER_PETTACHAINER_DIR"] = str(DEFAULT_PETTACHAINER_DIR)
    if "GOALCHAINER_PETTA_SWIPL" not in env and DEFAULT_SWIPL.exists():
        env["GOALCHAINER_PETTA_SWIPL"] = str(DEFAULT_SWIPL)
    # Keep this smoke on the deterministic offline path: no Ollama/semantic path
    # and no live PeTTaChainer compileadd/query gate.
    env.pop("GOALCHAINER_SEMANTIC", None)
    command = [
        sys.executable,
        "-m",
        "goal_chainer.cli",
        "demo",
        "--json",
        "--request",
        request,
    ]
    try:
        completed = runner(
            command,
            cwd=str(repo),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(f"GoalChainer smoke timed out after {timeout_sec}s") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise ValidationError(f"GoalChainer smoke failed with exit {completed.returncode}: {stderr}")
    try:
        goal_payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError("GoalChainer smoke did not emit JSON") from exc
    _validate_goalchainer_payload(goal_payload)
    return {
        "schema": "petta-memory-goalchainer-smoke-v1",
        "mode": "non-live-decision-payload-only",
        "goalchainer_repo": str(repo),
        "goalchainer_command": command,
        "timeout_sec": timeout_sec,
        "request": request,
        "input_cache_id": handoff_cache.get("cache_id"),
        "input_schema": handoff_cache.get("schema"),
        "selected_handoff_items": selected,
        "boundary": (
            "non-live smoke only; no OmegaClaw skill loaded, no task/directive claim, "
            "no memory write, no inferred-belief status"
        ),
        "decision_payload": {
            "scenario": goal_payload.get("scenario"),
            "runtime": goal_payload.get("runtime"),
            "decisions": goal_payload.get("decisions", []),
            "motivation": goal_payload.get("motivation"),
            "explanation": goal_payload.get("explanation", []),
            "notes": goal_payload.get("notes", []),
        },
        "checks": {
            "ranked_actions": len(goal_payload.get("decisions", [])) >= 2,
            "recommended_action_present": any(
                item.get("status") == "recommended" for item in goal_payload.get("decisions", [])
            ),
            "provenance_selected_handoff_items": bool(selected),
            "no_live_directive_or_task_claim": True,
            "no_memory_write": True,
        },
    }


def run_goalchainer_precompiled_handoff_smoke(
    handoff_cache: dict[str, object],
    *,
    goalchainer_repo: str | Path = DEFAULT_GOALCHAINER_REPO,
    request: str = DEFAULT_REQUEST,
    include_heuristic_memory_probe: bool = False,
    admitted_patham9_handoff: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run GoalChainer's decision engine from a precompiled handoff cache.

    This is the bounded bypass for the current PeTTaChainer ``compileadd``
    blocker: it imports only GoalChainer's scenario/scoring/explanation code,
    supplies a local reasoner backed by promoted STV handoff items, and never
    invokes ``goal_chainer.cli``, PeTTaChainer ``compileadd``, directive,
    execution, skill, or memory-write paths.
    """
    items = list(handoff_cache.get("items", []))
    if not items:
        raise ValidationError("GoalChainer precompiled smoke requires at least one handoff item")
    selected = _select_items(items)
    repo = Path(goalchainer_repo)
    src = repo / "src"
    if not (src / "goal_chainer" / "scenarios.py").exists():
        raise ValidationError(f"GoalChainer repo not found or incomplete: {repo}")

    action_evidence = _action_evidence_from_handoff(items)
    inserted = False
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
        inserted = True
    try:
        from goal_chainer.explain import explain_decisions
        from goal_chainer.models import CandidateAction, EvidenceProjection, Goal, GoalScenario, Norm
        from goal_chainer.scenarios import incident_response_scenario
        from goal_chainer.scoring import DecisionEngine

        class PrecompiledHandoffReasoner:
            source = "petta-memory-precompiled-handoff-cache"

            def project(self, action):
                row = action_evidence.get(action.id)
                if row is None:
                    row = _default_action_evidence(action)
                return EvidenceProjection(
                    strength=row["strength"],
                    confidence=row["confidence"],
                    source=self.source,
                    projection=row["projection"],
                    proofs=tuple(row["proofs"]),
                    deontic=row["deontic"],
                    expectation=row["expectation"],
                )

        _mark_patham9_admitted(action_evidence, admitted_patham9_handoff)
        scenario = _project_control_scenario(action_evidence) or incident_response_scenario(request)
        reasoner = PrecompiledHandoffReasoner()
        decisions = DecisionEngine(reasoner).rank(scenario)
        reasoner_result = {
            "source": reasoner.source,
            "engine": "GoalChainer scoring over PeTTa-memory precompiled handoff cache",
            "execution": {
                "mode": "non-live-precompiled-cache",
                "compileadd": "not-invoked",
                "directive": "not-invoked",
                "memory_write": "not-invoked",
            },
            "action_evidence": [
                action_evidence.get(action.id, _default_action_evidence(action))
                for action in scenario.actions
            ],
        }
        payload = {
            "scenario": scenario.title,
            "notes": list(scenario.notes),
            "runtime": {"reasoner": reasoner.source},
            "decisions": [decision.to_dict() for decision in decisions],
            "explanation": explain_decisions(decisions, reasoner_result),
            "motivation": None,
        }
    finally:
        if inserted:
            try:
                sys.path.remove(str(src))
            except ValueError:
                pass

    _validate_goalchainer_payload(payload)
    result = {
        "schema": "petta-memory-goalchainer-precompiled-smoke-v1",
        "mode": "non-live-precompiled-cache-decision-payload-only",
        "goalchainer_repo": str(repo),
        "request": request,
        "input_cache_id": handoff_cache.get("cache_id"),
        "input_schema": handoff_cache.get("schema"),
        "selected_handoff_items": selected,
        "boundary": (
            "non-live precompiled cache smoke only; PeTTaChainer compileadd/query not invoked; "
            "no OmegaClaw skill loaded, no task/directive claim, no memory write, no inferred-belief status"
        ),
        "decision_payload": payload,
        "checks": {
            "ranked_actions": len(payload["decisions"]) >= 2,
            "recommended_action_present": any(
                item.get("status") == "recommended" for item in payload["decisions"]
            ),
            "provenance_selected_handoff_items": bool(selected),
            "compileadd_not_invoked": True,
            "no_live_directive_or_task_claim": True,
            "no_memory_write": True,
        },
    }
    if include_heuristic_memory_probe:
        result["heuristic_memory_probe"] = _run_heuristic_memory_probe(
            items,
            goalchainer_repo=repo,
            request=request,
        )
        result["checks"]["heuristic_with_memory_path_checked"] = True
    else:
        result["checks"]["heuristic_with_memory_path_checked"] = False
    return result


def _run_heuristic_memory_probe(
    items: list[object],
    *,
    goalchainer_repo: Path,
    request: str,
) -> dict[str, object]:
    """Exercise GoalChainer's actual heuristic-with-memory path non-live.

    This complements the precompiled decision-payload bypass above by importing
    GoalChainer's pipeline and passing the same handoff items into
    ``solve_incident(memory_items=...)``.  It is still a local smoke: no
    OmegaClaw skill is loaded, no directive/task claim is accepted, no memory is
    written, and the only execution is GoalChainer's local redaction demo.
    """
    memory_items = [item for item in items if isinstance(item, dict)]
    if not memory_items:
        raise ValidationError("heuristic memory probe requires dict handoff items")
    src = goalchainer_repo / "src"
    if not (src / "goal_chainer" / "pipeline.py").exists():
        raise ValidationError(f"GoalChainer pipeline not found: {goalchainer_repo}")

    old_env = {key: os.environ.get(key) for key in ("GOALCHAINER_PETTA_DIR", "GOALCHAINER_PETTACHAINER_DIR", "GOALCHAINER_PETTA_SWIPL")}
    if "GOALCHAINER_PETTA_DIR" not in os.environ and (DEFAULT_PETTA_DIR / "src" / "main.pl").exists():
        os.environ["GOALCHAINER_PETTA_DIR"] = str(DEFAULT_PETTA_DIR)
    if (
        "GOALCHAINER_PETTACHAINER_DIR" not in os.environ
        and (DEFAULT_PETTACHAINER_DIR / "pettachainer" / "metta" / "petta_chainer.metta").exists()
    ):
        os.environ["GOALCHAINER_PETTACHAINER_DIR"] = str(DEFAULT_PETTACHAINER_DIR)
    if "GOALCHAINER_PETTA_SWIPL" not in os.environ and DEFAULT_SWIPL.exists():
        os.environ["GOALCHAINER_PETTA_SWIPL"] = str(DEFAULT_SWIPL)

    inserted = False
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
        inserted = True
    try:
        from goal_chainer.heuristic_beliefs import parse_memory_evidence
        from goal_chainer.pipeline import solve_incident

        parsed = parse_memory_evidence(memory_items)
        if not parsed:
            raise ValidationError("GoalChainer heuristic memory probe parsed no memory evidence")
        payload = solve_incident(request, memory_items=memory_items)
    finally:
        if inserted:
            try:
                sys.path.remove(str(src))
            except ValueError:
                pass
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    decisions = payload.get("decisions", [])
    recommended = next((item for item in decisions if item.get("status") == "recommended"), decisions[0] if decisions else {})
    proofs = recommended.get("evidence", {}).get("proofs", []) if isinstance(recommended, dict) else []
    executed = payload.get("executed", {})
    leak_check = executed.get("leak_check", {}) if isinstance(executed, dict) else {}
    return {
        "schema": "petta-memory-goalchainer-heuristic-memory-probe-v1",
        "mode": "non-live-goalchainer-solve-incident-memory-items",
        "parsed_memory_items": len(parsed),
        "decided": payload.get("decided"),
        "status": payload.get("status"),
        "recommended_action": recommended.get("action_id"),
        "recommended_evidence_source": recommended.get("evidence", {}).get("source") if isinstance(recommended, dict) else None,
        "recommended_evidence_projection": recommended.get("evidence", {}).get("projection") if isinstance(recommended, dict) else None,
        "memory_proof_present": any("memory_" in str(proof) or "+memory" in str(proof) for proof in proofs),
        "leak_check_safe": leak_check.get("safe"),
        "boundary": (
            "non-live heuristic-with-memory probe; no OmegaClaw skill loaded, no accepted task/directive claim, "
            "no memory write, no live Telegram/runtime bridge"
        ),
    }


def _select_items(items: list[object]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    wanted_slots = ["acceptability-belief-evidence", "contextual-appraisal-evidence"]
    for slot in wanted_slots:
        for item in items:
            if isinstance(item, dict) and item.get("goalchainer_slot") == slot:
                selected.append(_compact_item(item))
                break
    if not selected and isinstance(items[0], dict):
        selected.append(_compact_item(items[0]))
    return selected


def _compact_item(item: dict[str, object]) -> dict[str, object]:
    return {
        "goalchainer_slot": item.get("goalchainer_slot"),
        "belief_id": item.get("belief_id"),
        "cluster_id": item.get("cluster_id"),
        "promotion_event": item.get("promotion_event"),
        "promotion_domain": item.get("promotion_domain"),
        "source_kind": item.get("source_kind"),
        "atom": item.get("atom"),
        "boundary": item.get("boundary"),
    }


def _action_evidence_from_handoff(items: list[object]) -> dict[str, dict[str, Any]]:
    stv_rows: dict[str, dict[str, Any]] = {}
    ec_rows: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        atom = str(item.get("atom", ""))
        if item.get("goalchainer_slot") == "acceptability-belief-evidence":
            match = _ACCEPTABLE_STV_RE.search(atom)
            source_pattern = "Acceptable"
            deontic = "permitted"
            if match is None:
                match = _REQUIRES_STV_RE.search(atom)
                source_pattern = "Requires"
                deontic = "obligated"
            if match is None:
                match = _NOT_APPLICABLE_STV_RE.search(atom)
                source_pattern = "NotApplicable"
                deontic = "forbidden"
            if match is None:
                continue
            action = _goalchainer_action_id(match.groupdict().get("action") or match.groupdict().get("target") or "")
            strength = _bounded_float(match.group("strength"), "strength")
            confidence = _bounded_float(match.group("confidence"), "confidence")
            if source_pattern == "NotApplicable":
                strength = round(1.0 - strength, 6)
            candidate = {
                "action_id": action,
                "source_pattern": source_pattern,
                "deontic": deontic if deontic != "permitted" else _default_deontic(action),
                "expectation": _expectation(strength, confidence),
                "strength": strength,
                "confidence": confidence,
                "opinion": _sl_opinion(strength, confidence),
                "projection": atom,
                "proofs": [
                    "precompiled PeTTa-memory handoff STV item; PeTTaChainer compileadd not invoked",
                    f"belief_id={item.get('belief_id')} cluster_id={item.get('cluster_id')} promotion_event={item.get('promotion_event')}",
                ],
                "contextual_evidence": [],
                "patham9_admitted": False,
            }
            old = stv_rows.get(action)
            if old is None or candidate["confidence"] > old["confidence"]:
                stv_rows[action] = candidate
        elif item.get("goalchainer_slot") == "contextual-appraisal-evidence":
            match = _EVIDENCE_PACKET_ACCEPTABLE_RE.search(atom)
            if match is None:
                continue
            action = match.group("action")
            support = _non_negative_float(match.group("support"), "support")
            opposition = _non_negative_float(match.group("opposition"), "opposition")
            total = support + opposition
            if total <= 0:
                continue
            ec_rows.setdefault(action, []).append(
                {
                    "support": support,
                    "opposition": opposition,
                    "strength": support / total,
                    "confidence": total / (total + 2.0),
                    "atom": atom,
                    "belief_id": item.get("belief_id"),
                    "cluster_id": item.get("cluster_id"),
                    "promotion_event": item.get("promotion_event"),
                }
            )
    if not stv_rows:
        raise ValidationError("GoalChainer precompiled smoke found no Acceptable STV handoff items")
    for action, row in stv_rows.items():
        for ec in ec_rows.get(action, []):
            stv_weight = row["confidence"]
            ec_weight = ec["confidence"]
            total_weight = stv_weight + ec_weight
            if total_weight > 0:
                row["strength"] = round(
                    ((row["strength"] * stv_weight) + (ec["strength"] * ec_weight)) / total_weight,
                    6,
                )
                row["confidence"] = round(max(row["confidence"], ec["confidence"]), 6)
                row["expectation"] = _expectation(row["strength"], row["confidence"])
                row["opinion"] = _sl_opinion(row["strength"], row["confidence"])
            row["contextual_evidence"].append(
                {
                    "support": ec["support"],
                    "opposition": ec["opposition"],
                    "derived_strength": round(ec["strength"], 6),
                    "derived_confidence": round(ec["confidence"], 6),
                    "belief_id": ec["belief_id"],
                    "cluster_id": ec["cluster_id"],
                    "promotion_event": ec["promotion_event"],
                }
            )
            row["proofs"].append(
                "contextual EvidencePacket EC support/opposition influenced precompiled appraisal; "
                "PeTTaChainer compileadd not invoked"
            )
    return stv_rows


def _mark_patham9_admitted(
    action_evidence: dict[str, dict[str, Any]],
    admitted_patham9_handoff: dict[str, object] | None,
) -> None:
    if not isinstance(admitted_patham9_handoff, dict):
        return
    for item in admitted_patham9_handoff.get("items", []):
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", ""))
        match = _ACCEPTABLE_STV_RE.search(term) or re.search(r"^\(Acceptable\s+(?P<action>[A-Za-z0-9_:-]+)\)$", term)
        if match is None:
            continue
        action = _goalchainer_action_id(match.group("action"))
        row = action_evidence.get(action)
        if row is None:
            continue
        row["patham9_admitted"] = True
        row["proofs"].append(
            "patham9/PLN admitted this branch before GoalChainer appraisal; treated as acceptability evidence"
        )


def _project_control_scenario(action_evidence: dict[str, dict[str, Any]]) -> Any | None:
    """Build a small dynamic GoalChainer scenario for project-control decisions.

    This keeps the GoalChainer scoring/appraisal engine in the loop when the
    memory evidence is not about the default incident-redaction demo.  The first
    concrete target is the ThreadKeeper canary decision: a required PR
    reconciliation step should outrank the canary install as the immediate next
    action, while the patham9-admitted canary remains the next admissible branch.
    """
    has_canary_evidence = "install_threadkeeper_canary_on_protomegabot" in action_evidence
    has_pr_reconciliation_evidence = "reconcile_threadkeeper_pr" in action_evidence
    if not (has_canary_evidence or has_pr_reconciliation_evidence):
        return None
    from goal_chainer.models import CandidateAction, Goal, GoalScenario, Norm

    goals: list[Any] = []
    if has_pr_reconciliation_evidence:
        goals.append(
            Goal(
                id="preserve_sequence_integrity",
                owner="project-control",
                statement="Respect ordering constraints before live deployment.",
                weight=0.95,
                kind="collective",
                required=True,
            )
        )
    goals.extend(
        [
            Goal(
                id="advance_bounded_canary",
                owner="protobots",
                statement="Advance the approved bounded ThreadKeeper canary path.",
                weight=0.90,
                kind="collective",
                required=True,
            ),
            Goal(
                id="preserve_agent_safety",
                owner="protomegabot",
                statement="Keep rollback, supervision, and stop conditions explicit.",
                weight=0.90,
                kind="individual",
                required=True,
            ),
        ]
    )

    actions: list[Any] = []
    if has_pr_reconciliation_evidence:
        actions.append(
            CandidateAction(
                id="reconcile_threadkeeper_pr",
                label="Reconcile ThreadKeeper PR",
                description="Reconcile the pending ThreadKeeper PR before live canary installation.",
                satisfies=("preserve_sequence_integrity", "advance_bounded_canary", "preserve_agent_safety"),
                evidence_query="(: $prf (Requires ThreadKeeperPRReconciliation) $tv)",
                evidence_atoms=(),
                default_strength=0.85,
                default_confidence=0.80,
            )
        )
    actions.extend(
        [
            CandidateAction(
                id="install_threadkeeper_canary_on_protomegabot",
                label="Install ThreadKeeper canary on ProtomegaBot",
                description="Run a supervised ThreadKeeper canary with rollback and stop conditions.",
                satisfies=("advance_bounded_canary", "preserve_agent_safety"),
                evidence_query="(: $prf (Acceptable install_threadkeeper_canary_on_protomegabot) $tv)",
                evidence_atoms=(),
                default_strength=0.70,
                default_confidence=0.75,
            ),
            CandidateAction(
                id="defer_until_more_synthetic_tests",
                label="Defer for more synthetic tests",
                description="Delay live canary work and run more synthetic-only checks.",
                satisfies=("preserve_agent_safety",),
                evidence_query="(: $prf (Acceptable defer_until_more_synthetic_tests) $tv)",
                evidence_atoms=(),
                default_strength=0.42,
                default_confidence=0.60,
            ),
            CandidateAction(
                id="ask_ben_again",
                label="Ask Ben again",
                description="Ask for approval again despite existing approval evidence.",
                satisfies=(),
                evidence_query="(: $prf (NotApplicable ask_ben_again) $tv)",
                evidence_atoms=(),
                default_strength=0.30,
                default_confidence=0.70,
            ),
            CandidateAction(
                id="remove_threadkeeper",
                label="Remove ThreadKeeper",
                description="Remove ThreadKeeper before a canary failure has occurred.",
                satisfies=("preserve_agent_safety",),
                evidence_query="(: $prf (NotApplicable remove_threadkeeper) $tv)",
                evidence_atoms=(),
                default_strength=0.28,
                default_confidence=0.70,
            ),
        ]
    )

    norms: list[Any] = []
    if has_pr_reconciliation_evidence:
        norms.append(
            Norm(
                id="pr-before-canary",
                mode="oblige",
                target_action="reconcile_threadkeeper_pr",
                priority=20,
                reason="PR reconciliation is pending and should precede the live canary.",
            )
        )
    norms.append(
        Norm(
            id="approved-bounded-canary",
            mode="permit",
            target_action="install_threadkeeper_canary_on_protomegabot",
            priority=12,
            reason="Ben approved the canary and patham9/PLN admitted it under bounded-risk evidence.",
        )
    )

    notes = (
        (
            "Patham9/PLN admitted the canary acceptability branch, but the pending PR reconciliation remains the immediate ordering constraint."
            if has_pr_reconciliation_evidence
            else "Patham9/PLN admitted the canary acceptability branch; no PR-reconciliation requirement was present in this handoff."
        ),
        "Guardrails required before the canary: rollback path, supervised log, stop conditions for runaway loops, persona/session corruption, private bridge activation, and queue-claim errors.",
        "Asking Ben again is held because approval evidence is already present; removing ThreadKeeper is held unless canary failure evidence appears.",
    )
    return GoalScenario(
        title="ThreadKeeper project-control canary decision",
        goals=tuple(goals),
        norms=tuple(norms),
        actions=tuple(actions),
        notes=notes,
    )


def _goalchainer_action_id(raw: str) -> str:
    return {
        "ThreadKeeperPRReconciliation": "reconcile_threadkeeper_pr",
    }.get(raw, raw)


def _default_action_evidence(action: Any) -> dict[str, Any]:
    strength = float(action.default_strength)
    confidence = float(action.default_confidence)
    return {
        "action_id": action.id,
        "deontic": _default_deontic(action.id),
        "expectation": _expectation(strength, confidence),
        "strength": strength,
        "confidence": confidence,
        "opinion": _sl_opinion(strength, confidence),
        "projection": f"scenario default for {action.id}",
        "proofs": ["GoalChainer scenario default; no matching PeTTa-memory handoff item"],
    }


def _default_deontic(action_id: str) -> str:
    return {
        "publish_raw_log": "forbidden",
        "publish_redacted_summary": "obligated",
        "hold_external_update": "permitted",
    }.get(action_id, "unregulated")


def _bounded_float(text: str, label: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"GoalChainer precompiled {label} outside [0,1]: {value}")
    return value


def _non_negative_float(text: str, label: str) -> float:
    value = float(text)
    if value < 0.0:
        raise ValidationError(f"GoalChainer precompiled {label} is negative: {value}")
    return value


def _expectation(strength: float, confidence: float) -> float:
    return round(confidence * (strength - 0.5) + 0.5, 6)


def _sl_opinion(strength: float, confidence: float) -> dict[str, float]:
    return {
        "b": round(confidence * strength, 4),
        "d": round(confidence * (1.0 - strength), 4),
        "u": round(1.0 - confidence, 4),
        "a": 0.5,
    }


def _validate_goalchainer_payload(payload: dict[str, Any]) -> None:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValidationError("GoalChainer payload missing decisions")
    if not all(isinstance(item, dict) and item.get("action_id") for item in decisions):
        raise ValidationError("GoalChainer decisions are missing action ids")
    if not any(item.get("status") == "recommended" for item in decisions if isinstance(item, dict)):
        raise ValidationError("GoalChainer payload has no recommended action")
    if not isinstance(payload.get("explanation"), list) or not payload["explanation"]:
        raise ValidationError("GoalChainer payload missing explanation")
    if "claim" in payload or "directive" in payload or "executed" in payload:
        raise ValidationError("GoalChainer smoke payload crossed into directive/execution output")
