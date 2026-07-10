from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .goalchainer_smoke import DEFAULT_GOALCHAINER_REPO, DEFAULT_REQUEST, run_goalchainer_precompiled_handoff_smoke
from .patham9_pln import (
    patham9_pln_handoff_sentences,
    ranked_inference_control_plan,
    ranked_plan_admitted_handoff,
    run_patham9_pln_multi_sentence_derivation_smoke,
)
from .store import MediumMemoryStore, ValidationError


DEFAULT_PLN_REPO = Path(__file__).resolve().parents[3] / "patham9-pln"


def run_petta_memory_goalchainer_live_bridge(
    journal_path: str | Path,
    *,
    cache_id: str = "petta-memory-goalchainer-live-bridge",
    goalchainer_repo: str | Path = DEFAULT_GOALCHAINER_REPO,
    request: str = DEFAULT_REQUEST,
    query_target: str = "",
    max_branches: int = 20,
    seed: int | None = 17,
    min_estimated_probability: float = 0.0,
    require_query_relevance: bool = False,
    include_heuristic_memory_probe: bool = True,
    include_patham9_runtime: bool = False,
    pln_repo: str | Path = DEFAULT_PLN_REPO,
    patham9_timeout_sec: float = 30.0,
    patham9_runner: Callable[..., dict[str, Any]] = run_patham9_pln_multi_sentence_derivation_smoke,
    goalchainer_runner: Callable[..., dict[str, Any]] = run_goalchainer_precompiled_handoff_smoke,
) -> dict[str, Any]:
    """Run the first read-only live bridge from a journal into GoalChainer.

    "Live" here means the bridge consumes an actual ``MediumMemoryStore``
    journal path and invokes the local GoalChainer decision pipeline over the
    promoted evidence exported from that journal.  It remains a reviewed bridge,
    not an autonomous runtime loop: no OmegaClaw skill is loaded, no task or
    directive claim is accepted, no journal write is made, and no derived result
    is promoted back into memory.
    """
    journal = Path(journal_path)
    store = MediumMemoryStore(journal)
    clusters = store.clusters()
    if not clusters:
        raise ValidationError(f"live bridge needs a non-empty journal: {journal}")

    pettachainer_cache = store.pettachainer_handoff_cache(cache_id=f"{cache_id}-pettachainer")
    goalchainer_cache = store.goalchainer_handoff_cache(cache_id=f"{cache_id}-goalchainer")
    patham9_handoff = patham9_pln_handoff_sentences(pettachainer_cache)
    ranked_plan = ranked_inference_control_plan(
        patham9_handoff,
        query_target=query_target,
        max_branches=max_branches,
        seed=seed,
        min_estimated_probability=min_estimated_probability,
        require_query_relevance=require_query_relevance,
    )
    admitted = ranked_plan_admitted_handoff(patham9_handoff, ranked_plan)
    patham9_runtime_gate: dict[str, Any]
    if include_patham9_runtime:
        admitted_handoff = admitted["admitted_handoff"]
        if admitted_handoff["item_count"] <= 0:
            raise ValidationError("patham9 runtime gate needs at least one admitted handoff item")
        patham9_runtime_result = patham9_runner(
            admitted_handoff,
            pln_repo=pln_repo,
            timeout_sec=patham9_timeout_sec,
        )
        if not isinstance(patham9_runtime_result, dict):
            raise ValidationError(
                "patham9 runtime gate returned a non-object result; refusing GoalChainer appraisal"
            )
        runtime_schema = patham9_runtime_result.get("schema")
        if not isinstance(runtime_schema, str) or not runtime_schema:
            raise ValidationError(
                "patham9 runtime gate returned non-string schema; refusing GoalChainer appraisal"
            )
        runtime_returncode = patham9_runtime_result.get("returncode")
        semantic_markers = patham9_runtime_result.get("semantic_markers")
        if not isinstance(semantic_markers, dict):
            raise ValidationError(
                "patham9 runtime gate returned non-object semantic_markers; refusing GoalChainer appraisal"
            )
        if patham9_runtime_result.get("status") != "passed":
            raise ValidationError(
                "patham9 runtime gate did not pass; refusing GoalChainer appraisal "
                f"(status={patham9_runtime_result.get('status')!r}, returncode={runtime_returncode!r})"
            )
        if isinstance(runtime_returncode, bool) or runtime_returncode != 0:
            raise ValidationError(
                "patham9 runtime gate returned nonzero or non-integer returncode; refusing GoalChainer appraisal "
                f"(status={patham9_runtime_result.get('status')!r}, returncode={runtime_returncode!r})"
            )
        if semantic_markers.get("semantic_passed") is not True:
            raise ValidationError(
                "patham9 runtime gate semantic markers did not pass; refusing GoalChainer appraisal"
            )
        program = patham9_runtime_result.get("program")
        if not isinstance(program, dict):
            raise ValidationError(
                "patham9 runtime gate returned a non-object program artifact; refusing GoalChainer appraisal"
            )
        program_schema = program.get("schema")
        if not isinstance(program_schema, str) or not program_schema:
            raise ValidationError(
                "patham9 runtime gate returned non-string program schema; refusing GoalChainer appraisal"
            )
        patham9_runtime_gate = {
            "enabled": True,
            "schema": runtime_schema,
            "status": patham9_runtime_result.get("status"),
            "returncode": runtime_returncode,
            "semantic_markers": semantic_markers,
            "program_schema": program_schema,
            "boundary": (
                "bounded local patham9/PLN runtime over the admitted handoff; no PeTTaChainer compileadd; "
                "fail-closed before GoalChainer appraisal if the runtime gate does not pass; "
                "no memory append; no inferred-belief promotion; no OmegaClaw skill or task/directive claim"
            ),
        }
    else:
        patham9_runtime_gate = {
            "enabled": False,
            "status": "skipped",
            "boundary": "patham9/PLN runtime gate skipped by caller; ranked/admitted pi-PLN handoff still built",
        }
    goalchainer_result = goalchainer_runner(
        goalchainer_cache,
        goalchainer_repo=goalchainer_repo,
        request=request,
        include_heuristic_memory_probe=include_heuristic_memory_probe,
        admitted_patham9_handoff=admitted["admitted_handoff"],
    )
    if not isinstance(goalchainer_result, dict):
        raise ValidationError("GoalChainer gate returned a non-object result; refusing live bridge output")
    decision_payload = goalchainer_result.get("decision_payload")
    if not isinstance(decision_payload, dict):
        raise ValidationError("GoalChainer gate returned a non-object decision_payload; refusing live bridge output")
    checks = goalchainer_result.get("checks")
    if not isinstance(checks, dict):
        raise ValidationError("GoalChainer gate returned non-object checks; refusing live bridge output")
    if checks.get("no_memory_write") is not True:
        raise ValidationError("GoalChainer gate did not assert no_memory_write; refusing live bridge output")
    if checks.get("no_live_directive_or_task_claim") is not True:
        raise ValidationError(
            "GoalChainer gate did not assert no_live_directive_or_task_claim; refusing live bridge output"
        )
    goalchainer_schema = goalchainer_result.get("schema")
    if not isinstance(goalchainer_schema, str) or not goalchainer_schema:
        raise ValidationError("GoalChainer gate returned non-string schema; refusing live bridge output")
    goalchainer_mode = goalchainer_result.get("mode")
    if not isinstance(goalchainer_mode, str) or not goalchainer_mode:
        raise ValidationError("GoalChainer gate returned non-string mode; refusing live bridge output")
    goalchainer_boundary = goalchainer_result.get("boundary")
    if not isinstance(goalchainer_boundary, str) or not goalchainer_boundary:
        raise ValidationError("GoalChainer gate returned non-string boundary; refusing live bridge output")
    heuristic_memory_probe = goalchainer_result.get("heuristic_memory_probe")
    if heuristic_memory_probe is not None:
        if not isinstance(heuristic_memory_probe, dict):
            raise ValidationError(
                "GoalChainer gate returned non-object heuristic_memory_probe; refusing live bridge output"
            )
        for field in ("schema", "mode", "boundary"):
            value = heuristic_memory_probe.get(field)
            if not isinstance(value, str) or not value:
                raise ValidationError(
                    "GoalChainer gate returned heuristic_memory_probe with non-string metadata; "
                    "refusing live bridge output"
                )
        if heuristic_memory_probe.get("memory_proof_present") is not True:
            raise ValidationError(
                "GoalChainer gate heuristic_memory_probe did not confirm memory proof; refusing live bridge output"
            )
        if heuristic_memory_probe.get("leak_check_safe") is not True:
            raise ValidationError(
                "GoalChainer gate heuristic_memory_probe did not confirm leak_check_safe; refusing live bridge output"
            )

    decisions = decision_payload.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValidationError("GoalChainer gate returned non-list decisions; refusing live bridge output")
    if any(not isinstance(item, dict) for item in decisions):
        raise ValidationError("GoalChainer gate returned non-object decision entries; refusing live bridge output")
    notes = decision_payload.get("notes", [])
    if not isinstance(notes, list):
        raise ValidationError("GoalChainer gate returned non-list notes; refusing live bridge output")
    if any(not isinstance(note, str) or not note for note in notes):
        raise ValidationError("GoalChainer gate returned malformed note entry; refusing live bridge output")
    allowed_decision_statuses = {"recommended", "candidate", "held", "weak", "blocked"}
    decision_action_ids: set[str] = set()
    for decision in decisions:
        status = decision.get("status")
        if not isinstance(status, str) or not status:
            raise ValidationError(
                "GoalChainer gate returned decision with malformed status; refusing live bridge output"
            )
        if status not in allowed_decision_statuses:
            raise ValidationError(
                "GoalChainer gate returned decision with unknown status; refusing live bridge output"
            )
        evidence = decision.get("evidence")
        if evidence is not None:
            if not isinstance(evidence, dict):
                raise ValidationError(
                    "GoalChainer gate returned decision with non-object evidence; refusing live bridge output"
                )
            proofs = evidence.get("proofs")
            if proofs is not None:
                if not isinstance(proofs, list):
                    raise ValidationError(
                        "GoalChainer gate returned decision evidence with non-list proofs; refusing live bridge output"
                    )
                if any(not isinstance(proof, str) or not proof for proof in proofs):
                    raise ValidationError(
                        "GoalChainer gate returned decision evidence with malformed proof entry; "
                        "refusing live bridge output"
                    )
            contextual_evidence = evidence.get("contextual_evidence")
            if contextual_evidence is not None:
                if not isinstance(contextual_evidence, list):
                    raise ValidationError(
                        "GoalChainer gate returned decision evidence with non-list contextual_evidence; "
                        "refusing live bridge output"
                    )
                if any(not isinstance(item, dict) for item in contextual_evidence):
                    raise ValidationError(
                        "GoalChainer gate returned decision evidence with non-object contextual_evidence entry; "
                        "refusing live bridge output"
                    )
        if "action_id" not in decision:
            continue
        action_id = decision.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise ValidationError(
                "GoalChainer gate returned decision with malformed action_id; refusing live bridge output"
            )
        if action_id in decision_action_ids:
            raise ValidationError(
                "GoalChainer gate returned duplicate decision action_id; refusing live bridge output"
            )
        decision_action_ids.add(action_id)
    recommended_decisions = [item for item in decisions if item.get("status") == "recommended"]
    if len(recommended_decisions) > 1:
        raise ValidationError("GoalChainer gate returned multiple recommended decisions; refusing live bridge output")
    recommended = recommended_decisions[0] if recommended_decisions else None
    if recommended is not None:
        recommended_action = recommended.get("action_id")
        if not isinstance(recommended_action, str) or not recommended_action:
            raise ValidationError(
                "GoalChainer gate returned recommended decision without non-empty string action_id; "
                "refusing live bridge output"
            )
        recommended_status = recommended.get("status")
    else:
        recommended_action = None
        recommended_status = None
    return {
        "schema": "petta-memory-goalchainer-live-bridge-v1",
        "mode": "read-only-live-journal-to-local-goalchainer",
        "journal_path": str(journal),
        "cache_id": cache_id,
        "cluster_count": len(clusters),
        "input_counts": {
            "pettachainer_items": pettachainer_cache["item_count"],
            "goalchainer_items": goalchainer_cache["item_count"],
            "patham9_items": patham9_handoff["item_count"],
            "ranked_candidates": ranked_plan["candidate_count"],
            "admitted_items": admitted["admitted_handoff"]["item_count"],
        },
        "pi_pln_gate": {
            "schema": ranked_plan["schema"],
            "recommended_count": ranked_plan["recommended_count"],
            "held_count": ranked_plan["held_count"],
            "admitted_schema": admitted["schema"],
            "admitted_item_count": admitted["admitted_handoff"]["item_count"],
            "boundary": admitted["boundary"],
        },
        "patham9_runtime_gate": patham9_runtime_gate,
        "goalchainer_gate": {
            "schema": goalchainer_schema,
            "mode": goalchainer_mode,
            "recommended_action": recommended_action,
            "recommended_status": recommended_status,
            "decisions": decisions,
            "notes": notes,
            "heuristic_memory_probe": heuristic_memory_probe,
            "checks": checks,
            "boundary": goalchainer_boundary,
        },
        "checks": {
            "journal_read": True,
            "promoted_memory_evidence_present": goalchainer_cache["item_count"] > 0,
            "ranked_plan_built": ranked_plan["candidate_count"] > 0,
            "admitted_handoff_built": admitted["admitted_handoff"]["item_count"] == ranked_plan["recommended_count"],
            "patham9_runtime_passed_or_skipped": (
                (not include_patham9_runtime) or patham9_runtime_gate.get("status") == "passed"
            ),
            "goalchainer_recommended_action_present": isinstance(recommended, dict),
            "no_omegaclaw_skill_loaded": True,
            "no_task_or_directive_claim": True,
            "no_memory_write": True,
        },
        "boundary": (
            "read-only live bridge over an existing MediumMemoryStore journal; local GoalChainer decision/appraisal only; "
            "no OmegaClaw skill, no accepted directive/task claim, no memory write, no inferred-belief promotion"
        ),
    }
