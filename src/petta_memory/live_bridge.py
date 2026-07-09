from __future__ import annotations

from pathlib import Path
from typing import Any

from .goalchainer_smoke import DEFAULT_GOALCHAINER_REPO, DEFAULT_REQUEST, run_goalchainer_precompiled_handoff_smoke
from .patham9_pln import patham9_pln_handoff_sentences, ranked_inference_control_plan, ranked_plan_admitted_handoff
from .store import MediumMemoryStore, ValidationError


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
    goalchainer_result = run_goalchainer_precompiled_handoff_smoke(
        goalchainer_cache,
        goalchainer_repo=goalchainer_repo,
        request=request,
        include_heuristic_memory_probe=include_heuristic_memory_probe,
    )

    decision_payload = goalchainer_result["decision_payload"]
    decisions = decision_payload.get("decisions", []) if isinstance(decision_payload, dict) else []
    recommended = next((item for item in decisions if item.get("status") == "recommended"), None)
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
        "goalchainer_gate": {
            "schema": goalchainer_result["schema"],
            "mode": goalchainer_result["mode"],
            "recommended_action": recommended.get("action_id") if isinstance(recommended, dict) else None,
            "recommended_status": recommended.get("status") if isinstance(recommended, dict) else None,
            "heuristic_memory_probe": goalchainer_result.get("heuristic_memory_probe"),
            "checks": goalchainer_result["checks"],
            "boundary": goalchainer_result["boundary"],
        },
        "checks": {
            "journal_read": True,
            "promoted_memory_evidence_present": goalchainer_cache["item_count"] > 0,
            "ranked_plan_built": ranked_plan["candidate_count"] > 0,
            "admitted_handoff_built": admitted["admitted_handoff"]["item_count"] == ranked_plan["recommended_count"],
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
