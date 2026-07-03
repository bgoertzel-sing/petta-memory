"""Non-live GoalChainer smoke gate for PeTTa memory handoff artifacts.

This module deliberately runs GoalChainer as an external, bounded subprocess and
validates only a decision payload. It does not load an OmegaClaw skill, claim a
task, or write back to memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
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
