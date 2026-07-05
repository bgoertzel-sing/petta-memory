from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .sexpr import SExpressionSyntaxError, parse_one_list, to_source

PASSED_TRUE_RE = re.compile(r"\(Passed:\s*(?:#t|True|true)\)")
PASSED_FALSE_RE = re.compile(r"\(Passed:\s*(?:#f|False|false)\)")
ERROR_RE = re.compile(r"\(Error\b|Exception caught|Traceback \(most recent call last\)")


def parse_metta_test_output(text: str) -> dict[str, Any]:
    """Summarize semantic MeTTa test markers from patham9/PLN output.

    The Hyperon/MeTTa CLI can exit with status 0 even when a `(Test ...)` form
    reports `(Passed: #f)` or an `(Error ...)` atom is printed.  This parser is
    intentionally text-level and conservative so smoke gates do not mistake a
    shell-successful semantic failure for a pass.
    """
    passed_true_count = len(PASSED_TRUE_RE.findall(text))
    passed_false_count = len(PASSED_FALSE_RE.findall(text))
    error_markers = len(ERROR_RE.findall(text))
    diagnostic_lines = [
        line.strip()
        for line in text.splitlines()
        if "Passed:" in line or "(Error" in line or "Exception caught" in line
    ]
    return {
        "passed_true_count": passed_true_count,
        "passed_false_count": passed_false_count,
        "error_markers": error_markers,
        "diagnostic_lines": diagnostic_lines,
        "semantic_passed": passed_true_count > 0 and passed_false_count == 0 and error_markers == 0,
    }


def classify_smoke_result(result: dict[str, Any]) -> dict[str, Any]:
    """Classify one patham9/PLN smoke result record.

    Expected input may come from a live runner or a saved artifact.  When an
    artifact has already counted `Passed:` markers, those counts are trusted;
    otherwise callers can pass a `stdout`, `stderr`, or `output` field.
    """
    output_text = "\n".join(str(result.get(key, "")) for key in ("stdout", "stderr", "output"))
    parsed = parse_metta_test_output(output_text) if output_text.strip() else {}
    true_count = int(result.get("passed_true_count", parsed.get("passed_true_count", 0)) or 0)
    false_count = int(result.get("passed_false_count", parsed.get("passed_false_count", 0)) or 0)
    error_markers = int(result.get("error_markers", parsed.get("error_markers", 0)) or 0)
    returncode = result.get("returncode")
    shell_ok = returncode in (0, None)
    semantic_ok = true_count > 0 and false_count == 0 and error_markers == 0
    status = "passed" if shell_ok and semantic_ok else "failed"
    reasons: list[str] = []
    if not shell_ok:
        reasons.append(f"nonzero returncode {returncode}")
    if true_count == 0:
        reasons.append("no Passed: #t markers")
    if false_count:
        reasons.append(f"{false_count} Passed: #f marker(s)")
    if error_markers:
        reasons.append(f"{error_markers} error marker(s)")
    return {
        "test": result.get("test"),
        "status": status,
        "returncode": returncode,
        "passed_true_count": true_count,
        "passed_false_count": false_count,
        "error_markers": error_markers,
        "reasons": reasons,
        "log": result.get("log"),
    }


def _retry_log_path(log_path: str | Path) -> Path:
    path = Path(log_path)
    return path.with_name(f"{path.stem}.retry{path.suffix}")


def classify_smoke_result_with_retry(result: dict[str, Any]) -> dict[str, Any]:
    """Classify a result and, if present, its explicit retry log.

    The first patham9/PLN ruletest run can fail from module resolution when run
    outside the PLN checkout, while a retry from the correct checkout can pass.
    Keeping both classifications preserves the original failure provenance while
    distinguishing harness/environment drift from semantic PLN regressions.
    """
    primary = classify_smoke_result(result)
    primary["attempt"] = "primary"
    retry_path_value = result.get("retry_log")
    if not retry_path_value and primary["status"] != "passed" and result.get("log"):
        candidate = _retry_log_path(str(result["log"]))
        if candidate.exists():
            retry_path_value = str(candidate)
    if not retry_path_value:
        return primary
    retry_path = Path(str(retry_path_value))
    if not retry_path.exists():
        primary["retry_missing"] = str(retry_path)
        return primary
    parsed = parse_metta_test_output(retry_path.read_text(encoding="utf-8", errors="replace"))
    retry = classify_smoke_result(
        {
            "test": result.get("test"),
            "returncode": result.get("retry_returncode", 0),
            "passed_true_count": parsed["passed_true_count"],
            "passed_false_count": parsed["passed_false_count"],
            "error_markers": parsed["error_markers"],
            "log": str(retry_path),
        }
    )
    retry["attempt"] = "retry"
    retry["primary_status"] = primary["status"]
    retry["primary_reasons"] = primary["reasons"]
    retry["classification"] = "harness-or-environment-drift" if retry["status"] == "passed" else "semantic-or-runtime-failure"
    return retry


def summarize_smoke_results(results: list[dict[str, Any]], *, include_retries: bool = False) -> dict[str, Any]:
    classifier = classify_smoke_result_with_retry if include_retries else classify_smoke_result
    classified = [classifier(result) for result in results]
    passed = [item for item in classified if item["status"] == "passed"]
    failed = [item for item in classified if item["status"] != "passed"]
    return {
        "status": "passed" if not failed and classified else "failed",
        "total": len(classified),
        "passed": len(passed),
        "failed": len(failed),
        "results": classified,
        "gate": "shell returncode plus semantic Passed markers; Passed: #f and Error atoms are failures",
        "retry_policy": "failed primary records may be reclassified from explicit .retry.log files" if include_retries else "primary records only",
    }


def summarize_smoke_results_file(path: str | Path, *, include_retries: bool = False) -> dict[str, Any]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("patham9/PLN results artifact must contain a JSON list")
    return summarize_smoke_results(records, include_retries=include_retries)


def _parse_pettachainer_stv_statement(atom: str) -> tuple[str, str, str, str]:
    """Return belief id, statement term, strength, confidence from `(: id term (STV s c))`."""
    try:
        expr = parse_one_list(atom)
    except SExpressionSyntaxError as exc:
        raise ValueError(f"invalid PeTTaChainer STV statement atom: {exc}") from exc
    if len(expr) != 4 or expr[0] != ":" or not isinstance(expr[1], str):
        raise ValueError("expected PeTTaChainer STV statement shaped as (: belief-id statement (STV strength confidence))")
    tv = expr[3]
    if not isinstance(tv, tuple) or len(tv) != 3 or tv[0] != "STV" or not isinstance(tv[1], str) or not isinstance(tv[2], str):
        raise ValueError("expected PeTTaChainer STV truth value shaped as (STV strength confidence)")
    return expr[1], to_source(expr[2]), tv[1], tv[2]


def _parse_evidence_packet(atom: str) -> dict[str, str]:
    """Parse the petta-memory EvidencePacket subset used in handoff caches."""
    try:
        expr = parse_one_list(atom)
    except SExpressionSyntaxError as exc:
        raise ValueError(f"invalid EvidencePacket atom: {exc}") from exc
    if len(expr) != 5 or expr[0] != "EvidencePacket" or not isinstance(expr[4], str):
        raise ValueError("expected EvidencePacket shaped as (EvidencePacket statement (EC support opposition) metadata promotion-event)")
    ec = expr[2]
    if not isinstance(ec, tuple) or len(ec) != 3 or ec[0] != "EC" or not isinstance(ec[1], str) or not isinstance(ec[2], str):
        raise ValueError("expected EvidencePacket EC counts shaped as (EC support opposition)")
    return {
        "statement": to_source(expr[1]),
        "support": ec[1],
        "opposition": ec[2],
        "metadata": to_source(expr[3]),
        "promotion_event": expr[4],
    }


def _numeric_stamp(index: int) -> str:
    return f"({index})"


def patham9_pln_query_smoke_program(handoff: dict[str, Any], *, item_index: int = 0) -> dict[str, Any]:
    """Build a tiny patham9/PLN query smoke from generated handoff Sentences.

    patham9/PLN's current stamp utilities expect sortable evidence stamps; rich
    symbolic provenance such as `(PMEvidence ...)` can trip lower-level sorting
    code.  This gate therefore loads the selected Sentence with a numeric runtime
    stamp while preserving the full petta-memory evidence/provenance mapping in
    the returned sidecar metadata.
    """
    if handoff.get("schema") != "petta-memory-patham9-pln-handoff-v1":
        raise ValueError("expected petta-memory-patham9-pln-handoff-v1 handoff")
    items = list(handoff.get("items", []))
    if not items:
        raise ValueError("patham9/PLN handoff has no Sentence items")
    if item_index < 0 or item_index >= len(items):
        raise ValueError(f"item_index {item_index} out of range for {len(items)} item(s)")
    item = items[item_index]
    term = str(item["term"])
    strength = str(item["stv"]["strength"])
    confidence = str(item["stv"]["confidence"])
    runtime_stamp = _numeric_stamp(item_index)
    runtime_sentence = f"(Sentence ({term} (stv {strength} {confidence})) {runtime_stamp})"
    expected = f"((stv {strength} {confidence}) {runtime_stamp})"
    source_evidence_id = str(item.get("evidence_id", ""))
    program = "\n".join(
        [
            "!(import! &self PLN)",
            "!(PLN.Init ())",
            f"!(Test (PLN.Query ({runtime_sentence})",
            f"                  {term}",
            "                  1 3 5)",
            f"       {expected})",
            "",
        ]
    )
    return {
        "schema": "petta-memory-patham9-pln-query-smoke-program-v1",
        "mode": "read-only-single-sentence-query-smoke",
        "program": program,
        "query_term": term,
        "expected_result": expected,
        "runtime_sentence": runtime_sentence,
        "runtime_stamp": runtime_stamp,
        "runtime_stamp_policy": "numeric patham9/PLN stamp used for chainer compatibility; source evidence preserved in sidecar",
        "source_evidence_id": source_evidence_id,
        "source_item": item,
        "boundary": "loads a generated Sentence into local patham9/PLN only for a bounded query smoke; no memory append, no PLN.Derive result promotion, no OmegaClaw/GoalChainer live path",
    }


def run_patham9_pln_query_smoke(
    handoff: dict[str, Any],
    *,
    pln_repo: str | Path,
    env_script: str | Path | None = None,
    timeout_sec: float = 30.0,
    item_index: int = 0,
) -> dict[str, Any]:
    """Run the tiny patham9/PLN handoff query smoke in an isolated temp file."""
    program = patham9_pln_query_smoke_program(handoff, item_index=item_index)
    repo = Path(pln_repo)
    if env_script is None:
        env_script = repo.parents[1] / "local" / "pettachainer-env.sh"
    env_path = Path(env_script)
    try:
        with tempfile.TemporaryDirectory(prefix="petta-patham9-pln-") as td:
            metta_path = Path(td) / "handoff_query_smoke.metta"
            metta_path.write_text(program["program"], encoding="utf-8")
            command = (
                "set -euo pipefail; "
                f"source {str(env_path)!r}; "
                f"cd {str(repo)!r}; "
                f"../PeTTa/run.sh {str(metta_path)!r}"
            )
            completed = subprocess.run(
                ["bash", "-lc", command],
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                env=os.environ.copy(),
            )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = f"{stderr}\nTimeoutExpired after {timeout_sec}s".strip()
    output = f"{stdout}\n{stderr}"
    parsed = parse_metta_test_output(output)
    classified = classify_smoke_result(
        {
            "test": "patham9-pln-handoff-query-smoke",
            "returncode": completed.returncode,
            "output": output,
        }
    )
    return {
        "schema": "petta-memory-patham9-pln-query-smoke-result-v1",
        "status": classified["status"],
        "classification": classified,
        "semantic_markers": parsed,
        "program": program,
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def patham9_pln_handoff_sentences(handoff_cache: dict[str, Any]) -> dict[str, Any]:
    """Map petta-memory's non-live handoff cache into patham9/PLN Sentence inputs.

    This is an artifact-level bridge, not a live PLN runner.  It preserves the
    existing STV truth values as `Sentence` atoms and attaches π-PLN extension
    metadata for contextual EvidencePacket/EC/provenance handling so later work
    can implement truth-value formulas without losing source boundaries.
    """
    if handoff_cache.get("schema") != "petta-memory-pettachainer-handoff-v1":
        raise ValueError("expected petta-memory-pettachainer-handoff-v1 cache")
    packets_by_statement: dict[str, list[dict[str, str]]] = {}
    for item in handoff_cache.get("items", []):
        if item.get("kind") != "pettachainer-evidence-packet":
            continue
        packet = _parse_evidence_packet(str(item.get("atom", "")))
        packet["belief_id"] = str(item.get("belief_id", ""))
        packet["cluster_id"] = str(item.get("cluster_id", ""))
        packet["promotion_rule"] = str(item.get("promotion_rule", ""))
        packet["promotion_domain"] = str(item.get("promotion_domain", ""))
        packets_by_statement.setdefault(packet["statement"], []).append(packet)

    sentences: list[dict[str, Any]] = []
    for item in handoff_cache.get("items", []):
        if item.get("kind") != "pettachainer-stv-statement":
            continue
        belief_id, term, strength, confidence = _parse_pettachainer_stv_statement(str(item.get("atom", "")))
        evidence_id = (
            f"(PMEvidence {belief_id} {item.get('cluster_id')} {item.get('promotion_event')} "
            f"{item.get('promotion_rule')} {item.get('promotion_domain')})"
        )
        sentence_atom = f"(Sentence {term} (stv {strength} {confidence}) ({evidence_id}))"
        packets = packets_by_statement.get(term, [])
        sentences.append(
            {
                "kind": "patham9-pln-sentence-input",
                "atom": sentence_atom,
                "term": term,
                "stv": {"strength": strength, "confidence": confidence},
                "evidence_id": evidence_id,
                "belief_id": belief_id,
                "cluster_id": item.get("cluster_id"),
                "promotion_event": item.get("promotion_event"),
                "promotion_rule": item.get("promotion_rule"),
                "promotion_domain": item.get("promotion_domain"),
                "source_status": item.get("item_status"),
                "pi_pln_extension": {
                    "contextual_evidence_packets": packets,
                    "ec_projection_policy": "preserve packets first; later project EC support/opposition through reviewed pi-PLN truth-value formulas",
                    "context_selection": "not-run; no generated contexts in this handoff gate",
                },
            }
        )
    return {
        "schema": "petta-memory-patham9-pln-handoff-v1",
        "source_schema": handoff_cache.get("schema"),
        "source_cache_id": handoff_cache.get("cache_id"),
        "mode": "non-live-patham9-pln-sentence-handoff",
        "boundary": "read-only PLN input artifact; not an inferred belief, not appended to memory, and no PLN.Query/Derive run here",
        "sentence_format": "(Sentence $Term (stv S C) ($EvidenceID))",
        "item_count": len(sentences),
        "items": sentences,
    }
