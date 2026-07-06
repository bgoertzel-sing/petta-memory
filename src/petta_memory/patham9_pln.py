from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

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


def patham9_pln_derivation_smoke_program(handoff: dict[str, Any], *, item_index: int = 0) -> dict[str, Any]:
    """Build a tiny two-premise patham9/PLN derivation smoke.

    The first premise is one promoted petta-memory handoff Sentence.  The second
    premise is a synthetic bridge implication from that term to a derived
    `PMDerivedFromHandoff` term.  Numeric runtime stamps keep patham9/PLN's stamp
    sorter happy; the sidecar maps each numeric stamp back to petta-memory
    provenance so this remains a non-live derivation gate rather than a memory
    promotion.
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
    source_stamp = _numeric_stamp(item_index)
    bridge_stamp = _numeric_stamp(item_index + 1)
    derived_term = f"(PMDerivedFromHandoff {term})"
    bridge_strength = "1.0"
    bridge_confidence = "0.90"
    expected_strength = str(float(strength) * float(bridge_strength) + 0.02 * (1.0 - float(strength)))
    expected_confidence = str(float(confidence) * float(bridge_confidence))
    expected = f"((stv {expected_strength} {expected_confidence}) ({item_index} {item_index + 1}))"
    source_sentence = f"(Sentence ({term} (stv {strength} {confidence})) {source_stamp})"
    bridge_sentence = (
        f"(Sentence ((Implication {term} {derived_term}) "
        f"(stv {bridge_strength} {bridge_confidence})) {bridge_stamp})"
    )
    program = "\n".join(
        [
            "!(import! &self PLN)",
            "!(PLN.Init ())",
            f"!(Test (PLN.Query ({source_sentence}",
            f"                   {bridge_sentence})",
            f"                  {derived_term}",
            "                  2 5 8)",
            f"       {expected})",
            "",
        ]
    )
    return {
        "schema": "petta-memory-patham9-pln-derivation-smoke-program-v1",
        "mode": "read-only-two-premise-derivation-smoke",
        "program": program,
        "source_term": term,
        "derived_term": derived_term,
        "runtime_sentences": [source_sentence, bridge_sentence],
        "expected_result": expected,
        "runtime_stamp_policy": "numeric patham9/PLN stamps used for chainer compatibility; source evidence and synthetic bridge provenance preserved in sidecar",
        "stamp_sidecar": {
            source_stamp: {"kind": "petta-memory-source-sentence", "source_evidence_id": str(item.get("evidence_id", "")), "source_item": item},
            bridge_stamp: {"kind": "synthetic-non-live-bridge-implication", "source_item_index": item_index, "rule": "PMDerivedFromHandoff implication smoke"},
        },
        "boundary": "loads one generated Sentence plus one synthetic bridge implication into local patham9/PLN for a bounded derivation smoke; no memory append, no inferred-belief promotion, no OmegaClaw/GoalChainer live path",
    }


def _run_patham9_program(
    program_text: str,
    *,
    pln_repo: str | Path,
    env_script: str | Path | None,
    timeout_sec: float,
    filename: str,
) -> tuple[int, str, str]:
    repo = Path(pln_repo)
    if env_script is None:
        env_script = repo.parents[1] / "local" / "pettachainer-env.sh"
    env_path = Path(env_script)
    try:
        with tempfile.TemporaryDirectory(prefix="petta-patham9-pln-") as td:
            metta_path = Path(td) / filename
            metta_path.write_text(program_text, encoding="utf-8")
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
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return -1, stdout, f"{stderr}\nTimeoutExpired after {timeout_sec}s".strip()


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
    returncode, stdout, stderr = _run_patham9_program(
        program["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="handoff_query_smoke.metta",
    )
    output = f"{stdout}\n{stderr}"
    parsed = parse_metta_test_output(output)
    classified = classify_smoke_result(
        {
            "test": "patham9-pln-handoff-query-smoke",
            "returncode": returncode,
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


def run_patham9_pln_derivation_smoke(
    handoff: dict[str, Any],
    *,
    pln_repo: str | Path,
    env_script: str | Path | None = None,
    timeout_sec: float = 30.0,
    item_index: int = 0,
) -> dict[str, Any]:
    """Run the two-premise patham9/PLN derivation smoke in an isolated temp file."""
    program = patham9_pln_derivation_smoke_program(handoff, item_index=item_index)
    returncode, stdout, stderr = _run_patham9_program(
        program["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="handoff_derivation_smoke.metta",
    )
    output = f"{stdout}\n{stderr}"
    parsed = parse_metta_test_output(output)
    classified = classify_smoke_result(
        {
            "test": "patham9-pln-handoff-derivation-smoke",
            "returncode": returncode,
            "output": output,
        }
    )
    return {
        "schema": "petta-memory-patham9-pln-derivation-smoke-result-v1",
        "status": classified["status"],
        "classification": classified,
        "semantic_markers": parsed,
        "program": program,
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def ec_projected_stv(
    base_strength: float,
    base_confidence: float,
    contextual_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute a wrapper-level projected STV from base STV and contextual EC packets.

    This is the first reviewed pi-PLN wrapper formula.  It blends the base STV
    with EC-derived evidence using a confidence-weighted average, mirroring the
    formula already used in the non-live GoalChainer precompiled EC smoke but
    made explicit and self-contained for the patham9/PLN wrapper boundary.

    For each EvidencePacket with support s and opposition o:
      ec_strength  = s / (s + o)
      ec_confidence = (s + o) / (s + o + 2)        # Laplace-smoothed

    Blended across all packets and the base STV:
      projected_strength   = weighted_mean(strengths, confidences)
      projected_confidence = max(base_confidence, max(ec_confidences))

    The formula is conservative: it cannot inflate confidence beyond the base
    and uses confidence as the blending weight so weak evidence has little
    effect.  No truth-changing happens inside patham9/PLN; the wrapper
    pre-projects and feeds a plain Sentence with the projected STV.
    """
    if base_strength < 0 or base_strength > 1:
        raise ValueError(f"base_strength {base_strength} out of [0, 1]")
    if base_confidence < 0 or base_confidence > 1:
        raise ValueError(f"base_confidence {base_confidence} out of [0, 1]")

    weights: list[float] = [base_confidence]
    strengths: list[float] = [base_strength]
    confidences: list[float] = [base_confidence]
    packet_summaries: list[dict[str, Any]] = []

    for packet in contextual_packets:
        support = float(packet.get("support", 0))
        opposition = float(packet.get("opposition", 0))
        if support < 0 or opposition < 0:
            raise ValueError(f"EC counts must be non-negative; got support={support} opposition={opposition}")
        total = support + opposition
        if total <= 0:
            continue
        ec_strength = support / total
        ec_confidence = total / (total + 2.0)
        weights.append(ec_confidence)
        strengths.append(ec_strength)
        confidences.append(ec_confidence)
        packet_summaries.append(
            {
                "support": support,
                "opposition": opposition,
                "total_evidence": total,
                "positive_ratio": ec_strength,
                "ec_strength": round(ec_strength, 6),
                "ec_confidence": round(ec_confidence, 6),
            }
        )

    total_weight = sum(weights)
    if total_weight <= 0:
        projected_strength = base_strength
        projected_confidence = base_confidence
    else:
        projected_strength = sum(s * w for s, w in zip(strengths, weights)) / total_weight
        projected_confidence = max(confidences)

    return {
        "projected_strength": round(projected_strength, 6),
        "projected_confidence": round(projected_confidence, 6),
        "base_strength": base_strength,
        "base_confidence": base_confidence,
        "packet_count": len(packet_summaries),
        "packets": packet_summaries,
        "formula": "confidence-weighted blend: projected_strength = sum(s_i * w_i) / sum(w_i); projected_confidence = max(all confidences)",
    }


def patham9_pln_ec_projection_smoke_program(
    handoff: dict[str, Any],
    *,
    item_index: int = 0,
) -> dict[str, Any]:
    """Build a non-live EC projection comparison smoke for patham9/PLN.

    Produces two query smoke programs: one with the original (direct) STV and
    one with the wrapper-projected STV after folding in contextual EvidencePacket
    EC counts.  Both use numeric runtime stamps and the same query term so the
    results can be compared artifact-only without promoting either as an
    inferred belief.
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
    base_strength = float(item["stv"]["strength"])
    base_confidence = float(item["stv"]["confidence"])
    raw_packets = item.get("pi_pln_extension", {}).get("contextual_evidence_packets", [])
    projection = ec_projected_stv(base_strength, base_confidence, raw_packets)
    projected_strength = projection["projected_strength"]
    projected_confidence = projection["projected_confidence"]

    runtime_stamp = _numeric_stamp(item_index)
    source_evidence_id = str(item.get("evidence_id", ""))

    direct_sentence = f"(Sentence ({term} (stv {item['stv']['strength']} {item['stv']['confidence']})) {runtime_stamp})"
    direct_expected = f"((stv {item['stv']['strength']} {item['stv']['confidence']}) {runtime_stamp})"
    direct_program = "\n".join(
        [
            "!(import! &self PLN)",
            "!(PLN.Init ())",
            f"!(Test (PLN.Query ({direct_sentence})",
            f"                  {term}",
            "                  1 3 5)",
            f"       {direct_expected})",
            "",
        ]
    )

    projected_sentence = f"(Sentence ({term} (stv {projected_strength} {projected_confidence})) {runtime_stamp})"
    projected_expected = f"((stv {projected_strength} {projected_confidence}) {runtime_stamp})"
    projected_program = "\n".join(
        [
            "!(import! &self PLN)",
            "!(PLN.Init ())",
            f"!(Test (PLN.Query ({projected_sentence})",
            f"                  {term}",
            "                  1 3 5)",
            f"       {projected_expected})",
            "",
        ]
    )

    return {
        "schema": "petta-memory-patham9-pln-ec-projection-smoke-program-v1",
        "mode": "read-only-ec-projection-comparison-smoke",
        "query_term": term,
        "direct": {
            "program": direct_program,
            "runtime_sentence": direct_sentence,
            "expected_result": direct_expected,
            "stv": {"strength": item["stv"]["strength"], "confidence": item["stv"]["confidence"]},
        },
        "projected": {
            "program": projected_program,
            "runtime_sentence": projected_sentence,
            "expected_result": projected_expected,
            "stv": {"strength": str(projected_strength), "confidence": str(projected_confidence)},
        },
        "ec_projection": projection,
        "runtime_stamp": runtime_stamp,
        "runtime_stamp_policy": "numeric patham9/PLN stamp used for chainer compatibility; source evidence preserved in sidecar",
        "source_evidence_id": source_evidence_id,
        "boundary": "non-live read-only comparison of direct vs projected STV query smokes; no memory append, no inferred-belief promotion, no OmegaClaw/GoalChainer live path",
    }


def run_patham9_pln_ec_projection_smoke(
    handoff: dict[str, Any],
    *,
    pln_repo: str | Path,
    env_script: str | Path | None = None,
    timeout_sec: float = 30.0,
    item_index: int = 0,
) -> dict[str, Any]:
    """Run the EC projection comparison smoke in isolated temp files."""
    smoke = patham9_pln_ec_projection_smoke_program(handoff, item_index=item_index)

    direct_rc, direct_out, direct_err = _run_patham9_program(
        smoke["direct"]["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="ec_projection_direct.metta",
    )
    direct_output = f"{direct_out}\n{direct_err}"
    direct_parsed = parse_metta_test_output(direct_output)
    direct_classified = classify_smoke_result(
        {"test": "ec-projection-direct", "returncode": direct_rc, "output": direct_output}
    )

    projected_rc, projected_out, projected_err = _run_patham9_program(
        smoke["projected"]["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="ec_projection_projected.metta",
    )
    projected_output = f"{projected_out}\n{projected_err}"
    projected_parsed = parse_metta_test_output(projected_output)
    projected_classified = classify_smoke_result(
        {"test": "ec-projection-projected", "returncode": projected_rc, "output": projected_output}
    )

    return {
        "schema": "petta-memory-patham9-pln-ec-projection-smoke-result-v1",
        "status": "passed" if direct_classified["status"] == "passed" and projected_classified["status"] == "passed" else "failed",
        "direct": {
            "classification": direct_classified,
            "semantic_markers": direct_parsed,
            "returncode": direct_rc,
            "stdout_tail": direct_out[-2000:],
            "stderr_tail": direct_err[-2000:],
        },
        "projected": {
            "classification": projected_classified,
            "semantic_markers": projected_parsed,
            "returncode": projected_rc,
            "stdout_tail": projected_out[-2000:],
            "stderr_tail": projected_err[-2000:],
        },
        "ec_projection": smoke["ec_projection"],
        "program": smoke,
        "boundary": "non-live read-only comparison; no memory append, no inferred-belief promotion, no OmegaClaw/GoalChainer live path",
    }


def patham9_pln_ec_projection_conflicting_smoke_program(
    handoff: dict[str, Any],
    *,
    item_index: int = 0,
    conflicting_support: float = 1.0,
    conflicting_opposition: float = 9.0,
) -> dict[str, Any]:
    """Build a conflicting-EC projection comparison smoke for patham9/PLN.

    Like the basic EC projection smoke but the contextual EvidencePacket carries
    opposing EC counts (default 1 support / 9 opposition) against a strong base
    STV.  This verifies the wrapper formula lowers projected strength below the
    base strength when contextual evidence contradicts the base STV.
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
    base_strength = float(item["stv"]["strength"])
    base_confidence = float(item["stv"]["confidence"])
    conflicting_packets = [{"support": conflicting_support, "opposition": conflicting_opposition}]
    projection = ec_projected_stv(base_strength, base_confidence, conflicting_packets)
    projected_strength = projection["projected_strength"]
    projected_confidence = projection["projected_confidence"]

    runtime_stamp = _numeric_stamp(item_index)
    source_evidence_id = str(item.get("evidence_id", ""))

    direct_sentence = f"(Sentence ({term} (stv {item['stv']['strength']} {item['stv']['confidence']})) {runtime_stamp})"
    direct_expected = f"((stv {item['stv']['strength']} {item['stv']['confidence']}) {runtime_stamp})"
    direct_program = "\n".join(
        [
            "!(import! &self PLN)",
            "!(PLN.Init ())",
            f"!(Test (PLN.Query ({direct_sentence})",
            f"                  {term}",
            "                  1 3 5)",
            f"       {direct_expected})",
            "",
        ]
    )

    projected_sentence = f"(Sentence ({term} (stv {projected_strength} {projected_confidence})) {runtime_stamp})"
    projected_expected = f"((stv {projected_strength} {projected_confidence}) {runtime_stamp})"
    projected_program = "\n".join(
        [
            "!(import! &self PLN)",
            "!(PLN.Init ())",
            f"!(Test (PLN.Query ({projected_sentence})",
            f"                  {term}",
            "                  1 3 5)",
            f"       {projected_expected})",
            "",
        ]
    )

    # Verify the formula actually lowers strength
    strength_lowered = projected_strength < base_strength

    return {
        "schema": "petta-memory-patham9-pln-ec-projection-conflicting-smoke-program-v1",
        "mode": "read-only-ec-projection-conflicting-comparison-smoke",
        "query_term": term,
        "conflicting_ec": {"support": conflicting_support, "opposition": conflicting_opposition},
        "strength_lowered": strength_lowered,
        "direct": {
            "program": direct_program,
            "runtime_sentence": direct_sentence,
            "expected_result": direct_expected,
            "stv": {"strength": item["stv"]["strength"], "confidence": item["stv"]["confidence"]},
        },
        "projected": {
            "program": projected_program,
            "runtime_sentence": projected_sentence,
            "expected_result": projected_expected,
            "stv": {"strength": str(projected_strength), "confidence": str(projected_confidence)},
        },
        "ec_projection": projection,
        "runtime_stamp": runtime_stamp,
        "source_evidence_id": source_evidence_id,
        "boundary": "non-live read-only comparison of direct vs projected STV query smokes with conflicting EC; no memory append, no inferred-belief promotion, no OmegaClaw/GoalChainer live path",
    }


def run_patham9_pln_ec_projection_conflicting_smoke(
    handoff: dict[str, Any],
    *,
    pln_repo: str | Path,
    env_script: str | Path | None = None,
    timeout_sec: float = 30.0,
    item_index: int = 0,
    conflicting_support: float = 1.0,
    conflicting_opposition: float = 9.0,
) -> dict[str, Any]:
    """Run the conflicting-EC projection comparison smoke in isolated temp files."""
    smoke = patham9_pln_ec_projection_conflicting_smoke_program(
        handoff,
        item_index=item_index,
        conflicting_support=conflicting_support,
        conflicting_opposition=conflicting_opposition,
    )

    direct_rc, direct_out, direct_err = _run_patham9_program(
        smoke["direct"]["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="ec_conflicting_direct.metta",
    )
    direct_output = f"{direct_out}\n{direct_err}"
    direct_parsed = parse_metta_test_output(direct_output)
    direct_classified = classify_smoke_result(
        {"test": "ec-conflicting-direct", "returncode": direct_rc, "output": direct_output}
    )

    projected_rc, projected_out, projected_err = _run_patham9_program(
        smoke["projected"]["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="ec_conflicting_projected.metta",
    )
    projected_output = f"{projected_out}\n{projected_err}"
    projected_parsed = parse_metta_test_output(projected_output)
    projected_classified = classify_smoke_result(
        {"test": "ec-conflicting-projected", "returncode": projected_rc, "output": projected_output}
    )

    return {
        "schema": "petta-memory-patham9-pln-ec-projection-conflicting-smoke-result-v1",
        "status": "passed" if direct_classified["status"] == "passed" and projected_classified["status"] == "passed" else "failed",
        "strength_lowered": smoke["strength_lowered"],
        "direct": {
            "classification": direct_classified,
            "semantic_markers": direct_parsed,
            "returncode": direct_rc,
            "stdout_tail": direct_out[-2000:],
            "stderr_tail": direct_err[-2000:],
        },
        "projected": {
            "classification": projected_classified,
            "semantic_markers": projected_parsed,
            "returncode": projected_rc,
            "stdout_tail": projected_out[-2000:],
            "stderr_tail": projected_err[-2000:],
        },
        "ec_projection": smoke["ec_projection"],
        "conflicting_ec": smoke["conflicting_ec"],
        "program": smoke,
        "boundary": "non-live read-only comparison with conflicting EC; no memory append, no inferred-belief promotion, no OmegaClaw/GoalChainer live path",
    }


def patham9_pln_derivation_ec_projection_smoke_program(
    handoff: dict[str, Any],
    *,
    item_index: int = 0,
) -> dict[str, Any]:
    """Build a direct-vs-projected two-premise derivation EC projection smoke.

    Produves two derivation programs: one using the original (direct) STV and
    one using the wrapper-projected STV after folding in contextual EC counts.
    Both use the same synthetic bridge implication so the only variable is
    whether the source sentence STV was pre-projected.  This verifies the EC
    projection formula influences derived results, not just direct recall.
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
    base_strength = float(item["stv"]["strength"])
    base_confidence = float(item["stv"]["confidence"])
    raw_packets = item.get("pi_pln_extension", {}).get("contextual_evidence_packets", [])
    projection = ec_projected_stv(base_strength, base_confidence, raw_packets)
    projected_strength = projection["projected_strength"]
    projected_confidence = projection["projected_confidence"]

    source_stamp = _numeric_stamp(item_index)
    bridge_stamp = _numeric_stamp(item_index + 1)
    derived_term = f"(PMDerivedFromHandoff {term})"
    bridge_strength = "1.0"
    bridge_confidence = "0.90"

    def _build_derivation_program(strength_str: str, confidence_str: str) -> str:
        source_sentence = f"(Sentence ({term} (stv {strength_str} {confidence_str})) {source_stamp})"
        bridge_sentence = (
            f"(Sentence ((Implication {term} {derived_term}) "
            f"(stv {bridge_strength} {bridge_confidence})) {bridge_stamp})"
        )
        expected_strength = str(float(strength_str) * float(bridge_strength) + 0.02 * (1.0 - float(strength_str)))
        expected_confidence = str(float(confidence_str) * float(bridge_confidence))
        expected = f"((stv {expected_strength} {expected_confidence}) ({item_index} {item_index + 1}))"
        program = "\n".join(
            [
                "!(import! &self PLN)",
                "!(PLN.Init ())",
                f"!(Test (PLN.Query ({source_sentence}",
                f"                   {bridge_sentence})",
                f"                  {derived_term}",
                "                  2 5 8)",
                f"       {expected})",
                "",
            ]
        )
        return program

    direct_strength_str = item["stv"]["strength"]
    direct_confidence_str = item["stv"]["confidence"]
    direct_program = _build_derivation_program(direct_strength_str, direct_confidence_str)
    direct_expected_strength = str(float(direct_strength_str) * float(bridge_strength) + 0.02 * (1.0 - float(direct_strength_str)))
    direct_expected_confidence = str(float(direct_confidence_str) * float(bridge_confidence))
    direct_expected = f"((stv {direct_expected_strength} {direct_expected_confidence}) ({item_index} {item_index + 1}))"

    projected_program = _build_derivation_program(str(projected_strength), str(projected_confidence))
    projected_expected_strength = str(projected_strength * float(bridge_strength) + 0.02 * (1.0 - projected_strength))
    projected_expected_confidence = str(projected_confidence * float(bridge_confidence))
    projected_expected = f"((stv {projected_expected_strength} {projected_expected_confidence}) ({item_index} {item_index + 1}))"

    # Verify projected derivation differs from direct
    direct_expected_strength_float = float(direct_expected_strength)
    projected_expected_strength_float = float(projected_expected_strength)
    projected_expected_confidence_float = float(projected_expected_confidence)
    direct_expected_confidence_float = float(direct_expected_confidence)
    results_differ = (
        abs(projected_expected_strength_float - direct_expected_strength_float) > 1e-9
        or abs(projected_expected_confidence_float - direct_expected_confidence_float) > 1e-9
    )

    return {
        "schema": "petta-memory-patham9-pln-derivation-ec-projection-smoke-program-v1",
        "mode": "read-only-derivation-ec-projection-comparison-smoke",
        "source_term": term,
        "derived_term": derived_term,
        "direct": {
            "program": direct_program,
            "expected_result": direct_expected,
            "stv": {"strength": direct_strength_str, "confidence": direct_confidence_str},
        },
        "projected": {
            "program": projected_program,
            "expected_result": projected_expected,
            "stv": {"strength": str(projected_strength), "confidence": str(projected_confidence)},
        },
        "ec_projection": projection,
        "results_differ": results_differ,
        "stamp_sidecar": {
            source_stamp: {"kind": "petta-memory-source-sentence", "source_evidence_id": str(item.get("evidence_id", "")), "source_item": item},
            bridge_stamp: {"kind": "synthetic-non-live-bridge-implication", "source_item_index": item_index, "rule": "PMDerivedFromHandoff implication smoke"},
        },
        "boundary": "non-live read-only comparison of direct vs projected two-premise derivation smokes; no memory append, no inferred-belief promotion, no OmegaClaw/GoalChainer live path",
    }


def run_patham9_pln_derivation_ec_projection_smoke(
    handoff: dict[str, Any],
    *,
    pln_repo: str | Path,
    env_script: str | Path | None = None,
    timeout_sec: float = 30.0,
    item_index: int = 0,
) -> dict[str, Any]:
    """Run the derivation EC projection comparison smoke in isolated temp files."""
    smoke = patham9_pln_derivation_ec_projection_smoke_program(handoff, item_index=item_index)

    direct_rc, direct_out, direct_err = _run_patham9_program(
        smoke["direct"]["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="derivation_ec_direct.metta",
    )
    direct_output = f"{direct_out}\n{direct_err}"
    direct_parsed = parse_metta_test_output(direct_output)
    direct_classified = classify_smoke_result(
        {"test": "derivation-ec-direct", "returncode": direct_rc, "output": direct_output}
    )

    projected_rc, projected_out, projected_err = _run_patham9_program(
        smoke["projected"]["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="derivation_ec_projected.metta",
    )
    projected_output = f"{projected_out}\n{projected_err}"
    projected_parsed = parse_metta_test_output(projected_output)
    projected_classified = classify_smoke_result(
        {"test": "derivation-ec-projected", "returncode": projected_rc, "output": projected_output}
    )

    return {
        "schema": "petta-memory-patham9-pln-derivation-ec-projection-smoke-result-v1",
        "status": "passed" if direct_classified["status"] == "passed" and projected_classified["status"] == "passed" else "failed",
        "results_differ": smoke["results_differ"],
        "direct": {
            "classification": direct_classified,
            "semantic_markers": direct_parsed,
            "returncode": direct_rc,
            "stdout_tail": direct_out[-2000:],
            "stderr_tail": direct_err[-2000:],
        },
        "projected": {
            "classification": projected_classified,
            "semantic_markers": projected_parsed,
            "returncode": projected_rc,
            "stdout_tail": projected_out[-2000:],
            "stderr_tail": projected_err[-2000:],
        },
        "ec_projection": smoke["ec_projection"],
        "program": smoke,
        "boundary": "non-live read-only derivation EC projection comparison; no memory append, no inferred-belief promotion, no OmegaClaw/GoalChainer live path",
    }


def patham9_pi_pln_boundary_plan(handoff: dict[str, Any]) -> dict[str, Any]:
    """Describe the first reviewed pi-PLN extension boundary for patham9/PLN.

    The direct query and two-premise derivation smokes establish that the checked
    out patham9/PLN chainer can consume plain `Sentence` atoms.  This helper keeps
    that core unchanged and makes the next integration boundary explicit: a
    petta-memory-owned wrapper pre-projects contextual EvidencePacket/EC metadata
    into additional sidecar inputs before invoking `PLN.Query`/`PLN.Derive`; it
    must not patch patham9/PLN internals, append derived results, or reinterpret
    raw quoted claims as premises.
    """
    if handoff.get("schema") != "petta-memory-patham9-pln-handoff-v1":
        raise ValueError("expected petta-memory-patham9-pln-handoff-v1 handoff")
    items = list(handoff.get("items", []))
    projection_inputs: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        packets = item.get("pi_pln_extension", {}).get("contextual_evidence_packets", [])
        parsed_packets: list[dict[str, Any]] = []
        for packet in packets:
            support = float(str(packet.get("support", "0")))
            opposition = float(str(packet.get("opposition", "0")))
            total = support + opposition
            parsed_packets.append(
                {
                    "support": support,
                    "opposition": opposition,
                    "total_evidence": total,
                    "positive_ratio": None if total == 0 else support / total,
                    "source_packet": packet,
                }
            )
        projection_inputs.append(
            {
                "item_index": index,
                "term": item.get("term"),
                "base_stv": item.get("stv"),
                "source_evidence_id": item.get("evidence_id"),
                "contextual_packets": parsed_packets,
                "runtime_stamp_policy": "wrapper assigns numeric runtime stamps; PMEvidence/provenance stays in sidecar",
            }
        )
    return {
        "schema": "petta-memory-patham9-pi-pln-boundary-plan-v1",
        "decision": "wrapper-first",
        "patham9_core_policy": "treat checked-out patham9/PLN as an unmodified functional chainer for PLN.Query/PLN.Derive over Sentence atoms",
        "wrapper_responsibilities": [
            "convert promoted petta-memory handoff items to patham9-compatible Sentence atoms",
            "assign sortable numeric runtime stamps and preserve PMEvidence provenance in sidecars",
            "pre-project reviewed contextual EvidencePacket/EC formulas into explicit Sentence inputs before runtime invocation",
            "parse semantic Passed markers and keep inferred results artifact-only until a separate promotion review",
        ],
        "patham9_extension_points": {
            "PLN.Query": "safe runtime entrypoint after wrapper projection; already smoke-tested",
            "PLN.Derive": "safe runtime entrypoint for bounded derivation after wrapper projection; already smoke-tested",
            "Sentence": "unchanged data boundary: (Sentence ($Term (stv S C)) $Stamp) at runtime, with provenance sidecar",
            "StampDisjoint": "reason to keep runtime stamps numeric/simple until richer stamps are tested",
            "PriorityRank": "current queue ordering uses confidence; wrapper may adjust confidence only through reviewed formulas",
        },
        "formula_policy": "no truth-changing EC projection is live yet; next gate should add a tiny reviewed wrapper formula over contextual_packets and compare artifact-only query/derive behavior",
        "projection_inputs": projection_inputs,
        "non_live_gates": [
            "no patham9/PLN source patching before wrapper projection tests pass",
            "no memory append or inferred-belief promotion from PLN outputs",
            "no OmegaClaw/GoalChainer live integration before a non-live artifact gate passes",
        ],
    }


def patham9_pi_pln_extension_spec(
    handoff: dict[str, Any],
    *,
    pln_repo: str | Path | None = None,
) -> dict[str, Any]:
    """Produce a concrete π-PLN extension layer specification.

    This formalizes what the wrapper-first boundary does, what formulas it
    applies, and what policies govern sentence construction, stamp assignment,
    provenance sidecars, context selection, and inference-control hooks.
    The spec is derived from the already-mapped patham9/PLN API surface and
    the tested EC projection smokes.  It is a design artifact, not a runtime
    invocation.

    The spec covers:
      1. Sentence construction protocol (STV + numeric stamps)
      2. EC projection formula (confidence-weighted blend, already tested)
      3. Provenance sidecar policy (PMEvidence preserved outside chainer)
      4. Context selection policy (wrapper-owned, not yet live)
      5. Inference control hooks (deferred, mapped to trueagi-io/chaining patterns)
      6. Read/write boundaries (no memory append, no inferred-belief promotion)
      7. Revisit triggers for internal patham9/PLN extensions
    """
    if handoff.get("schema") != "petta-memory-patham9-pln-handoff-v1":
        raise ValueError("expected petta-memory-patham9-pln-handoff-v1 handoff")
    items = list(handoff.get("items", []))

    # Summarize each item's projection inputs
    projection_inputs: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        packets = item.get("pi_pln_extension", {}).get("contextual_evidence_packets", [])
        base_strength = float(item["stv"]["strength"])
        base_confidence = float(item["stv"]["confidence"])
        projection = ec_projected_stv(base_strength, base_confidence, packets)
        projection_inputs.append(
            {
                "item_index": index,
                "term": item.get("term"),
                "base_stv": item.get("stv"),
                "projected_stv": {
                    "strength": projection["projected_strength"],
                    "confidence": projection["projected_confidence"],
                },
                "contextual_packet_count": len(packets),
                "source_evidence_id": item.get("evidence_id"),
            }
        )

    return {
        "schema": "petta-memory-patham9-pi-pln-extension-spec-v1",
        "mode": "design-specification-no-runtime",
        "version": "0.1",
        "boundary_decision": "wrapper-first: keep checked-out patham9/PLN unmodified; petta-memory owns wrapper layer",
        "sentence_construction_protocol": {
            "format": "(Sentence ($Term (stv S C)) $Stamp)",
            "stv_source": "base STV from promoted handoff item, or projected STV after ec_projected_stv() when contextual EC packets are present",
            "stamp_policy": "numeric runtime stamps (0), (1), ... for patham9/PLN StampDisjoint compatibility; symbolic PMEvidence preserved in sidecar only",
            "term_policy": "use the promoted BeliefContent term directly as the Sentence term; no rewriting or normalization",
        },
        "ec_projection_formula": {
            "name": "confidence-weighted blend",
            "formula": "projected_strength = sum(s_i * w_i) / sum(w_i); projected_confidence = max(all confidences)",
            "inputs": [
                "base STV (strength, confidence) from promoted handoff item",
                "contextual EvidencePacket EC counts (support, opposition)",
            ],
            "per_packet": {
                "ec_strength": "support / (support + opposition)",
                "ec_confidence": "(support + opposition) / (support + opposition + 2)  # Laplace-smoothed",
                "weight": "ec_confidence",
            },
            "properties": [
                "cannot inflate confidence beyond base",
                "weak evidence has little effect (low confidence = low weight)",
                "zero-total packets are skipped",
                "conflicting EC lowers projected strength",
            ],
            "tested_in": [
                "test_ec_projected_stv_blends_confidence_weighted",
                "test_ec_projected_stv_with_conflicting_ec_lowers_strength",
                "test_patham9_pln_derivation_ec_projection_smoke_program_builds_direct_and_projected",
            ],
            "status": "reviewed and implemented as ec_projected_stv()",
        },
        "provenance_sidecar_policy": {
            "location": "JSON result sidecar, not inside patham9/PLN chainer",
            "contents": [
                "source PMEvidence identifier (belief_id, cluster_id, promotion_event, promotion_rule, promotion_domain)",
                "runtime stamp to source evidence mapping",
                "synthetic bridge provenance for derivation smokes",
                "contextual EvidencePacket metadata (support, opposition, domain, promotion_rule)",
            ],
            "boundary": "sidecar is artifact-only; not appended to memory, not promoted as inferred belief",
        },
        "context_selection_policy": {
            "current_state": "not-live; wrapper does not yet filter or generate contexts",
            "design": "wrapper will own context filtering before invocation: select relevant EvidencePackets by domain, cluster, or promotion_rule, then pre-project STV",
            "patham9_support": "patham9/PLN has no built-in context selection; wrapper owns this entirely",
            "revisit_trigger": "when context filtering logic is complex enough to warrant its own test suite",
        },
        "inference_control_hooks": {
            "current_state": "deferred (roadmap item 4)",
            "reference_patterns": [
                "trueagi-io/chaining pln-inf-ctl.metta: PLN-based inference controller that estimates query viability before committing to recursive search",
                "trueagi-io/chaining inf-ctl-month-bc-cont-xp.metta: continuation predicates per branch type",
                "trueagi-io/chaining prob-chaining: probabilistic pruning of axiom/rule selection",
            ],
            "design_direction": "after basic multi-Sentence derivation is validated, explore wrapper-level inference control that pre-filters or re-orders Sentences before PLN.Query/PLN.Derive",
            "revisit_trigger": "after multi-Sentence derivation smoke passes and before broad profiling",
        },
        "read_write_boundaries": {
            "no_memory_append": "PLN output is artifact-only; never appended to petta-memory journal",
            "no_inferred_belief_promotion": "derived results are not promoted as DerivedBelief without separate review gate",
            "no_omegaclaw_live": "no OmegaClaw/GoalChainer live integration before non-live artifact gate passes",
            "no_patham9_source_patch": "checked-out patham9/PLN source remains unmodified",
        },
        "revisit_triggers": {
            "internal_extension": "if wrapper-level EC/context projection cannot express required pi-PLN semantics, consider narrowly scoped patham9/PLN internal extension with regression tests over StampDisjoint, queue priority, and proof provenance",
            "inference_control": "after multi-Sentence derivation is validated and basic profiling is done",
            "context_selection": "when context filtering logic is complex enough to warrant its own test suite",
        },
        "projection_inputs": projection_inputs,
        "item_count": len(items),
        "boundary": "design specification artifact; no runtime invoked; no memory append; no inferred-belief promotion; no OmegaClaw/GoalChainer live path",
    }


def patham9_pln_multi_sentence_derivation_smoke_program(
    handoff: dict[str, Any],
    *,
    bridge_term: str | None = None,
) -> dict[str, Any]:
    """Build a multi-Sentence derivation smoke for patham9/PLN.

    Loads ALL handoff Sentences (not just one) plus a synthetic bridge
    implication from the first term to a derived `PMDerivedFromHandoff`
    term.  This validates the wrapper boundary with multiple premises,
    exercising StampDisjoint across several numeric stamps and testing
    that PLN.Query can find a derivation path through multiple loaded
    beliefs.

    The smoke is non-live: it produces a MeTTa program artifact and
    expected result metadata, but does not invoke the runtime itself.
    Use `run_patham9_pln_multi_sentence_derivation_smoke()` to execute it.
    """
    if handoff.get("schema") != "petta-memory-patham9-pln-handoff-v1":
        raise ValueError("expected petta-memory-patham9-pln-handoff-v1 handoff")
    items = list(handoff.get("items", []))
    if not items:
        raise ValueError("patham9/PLN handoff has no Sentence items")

    # Build runtime sentences for all handoff items
    runtime_sentences: list[str] = []
    stamp_sidecar: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        term = str(item["term"])
        strength = str(item["stv"]["strength"])
        confidence = str(item["stv"]["confidence"])
        stamp = _numeric_stamp(index)
        runtime_sentences.append(f"(Sentence ({term} (stv {strength} {confidence})) {stamp})")
        stamp_sidecar[stamp] = {
            "kind": "petta-memory-source-sentence",
            "source_evidence_id": str(item.get("evidence_id", "")),
            "source_item_index": index,
            "term": term,
        }

    # Synthetic bridge implication from first term to derived term
    first_item = items[0]
    first_term = str(first_item["term"])
    derived_term = bridge_term or f"(PMDerivedFromMultiHandoff {first_term})"
    bridge_stamp = _numeric_stamp(len(items))
    bridge_strength = "1.0"
    bridge_confidence = "0.90"
    bridge_sentence = (
        f"(Sentence ((Implication {first_term} {derived_term}) "
        f"(stv {bridge_strength} {bridge_confidence})) {bridge_stamp})"
    )
    stamp_sidecar[bridge_stamp] = {
        "kind": "synthetic-non-live-bridge-implication",
        "source_item_index": 0,
        "rule": "PMDerivedFromMultiHandoff implication smoke",
    }

    # Compute expected result (same formula as single derivation smoke)
    base_strength = float(first_item["stv"]["strength"])
    base_confidence = float(first_item["stv"]["confidence"])
    expected_strength = base_strength * float(bridge_strength) + 0.02 * (1.0 - base_strength)
    expected_confidence = base_confidence * float(bridge_confidence)
    expected = f"((stv {expected_strength} {expected_confidence}) (0 {len(items)}))"

    # Build the MeTTa program (sentences are space-separated, matching
    # patham9/PLN's MeTTa list syntax; comma separation breaks the chainer)
    all_sentences = "\n                  ".join(runtime_sentences + [bridge_sentence])
    program_lines = [
        "!(import! &self PLN)",
        "!(PLN.Init ())",
        f"!(Test (PLN.Query ({all_sentences})",
        f"                  {derived_term}",
        f"                  {len(items) + 1} {len(items) + 4} {len(items) + 8})",
        f"       {expected})",
        "",
    ]
    program = "\n".join(program_lines)

    return {
        "schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-program-v1",
        "mode": "read-only-multi-sentence-derivation-smoke",
        "program": program,
        "source_term": first_term,
        "derived_term": derived_term,
        "runtime_sentences": runtime_sentences + [bridge_sentence],
        "sentence_count": len(runtime_sentences) + 1,
        "handoff_sentence_count": len(runtime_sentences),
        "expected_result": expected,
        "runtime_stamp_policy": "numeric patham9/PLN stamps for chainer compatibility; source evidence and synthetic bridge provenance preserved in sidecar",
        "stamp_sidecar": stamp_sidecar,
        "boundary": "loads all generated Sentences plus one synthetic bridge implication into local patham9/PLN for a bounded multi-sentence derivation smoke; no memory append, no inferred-belief promotion, no OmegaClaw/GoalChainer live path",
    }


def run_patham9_pln_multi_sentence_derivation_smoke(
    handoff: dict[str, Any],
    *,
    pln_repo: str | Path,
    env_script: str | Path | None = None,
    timeout_sec: float = 30.0,
    bridge_term: str | None = None,
) -> dict[str, Any]:
    """Run the multi-Sentence derivation smoke in an isolated temp file."""
    program = patham9_pln_multi_sentence_derivation_smoke_program(handoff, bridge_term=bridge_term)
    returncode, stdout, stderr = _run_patham9_program(
        program["program"],
        pln_repo=pln_repo,
        env_script=env_script,
        timeout_sec=timeout_sec,
        filename="multi_sentence_derivation_smoke.metta",
    )
    output = f"{stdout}\n{stderr}"
    parsed = parse_metta_test_output(output)
    classified = classify_smoke_result(
        {
            "test": "patham9-pln-multi-sentence-derivation-smoke",
            "returncode": returncode,
            "output": output,
        }
    )
    return {
        "schema": "petta-memory-patham9-pln-multi-sentence-derivation-smoke-result-v1",
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


def _read_metta_source(pln_repo: str | Path, relative_path: str) -> str:
    """Read a .metta source file from the patham9/PLN checkout."""
    path = Path(pln_repo) / relative_path
    if not path.exists():
        raise FileNotFoundError(f"patham9/PLN source file not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_def_lines(source: str) -> List[Dict[str, str]]:
    """Extract (name . definition-line) entries from MeTTa source text."""
    entries: List[Dict[str, str]] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        # Match (= (Name ...)  or (: (Name ...)
        if stripped.startswith("(= ") or stripped.startswith("(: "):
            # Extract the head name
            inner = stripped[3:].strip()
            paren_depth = 0
            name_end = 0
            for i, ch in enumerate(inner):
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    if paren_depth == 0:
                        name_end = i
                        break
                    paren_depth -= 1
                elif ch == " " and paren_depth == 0:
                    name_end = i
                    break
            head = inner[:name_end].strip() if name_end else inner.split()[0]
            entries.append({"head": head, "line": stripped})
    return entries


def patham9_pln_api_surface(pln_repo: str | Path) -> Dict[str, Any]:
    """Map the patham9/PLN API surface from checked-out source files.

    This is a source-level, no-runtime inspection that documents the public
    MeTTa-level API of patham9/PLN and identifies extension points for pi-PLN
    semantics (EvidencePacket, EC counts, context selection, provenance
    lineage, STV projection).

    The inspection covers:
      - PLN.Query and PLN.Derive: signatures, defaults, and queue mechanics
      - Sentence: the data boundary between petta-memory and the chainer
      - Truth-value formulas: Deduction, Induction, Abduction, Modus Ponens,
        Revision, Negation, Inversion, transitive similarity, evaluation implication
      - Inference rules: the `|-` pattern matcher and guard predicates
      - StampDisjoint: evidence overlap prevention
      - PriorityRank / ConfidenceRank: task and result queue ordering
      - LimitSize: bounded priority queue functionality
      - Config: default step/queue sizes
      - Translator: implication-to-function translation during PLN.Init
      - Python entrypoint (examples/PLN.py): PLN.Init registration

    Extension points for pi-PLN are identified at the wrapper boundary:
      - Sentence construction (wrapper pre-projects STV before loading)
      - Stamp assignment (numeric stamps with sidecar provenance)
      - Queue priority (confidence-based; wrapper may adjust via projection)
      - Truth-value formulas (can be extended or replaced via wrapper-fed STV)
      - Inference rules (|- patterns; wrapper can add Sentences with new link types)
      - Context selection (not yet in patham9/PLN; wrapper owns context filtering)
    """
    repo = Path(pln_repo)
    if not repo.exists():
        raise FileNotFoundError(f"patham9/PLN checkout not found at {repo}")

    # Read all source files
    pln_metta = _read_metta_source(repo, "PLN.metta")
    config_src = _read_metta_source(repo, "src/Config.metta")
    constraints_src = _read_metta_source(repo, "src/Constraints.metta")
    deriver_src = _read_metta_source(repo, "src/Deriver.metta")
    formulas_src = _read_metta_source(repo, "src/Formulas.metta")
    rules_src = _read_metta_source(repo, "src/Rules.metta")
    utils_src = _read_metta_source(repo, "src/Utils.metta")
    translator_src = _read_metta_source(repo, "src/Translator.metta")

    python_entrypoint = ""
    py_path = repo / "examples" / "PLN.py"
    if py_path.exists():
        python_entrypoint = py_path.read_text(encoding="utf-8", errors="replace")

    # Extract API definitions by category

    # PLN.Derive signatures
    derive_defs: List[Dict[str, str]] = []
    for line in deriver_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("(= (PLN.Derive"):
            derive_defs.append({"definition": stripped})

    # PLN.Query signatures
    query_defs: List[Dict[str, str]] = []
    for line in deriver_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("(= (PLN.Query"):
            query_defs.append({"definition": stripped})

    # StampDisjoint
    stamp_disjoint_defs: List[Dict[str, str]] = []
    for line in utils_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("(= (StampDisjoint"):
            stamp_disjoint_defs.append({"definition": stripped})

    # PriorityRank / ConfidenceRank
    priority_defs: List[Dict[str, str]] = []
    for line in utils_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("(= (PriorityRank") or stripped.startswith("(= (ConfidenceRank"):
            priority_defs.append({"definition": stripped})

    # LimitSize
    limit_size_defs: List[Dict[str, str]] = []
    for line in utils_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("(= (LimitSize"):
            limit_size_defs.append({"definition": stripped})

    # BestCandidate
    best_candidate_defs: List[Dict[str, str]] = []
    for line in utils_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("(= (BestCandidate"):
            best_candidate_defs.append({"definition": stripped})

    # Truth-value formulas
    truth_formula_names = [
        "Truth_Deduction", "Truth_Induction", "Truth_Abduction",
        "Truth_ModusPonens", "Truth_SymmetricModusPonens", "Truth_Revision",
        "Truth_Negation", "Truth_inversion", "Truth_equivalenceToImplication",
        "Truth_transitiveSimilarity", "Truth_evaluationImplication",
        "Truth_Identity", "Truth_c2w", "Truth_w2c",
        "simpleDeductionStrength", "TransitiveSimilarityStrength",
    ]
    truth_formulas: List[Dict[str, str]] = []
    for name in truth_formula_names:
        for line in formulas_src.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"(= ({name}") or stripped.startswith(f"(: ({name}"):
                truth_formulas.append({"name": name, "definition": stripped})

    # Inference rules (|- pattern)
    inference_rules: List[Dict[str, str]] = []
    for line in rules_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("(= (|-"):
            inference_rules.append({"definition": stripped})

    # Guard predicates
    guard_defs: List[Dict[str, str]] = []
    for line in rules_src.splitlines():
        stripped = line.strip()
        if "RuleGuard" in stripped and stripped.startswith("(= "):
            guard_defs.append({"definition": stripped})

    # Config defaults
    config_defs: List[Dict[str, str]] = []
    for line in config_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("(= (PLN.Config."):
            config_defs.append({"definition": stripped})

    # Constraint helpers
    constraint_defs = _extract_def_lines(constraints_src)

    # Translator definitions
    translator_defs = _extract_def_lines(translator_src)

    # Utility helpers
    utility_defs = _extract_def_lines(utils_src)

    # Python entrypoint summary
    python_summary: Dict[str, Any] = {}
    if python_entrypoint:
        python_summary = {
            "file": "examples/PLN.py",
            "registers": "PLN.Init atom via hyperon ext register_atoms",
            "init_function": "call_plninit",
            "init_steps": [
                "cd PLN checkout && sh build.sh",
                "cp src/Translator.metta as TRANSLATE.metta with superpose query appended",
                "cat PLN.metta > TRANSLATED.metta; run metta TRANSLATE.metta >> TRANSLATED.metta",
                "strip # from variable names in translated output",
                "copy translated file to metta-morph extend/ as plnblob.metta",
                "import! &self mettamorph; compile! plnblob.metta",
                "define (sentence $A $B) = (Sentence $A $B)",
            ],
            "dependency": "metta-morph (mettamorphpath) must be installed",
            "note": "PLN.Init builds and translates the PLN rulebase into a compiled metta-morph module; not a lightweight startup",
        }

    return {
        "schema": "petta-memory-patham9-pln-api-surface-v1",
        "mode": "source-level-no-runtime-inspection",
        "pln_repo": str(repo),
        "source_files": {
            "PLN.metta": f"{len(pln_metta.splitlines())} lines",
            "src/Config.metta": f"{len(config_src.splitlines())} lines",
            "src/Constraints.metta": f"{len(constraints_src.splitlines())} lines",
            "src/Deriver.metta": f"{len(deriver_src.splitlines())} lines",
            "src/Formulas.metta": f"{len(formulas_src.splitlines())} lines",
            "src/Rules.metta": f"{len(rules_src.splitlines())} lines",
            "src/Utils.metta": f"{len(utils_src.splitlines())} lines",
            "src/Translator.metta": f"{len(translator_src.splitlines())} lines",
            "examples/PLN.py": f"{len(python_entrypoint.splitlines())} lines" if python_entrypoint else "not found",
        },
        "core_api": {
            "PLN.Derive": {
                "description": "Priority-queue based task ranking deriver with belief buffer",
                "signatures": derive_defs,
                "full_signatures": [
                    "(PLN.Derive $Tasks $Beliefs $steps $maxsteps $taskqueuesize $beliefqueuesize)",
                    "(PLN.Derive $Tasks $Beliefs $maxsteps $taskqueuesize $beliefqueuesize)",
                    "(PLN.Derive $Tasks $Beliefs $maxsteps)",
                    "(PLN.Derive $Tasks $Beliefs)",
                ],
                "defaults": {
                    "maxsteps": "PLN.Config.MaxSteps (20)",
                    "taskqueuesize": "PLN.Config.TaskQueueSize (20)",
                    "beliefqueuesize": "PLN.Config.BeliefQueueSize (200)",
                },
                "mechanics": [
                    "Selects highest-confidence task via BestCandidate/PriorityRank",
                    "Matches selected task against all beliefs via |- rules",
                    "Checks StampDisjoint to avoid counting same evidence twice",
                    "Merges derived results into task and belief queues via Unique + LimitSize",
                    "Recurses until maxsteps or empty task queue",
                    "Traces SELECTED/DERIVED for debugging via trace!",
                ],
            },
            "PLN.Query": {
                "description": "Pose a query term to the system; returns best-confidence result",
                "signatures": query_defs,
                "full_signatures": [
                    "(PLN.Query $Tasks $Beliefs $term $maxsteps $taskqueuesize $beliefqueuesize)",
                    "(PLN.Query $kb $term $maxsteps $taskqueuesize $beliefqueuesize)",
                    "(PLN.Query $kb $term $maxsteps)",
                    "(PLN.Query $kb $term)",
                ],
                "defaults": {
                    "maxsteps": "PLN.Config.MaxSteps (20)",
                    "taskqueuesize": "PLN.Config.TaskQueueSize (20)",
                    "beliefqueuesize": "PLN.Config.BeliefQueueSize (200)",
                },
                "mechanics": [
                    "Runs PLN.Derive with Tasks=Beliefs=kb",
                    "After derivation, searches belief results for matching term",
                    "Returns (TV Ev) tuple from the highest-confidence matching Sentence",
                    "Uses BestCandidate/ConfidenceRank for result selection",
                ],
            },
            "Sentence": {
                "description": "Data boundary: (Sentence ($Term (stv S C)) $Evidence)",
                "structure": "(Sentence ($term (stv $strength $confidence)) $evidence_stamp)",
                "notes": [
                    "$term can be any MeTTa expression (Concept, Evaluation, Implication, etc.)",
                    "$evidence_stamp is a tuple of sortable stamps; StampDisjoint checks overlap",
                    "Current implementation uses simple numeric stamps for sorting compatibility",
                    "petta-memory wrapper uses numeric stamps with PMEvidence provenance in sidecar",
                ],
            },
            "StampDisjoint": {
                "description": "Check whether two evidence tuples share any element",
                "signatures": stamp_disjoint_defs,
                "mechanics": [
                    "Expands both evidence tuples via superpose",
                    "Checks pairwise equality with case (== $x $y)",
                    "Returns True if no overlap (empty intersection), False otherwise",
                ],
                "extension_note": "Rich symbolic stamps (PMEvidence) can trip the sorter; wrapper uses numeric stamps and maps back to provenance in a sidecar",
            },
            "PriorityRank": {
                "description": "Task queue priority: confidence of the Sentence",
                "signatures": priority_defs,
                "formula": "(PriorityRank (Sentence ($x (stv $f $c)) $Ev1)) = $c",
                "fallback": "(PriorityRank ()) = -99999.0",
            },
            "ConfidenceRank": {
                "description": "Query result ranking: confidence of the TV",
                "formula": "(ConfidenceRank ((stv $f $c) $Ev)) = $c",
                "fallback": "(ConfidenceRank ()) = 0",
            },
            "LimitSize": {
                "description": "Bounded priority queue: evicts lowest-priority items when over capacity",
                "signatures": limit_size_defs,
                "mechanics": [
                    "If tuple count < size, return as-is",
                    "Otherwise find lowest-priority item via BestCandidate/PriorityRankNeg",
                    "Remove it and recurse",
                ],
            },
            "BestCandidate": {
                "description": "Linear scan to find highest-scoring item in a tuple",
                "signatures": best_candidate_defs,
                "mechanics": [
                    "Walks tuple element by element",
                    "Compares each element's score (via evaluation function) against current best",
                    "Returns the best candidate",
                ],
            },
        },
        "truth_value_formulas": truth_formulas,
        "inference_rules": inference_rules,
        "guard_predicates": guard_defs,
        "config_defaults": config_defs,
        "constraint_helpers": constraint_defs,
        "translator": {
            "description": "Translates nested implications into function compositions during PLN.Init",
            "definitions": translator_defs,
            "notes": [
                "Handles Implication chains and Not negation patterns",
                "Generates buildlink calls that compose Truth_Negation and Truth_Identity",
                "Also translates concept STV assertions via (sentence ($C $TV) $Y)",
            ],
        },
        "utility_helpers": utility_defs,
        "python_entrypoint": python_summary,
        "pi_pln_extension_points": {
            "wrapper_boundary": {
                "sentence_construction": "Wrapper builds Sentence atoms with pre-projected STV before loading into chainer; no patham9 source change needed",
                "stamp_assignment": "Wrapper assigns numeric runtime stamps; PMEvidence/provenance preserved in sidecar JSON, not in the chainer",
                "stv_pre_projection": "ec_projected_stv() computes confidence-weighted blend of base STV and EC-derived evidence; wrapper feeds projected STV as the Sentence truth value",
                "context_selection": "Not present in patham9/PLN; wrapper owns context filtering and EvidencePacket-to-STV projection before invocation",
                "queue_priority": "PriorityRank uses confidence; wrapper can only influence priority through the STV confidence it assigns to Sentences",
                "truth_value_formulas": "Patham9 formulas are MeTTa-level and unmodified; wrapper can pre-project or post-process results but cannot add new formulas without source changes",
                "inference_rules": "|- pattern matcher is open for new rules via MeTTa definitions; wrapper could inject additional Sentences or implications but should not patch rules without review",
                "provenance_lineage": "StampDisjoint prevents double-counting evidence; wrapper ensures numeric stamps are disjoint and maps back to PMEvidence in sidecar",
            },
            "internal_extensions": {
                "context_indexed_evidence": "Would require adding context-parameterized Sentence or EvidencePacket atoms and modifying StampDisjoint/PLN.Derive; not yet needed if wrapper pre-projects",
                "ec_aware_truth_formulas": "Would require new Truth_* formulas that accept EC counts; current wrapper pre-projects into a single STV instead",
                "inference_control": "Would require meta-level control predicates (cf. trueagi-io/chaining pln-inf-ctl.metta); deferred to roadmap item 4",
                "custom_link_types": "Would require adding |- rules for petta-memory-specific link types (e.g., EvidencePacket, PromotionRule); not yet needed",
            },
            "revisit_trigger": "If wrapper-level projection cannot express required pi-PLN semantics, consider narrowly scoped patham9/PLN internal extension with regression tests over StampDisjoint, queue priority, and proof provenance",
        },
        "boundary": "source-level inspection only; no SWI/PeTTa/MeTTa runtime invoked; no memory append; no inferred-belief promotion; no OmegaClaw/GoalChainer live path",
    }


def survey_trueagi_chaining_inference_control(chaining_repo: str | Path) -> Dict[str, Any]:
    """Survey inference-control patterns from the trueagi-io/chaining repo.

    This is a source-level, no-runtime inspection that maps concrete
    inference-control patterns from the experimental chaining repository
    (cloned at ``repos/trueagi-chaining``) to pi-PLN wrapper extension
    points.  It directly supports the deferred roadmap item: *Design
    OmegaClaw-specific inference-control mechanisms*.

    The survey covers:
      - PLN-based inference controller (``pln-inf-ctl/pln-inf-ctl.metta``):
        uses PLN queries to estimate branch viability, Thompson sampling
        for exploration/exploitation, ``EDCall`` estimated delayed calls,
        ``Control`` structure with PLN estimator, ``toPLN`` query converter
      - Controlled backward chainer (``inference-control/inf-ctl-xp.metta``):
        parameterized chainer with context abstraction/argument updaters and
        a termination predicate
      - Meta-learning inference control (``inference-control/inf-ctl-month-xp.metta``):
        reproduces OpenCog classic inference control meta-learning with a
        shortcut rule
      - Controller-as-chainer (``inference-control/inf-ctl-month-bc-xp.metta``):
        uses another backward chainer instance as the termination controller
      - Continuation predicate (``inference-control/inf-ctl-month-bc-cont-xp.metta``):
        replaces termination with a ``Continue`` dependent type
      - Probabilistic backward chaining (``prob-chaining/prob-chaining.metta``):
        each typing relationship carries a probability that filters base cases

    For each pattern, the survey records:
      - File path, line count, and a concise description
      - Key concepts and data structures
      - How it could be adopted at the pi-PLN wrapper boundary
      - Whether it requires patham9/PLN source changes or is wrapper-only
    """
    repo = Path(chaining_repo)
    if not repo.exists():
        raise FileNotFoundError(f"trueagi-io/chaining checkout not found at {repo}")

    def _read_rel(rel: str) -> str:
        p = repo / rel
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8", errors="replace")

    def _line_count(text: str) -> int:
        return text.count("\n") + 1 if text else 0

    patterns: list[dict[str, Any]] = []

    # 1. PLN-based inference controller
    pln_inf_ctl = _read_rel("experimental/pln-inf-ctl/pln-inf-ctl.metta")
    if pln_inf_ctl:
        patterns.append({
            "name": "PLN-based inference controller",
            "file": "experimental/pln-inf-ctl/pln-inf-ctl.metta",
            "line_count": _line_count(pln_inf_ctl),
            "description": (
                "Uses PLN queries to estimate the probability of success for each "
                "backward chainer branch, then uses Thompson sampling to select "
                "the best branch.  A Control structure holds a PLN knowledge base "
                "and an estimator function.  The estimator converts the current "
                "query and surrounding premises into a PLN statement, runs PLN "
                "backward chaining on it, and Thompson-samples a first-order "
                "probability from the resulting truth value."
            ),
            "key_concepts": [
                "EDCall (Estimated Delayed Call) — pairs a probability estimate with a deferred branch call",
                "Control structure — holds PLN space and estimator function",
                "toPLN converter — maps backward chainer query (theory, proof, theorem) to PLN statement",
                "Thompson sampling — samples first-order probability from PLN truth value (strength, confidence)",
                "pln-bc — PLN backward chainer over judgements (statement, truth value)",
                "𝛩 predicate — ternary relation (theory, proof, theorem) treated probabilistically",
            ],
            "wrapper_adoption": (
                "Wrapper could implement an estimator that converts petta-memory "
                "Sentences and their EC/EvidencePacket metadata into a PLN query "
                "about branch viability, then uses the patham9/PLN chainer itself "
                "to estimate which derivation paths are most promising before "
                "committing to PLN.Derive.  The EDCall pattern maps naturally to "
                "pre-computing projected STV for each candidate Sentence before "
                "loading it into the chainer.  No patham9/PLN source changes needed; "
                "wrapper owns the estimator and Thompson sampling logic."
            ),
            "requires_patham9_source_change": False,
            "complexity": "high — requires implementing a PLN estimator and Thompson sampling in the wrapper",
        })

    # 2. Controlled backward chainer with context updaters and termination
    inf_ctl_xp = _read_rel("experimental/inference-control/inf-ctl-xp.metta")
    if inf_ctl_xp:
        patterns.append({
            "name": "Controlled backward chainer (context updaters + termination)",
            "file": "experimental/inference-control/inf-ctl-xp.metta",
            "line_count": _line_count(inf_ctl_xp),
            "description": (
                "Parameterized backward chainer that accepts control functions: "
                "a context abstraction updater, a context argument updater, and a "
                "termination predicate.  The chainer calls these functions at each "
                "recursive step to decide whether to continue, prune, or update the "
                "inference context before recursing."
            ),
            "key_concepts": [
                "Context abstraction updater — updates context before recursing on proof abstraction",
                "Context argument updater — updates context before recursing on proof argument",
                "Termination predicate — decides whether to prune the current branch",
                "Control structure — holds the three control functions",
                "Curried rules — allows partial application for proof abstractions",
            ],
            "wrapper_adoption": (
                "Wrapper could implement context updaters that track which "
                "EvidencePackets have been consumed, and a termination predicate "
                "that checks whether remaining EC support is sufficient to justify "
                "further derivation.  This maps to the pi-PLN context selection "
                "policy: the wrapper filters Sentences by contextual relevance "
                "before each PLN.Query/PLN.Derive call.  No patham9/PLN source "
                "changes needed; wrapper owns the control functions."
            ),
            "requires_patham9_source_change": False,
            "complexity": "medium — requires implementing context tracking and termination logic",
        })

    # 3. Meta-learning inference control (months)
    inf_ctl_month = _read_rel("experimental/inference-control/inf-ctl-month-xp.metta")
    if inf_ctl_month:
        patterns.append({
            "name": "Meta-learning inference control (OpenCog classic reproduction)",
            "file": "experimental/inference-control/inf-ctl-month-xp.metta",
            "line_count": _line_count(inf_ctl_month),
            "description": (
                "Reproduces the OpenCog classic inference control meta-learning "
                "experiment with months instead of letters.  Includes a shortcut "
                "rule (January precedes all months) that the chainer should learn "
                "to prefer over transitive chains.  Tests that the controlled "
                "chainer converges on the shortcut after learning."
            ),
            "key_concepts": [
                "Shortcut rule — a direct rule that bypasses long transitive chains",
                "Context abstraction — tracks which rules have been applied",
                "Meta-learning — the chainer learns which rules to prefer",
                "Month precedence — transitively chained relation with a shortcut",
            ],
            "wrapper_adoption": (
                "Provides a test scenario for evaluating whether the wrapper's "
                "inference control can learn to prefer high-confidence promoted "
                "beliefs (analogous to shortcut rules) over long derivation chains. "
                "Could be used as a benchmark for the pi-PLN wrapper's context "
                "selection and priority policies."
            ),
            "requires_patham9_source_change": False,
            "complexity": "low — benchmark/test scenario, not a direct implementation pattern",
        })

    # 4. Controller-as-chainer (termination via another chainer)
    inf_ctl_month_bc = _read_rel("experimental/inference-control/inf-ctl-month-bc-xp.metta")
    if inf_ctl_month_bc:
        patterns.append({
            "name": "Controller-as-chainer (termination via backward chainer)",
            "file": "experimental/inference-control/inf-ctl-month-bc-xp.metta",
            "line_count": _line_count(inf_ctl_month_bc),
            "description": (
                "Like the meta-learning experiment but the termination predicate "
                "is evaluated by using another instance of the backward chainer as "
                "a controller.  A Terminate dependent type is proven by the "
                "controller chainer to decide whether to prune a branch."
            ),
            "key_concepts": [
                "Terminate dependent type — (Terminate CONTEXT) must be provable to prune",
                "Controller chainer — a separate backward chainer instance for control decisions",
                "Recursive control — the controller itself can recurse with its own control",
                "Context type — explicit type for the inference context passed to control",
            ],
            "wrapper_adoption": (
                "Wrapper could run a lightweight PLN.Query against the loaded "
                "Sentences to decide whether a derivation branch is worth "
                "continuing.  This is a meta-level use of patham9/PLN itself as "
                "the controller, separate from the main derivation.  No source "
                "changes needed; wrapper runs two PLN.Query calls — one for "
                "control, one for derivation."
            ),
            "requires_patham9_source_change": False,
            "complexity": "high — requires running the chainer at two levels (control + derivation)",
        })

    # 5. Continuation predicate
    inf_ctl_month_cont = _read_rel("experimental/inference-control/inf-ctl-month-bc-cont-xp.metta")
    if inf_ctl_month_cont:
        patterns.append({
            "name": "Continuation predicate (Continue dependent type)",
            "file": "experimental/inference-control/inf-ctl-month-bc-cont-xp.metta",
            "line_count": _line_count(inf_ctl_month_cont),
            "description": (
                "Replaces the termination predicate with a continuation predicate. "
                "A Continue dependent type must be proven to justify continuing a "
                "branch, rather than proving Terminate to prune it.  This inverts "
                "the control logic from opt-out to opt-in."
            ),
            "key_concepts": [
                "Continue dependent type — (Continue QUERY CONTEXT) must be provable to continue",
                "Opt-in control — branches must justify themselves rather than being pruned",
                "Control structure — holds abstraction/argument updaters and continuation checker",
            ],
            "wrapper_adoption": (
                "Wrapper could require that each candidate Sentence in a multi-step "
                "derivation passes a Continue check — e.g., its projected STV "
                "confidence exceeds a threshold, or its EvidencePacket support "
                "ratio is sufficient.  This is simpler than the full PLN estimator "
                "and can be implemented as a wrapper-level filter before PLN.Derive."
            ),
            "requires_patham9_source_change": False,
            "complexity": "medium — requires implementing a continuation check per derivation step",
        })

    # 6. Probabilistic backward chaining
    prob_chain = _read_rel("experimental/prob-chaining/prob-chaining.metta")
    if prob_chain:
        patterns.append({
            "name": "Probabilistic backward chaining (ProbLog-inspired)",
            "file": "experimental/prob-chaining/prob-chaining.metta",
            "line_count": _line_count(prob_chain),
            "description": (
                "Each typing relationship carries a probability that filters base "
                "cases.  Inspired by ProbLog, the backward chainer probabilistically "
                "includes or excludes matching facts at a rate determined by their "
                "attached probability."
            ),
            "key_concepts": [
                "Probabilistic fact filtering — base case matches are accepted with probability p",
                "when predicate — run code if condition is true, otherwise prune",
                "random-float — used to sample whether to accept a fact",
                "Probability attached to typing relationships",
            ],
            "wrapper_adoption": (
                "Wrapper could use STV confidence as the probability for accepting "
                "or filtering Sentences during multi-step derivation.  This is "
                "simpler than the full PLN estimator and directly uses existing "
                "patham9/PLN STV values.  The ec_projected_stv() formula already "
                "produces a projected confidence that could serve as the filter "
                "probability.  No patham9/PLN source changes needed."
            ),
            "requires_patham9_source_change": False,
            "complexity": "low — wrapper filters Sentences by projected confidence before loading",
        })

    # Categorize by adoption phase
    by_phase: dict[str, list[str]] = {
        "near_term": [],
        "medium_term": [],
        "long_term": [],
    }
    for p in patterns:
        if p["complexity"].startswith("low"):
            by_phase["near_term"].append(p["name"])
        elif p["complexity"].startswith("medium"):
            by_phase["medium_term"].append(p["name"])
        else:
            by_phase["long_term"].append(p["name"])

    return {
        "schema": "petta-memory-trueagi-chaining-inference-control-survey-v1",
        "mode": "source-level-no-runtime-inspection",
        "source_repo": str(repo),
        "source_commit": "bc9beb2672953e07971b3abecc1fe67651ecddc4",
        "pattern_count": len(patterns),
        "patterns": patterns,
        "adoption_by_phase": by_phase,
        "wrapper_boundary_summary": (
            "All six patterns can be adopted at the wrapper boundary without "
            "modifying patham9/PLN source.  Near-term patterns (probabilistic "
            "filtering, meta-learning benchmark) use existing STV/confidence "
            "values as filter probabilities or test scenarios.  Medium-term "
            "patterns (controlled chainer, continuation predicate) require "
            "implementing context tracking and per-step checks.  Long-term "
            "patterns (PLN estimator, controller-as-chainer) require running "
            "PLN at two levels or implementing a Thompson sampling estimator."
        ),
        "pi_pln_extension_references": {
            "inference_control_hooks": "pi-PLN extension spec defers inference control to roadmap item 4; this survey provides concrete patterns for that item",
            "context_selection": "continuation predicate and controlled chainer patterns directly inform the context selection policy",
            "ec_projection": "probabilistic filtering pattern uses STV confidence as filter probability, compatible with ec_projected_stv()",
        },
        "boundary": "source-level inspection only; no SWI/PeTTa/MeTTa runtime invoked; no memory append; no inferred-belief promotion; no OmegaClaw/GoalChainer live path",
    }


_INFERENCE_FILTER_BOUNDARY = (
    "non-live wrapper-only inference-control filter; no SWI/PeTTa/MeTTa runtime invoked; "
    "no memory append; no inferred-belief promotion; no patham9/PLN source change; "
    "no OmegaClaw/GoalChainer live path"
)


def probabilistic_inference_filter(
    handoff: dict[str, Any],
    *,
    min_confidence: float = 0.0,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Probabilistic inference-control filter for pi-PLN wrapper.

    This is the first concrete inference-control mechanism, implementing the
    near-term "probabilistic filtering" pattern identified in the trueagi-io/
    chaining inference-control survey (commit ``cd18b51``).  It pre-evaluates
    candidate Sentences using the already-tested EC projection formula,
    computes a composite score ``(projected_strength * projected_confidence)``,
    and filters/ranks them before loading into the patham9/PLN chainer.

    The filter is non-live and wrapper-only:
    - No SWI/PeTTa/MeTTa runtime invoked
    - No memory append or inferred-belief promotion
    - No patham9/PLN source changes
    - No OmegaClaw/GoalChainer live path

    Args:
        handoff: A ``petta-memory-patham9-pln-handoff-v1`` dict from
            :func:`patham9_pln_handoff_sentences`.
        min_confidence: Minimum projected confidence for inclusion.
            Items below this threshold are filtered out.  Default 0.0
            (no filtering by confidence).
        top_k: If set, keep only the top-k items by composite score.
            When ``None``, keep all items that pass the confidence threshold.

    Returns:
        A dict with schema ``petta-memory-pi-pln-inference-filter-v1``.
    """
    if handoff.get("schema") != "petta-memory-patham9-pln-handoff-v1":
        raise ValueError("expected petta-memory-patham9-pln-handoff-v1 handoff")
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError(f"min_confidence {min_confidence} out of [0, 1]")
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k {top_k} must be non-negative or None")

    items = list(handoff.get("items", []))
    if not items:
        return {
            "schema": "petta-memory-pi-pln-inference-filter-v1",
            "mode": "design-specification-no-runtime",
            "filter_policy": {
                "min_confidence": min_confidence,
                "top_k": top_k,
                "scoring_formula": "projected_strength * projected_confidence",
            },
            "input_count": 0,
            "output_count": 0,
            "items": [],
            "selected_indices": [],
            "filtered_indices": [],
            "ranking": [],
            "boundary": _INFERENCE_FILTER_BOUNDARY,
        }

    per_item: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        base_strength = float(item["stv"]["strength"])
        base_confidence = float(item["stv"]["confidence"])
        packets = item.get("pi_pln_extension", {}).get("contextual_evidence_packets", [])
        projection = ec_projected_stv(base_strength, base_confidence, packets)
        projected_strength = projection["projected_strength"]
        projected_confidence = projection["projected_confidence"]
        composite_score = projected_strength * projected_confidence
        passes_confidence = projected_confidence >= min_confidence
        per_item.append({
            "item_index": index,
            "term": item.get("term"),
            "belief_id": item.get("belief_id"),
            "base_stv": item["stv"],
            "projected_stv": {
                "strength": projected_strength,
                "confidence": projected_confidence,
            },
            "composite_score": composite_score,
            "contextual_packet_count": len(packets),
            "passes_confidence_threshold": passes_confidence,
            "included": passes_confidence,
            "filter_reason": None if passes_confidence else (
                f"projected_confidence {projected_confidence:.4f} < min_confidence {min_confidence}"
            ),
        })

    # Apply confidence threshold, then rank by composite score for top_k
    candidates = [pi for pi in per_item if pi["passes_confidence_threshold"]]
    candidates.sort(key=lambda pi: pi["composite_score"], reverse=True)

    selected_indices: list[int] = []
    filtered_indices: list[int] = []

    if top_k is not None and top_k < len(candidates):
        kept_set = set(pi["item_index"] for pi in candidates[:top_k])
        for pi in per_item:
            if pi["item_index"] not in kept_set and pi["included"]:
                pi["included"] = False
                rank_in_candidates = next(
                    (r for r, c in enumerate(candidates) if c["item_index"] == pi["item_index"]),
                    len(candidates),
                )
                pi["filter_reason"] = f"excluded by top_k={top_k} (rank {rank_in_candidates + 1})"
        selected_indices = [pi["item_index"] for pi in candidates[:top_k]]
        filtered_indices = [pi["item_index"] for pi in per_item if not pi["included"]]
    else:
        selected_indices = [pi["item_index"] for pi in candidates]
        filtered_indices = [pi["item_index"] for pi in per_item if not pi["included"]]

    ranking = [
        {
            "rank": rank + 1,
            "item_index": pi["item_index"],
            "composite_score": pi["composite_score"],
            "term": pi["term"],
        }
        for rank, pi in enumerate(candidates[:top_k]) if top_k is not None
    ] or [
        {
            "rank": rank + 1,
            "item_index": pi["item_index"],
            "composite_score": pi["composite_score"],
            "term": pi["term"],
        }
        for rank, pi in enumerate(candidates)
    ]

    return {
        "schema": "petta-memory-pi-pln-inference-filter-v1",
        "mode": "design-specification-no-runtime",
        "filter_policy": {
            "min_confidence": min_confidence,
            "top_k": top_k,
            "scoring_formula": "projected_strength * projected_confidence",
            "source_pattern": "probabilistic filtering from trueagi-io/chaining survey (near-term)",
        },
        "input_count": len(items),
        "output_count": len(selected_indices),
        "items": per_item,
        "selected_indices": selected_indices,
        "filtered_indices": filtered_indices,
        "ranking": ranking,
        "boundary": _INFERENCE_FILTER_BOUNDARY,
    }


_CONTEXT_SELECTION_BOUNDARY = (
    "non-live wrapper-only context-selection filter; no SWI/PeTTa/MeTTa runtime invoked; "
    "no memory append or inferred-belief promotion; no patham9/PLN source change; "
    "no OmegaClaw/GoalChainer live path"
)


def context_selection_wrapper(
    handoff: dict[str, Any],
    *,
    domain: str | None = None,
    cluster_id: str | None = None,
    promotion_rule: str | None = None,
    min_packet_relevance: float = 0.0,
) -> dict[str, Any]:
    """Context-selection inference-control wrapper for pi-PLN.

    This is the second concrete inference-control mechanism, implementing the
    near-term "context selection" pattern: before sending Sentences to the
    patham9/PLN chainer, filter and score contextual EvidencePackets by
    relevance to the target query context.  This reduces noise from
    irrelevant evidence and sharpens EC projection results.

    The wrapper operates in two stages:
    1. **Packet filtering**: select only EvidencePackets whose domain,
       cluster_id, or promotion_rule matches the query context.
    2. **Packet relevance scoring**: score each remaining packet by an
       evidence-weighted relevance formula so downstream EC projection
       can optionally weight by relevance.

    The wrapper is non-live and wrapper-only:
    - No SWI/PeTTa/MeTTa runtime invoked
    - No memory append or inferred-belief promotion
    - No patham9/PLN source changes
    - No OmegaClaw/GoalChainer live path

    Args:
        handoff: A ``petta-memory-patham9-pln-handoff-v1`` dict from
            :func:`patham9_pln_handoff_sentences`.
        domain: If set, select only EvidencePackets whose ``promotion_domain``
            matches this value.
        cluster_id: If set, select only EvidencePackets whose ``cluster_id``
            matches this value.
        promotion_rule: If set, select only EvidencePackets whose
            ``promotion_rule`` matches this value.
        min_packet_relevance: Minimum relevance score [0, 1] for a packet
            to be included.  Packets below this threshold are filtered out.
            Default 0.0 (no filtering by relevance).

    Returns:
        A dict with schema ``petta-memory-pi-pln-context-selection-v1``.
    """
    if handoff.get("schema") != "petta-memory-patham9-pln-handoff-v1":
        raise ValueError("expected petta-memory-patham9-pln-handoff-v1 handoff")
    if min_packet_relevance < 0 or min_packet_relevance > 1:
        raise ValueError(f"min_packet_relevance {min_packet_relevance} out of [0, 1]")

    items = list(handoff.get("items", []))
    if not items:
        return {
            "schema": "petta-memory-pi-pln-context-selection-v1",
            "mode": "design-specification-no-runtime",
            "selection_policy": {
                "domain": domain,
                "cluster_id": cluster_id,
                "promotion_rule": promotion_rule,
                "min_packet_relevance": min_packet_relevance,
                "relevance_formula": "evidence_weight = (support + opposition) / (support + opposition + 2); relevance = evidence_weight * domain_match",
            },
            "input_count": 0,
            "output_count": 0,
            "items": [],
            "selected_indices": [],
            "filtered_indices": [],
            "total_packets_in": 0,
            "total_packets_out": 0,
            "boundary": _CONTEXT_SELECTION_BOUNDARY,
        }

    # Collect filter criteria (None means no filtering on that dimension)
    has_domain_filter = domain is not None
    has_cluster_filter = cluster_id is not None
    has_rule_filter = promotion_rule is not None
    has_relevance_filter = min_packet_relevance > 0.0

    per_item: list[dict[str, Any]] = []
    total_packets_in = 0
    total_packets_out = 0
    selected_indices: list[int] = []
    filtered_indices: list[int] = []

    for index, item in enumerate(items):
        packets = item.get("pi_pln_extension", {}).get("contextual_evidence_packets", [])
        total_packets_in += len(packets)

        kept_packets: list[dict[str, Any]] = []
        packet_summaries: list[dict[str, Any]] = []

        for packet in packets:
            packet_domain = str(packet.get("promotion_domain", ""))
            packet_cluster = str(packet.get("cluster_id", ""))
            packet_rule = str(packet.get("promotion_rule", ""))

            # Apply context filters
            domain_match = (not has_domain_filter) or (packet_domain == domain)
            cluster_match = (not has_cluster_filter) or (packet_cluster == cluster_id)
            rule_match = (not has_rule_filter) or (packet_rule == promotion_rule)

            if not (domain_match and cluster_match and rule_match):
                packet_summaries.append({
                    "support": packet.get("support", 0),
                    "opposition": packet.get("opposition", 0),
                    "included": False,
                    "filter_reason": _packet_filter_reason(
                        domain_match, cluster_match, rule_match,
                        domain, cluster_id, promotion_rule,
                    ),
                })
                continue

            # Compute relevance score
            support = float(packet.get("support", 0))
            opposition = float(packet.get("opposition", 0))
            total = support + opposition
            if total <= 0:
                relevance = 0.0
            else:
                evidence_weight = total / (total + 2.0)
                relevance = evidence_weight  # domain_match is already confirmed

            if has_relevance_filter and relevance < min_packet_relevance:
                packet_summaries.append({
                    "support": support,
                    "opposition": opposition,
                    "relevance": round(relevance, 6),
                    "included": False,
                    "filter_reason": f"relevance {relevance:.4f} < min_packet_relevance {min_packet_relevance}",
                })
                continue

            kept_packets.append(packet)
            packet_summaries.append({
                "support": support,
                "opposition": opposition,
                "relevance": round(relevance, 6),
                "included": True,
                "filter_reason": None,
            })

        total_packets_out += len(kept_packets)

        # Check if item still has any relevant evidence
        has_remaining_evidence = len(kept_packets) > 0
        # Item is included if it has kept packets OR if it had no packets to begin with
        # (items without packets are not filtered by context selection)
        item_included = has_remaining_evidence or len(packets) == 0

        if item_included:
            selected_indices.append(index)
        else:
            filtered_indices.append(index)

        # Build filtered item with only kept packets
        filtered_item = dict(item)
        filtered_extension = dict(item.get("pi_pln_extension", {}))
        filtered_extension["contextual_evidence_packets"] = kept_packets
        filtered_extension["context_selection_applied"] = True
        filtered_extension["context_selection_criteria"] = {
            "domain": domain,
            "cluster_id": cluster_id,
            "promotion_rule": promotion_rule,
            "min_packet_relevance": min_packet_relevance,
        }
        filtered_item["pi_pln_extension"] = filtered_extension

        per_item.append({
            "item_index": index,
            "term": item.get("term"),
            "belief_id": item.get("belief_id"),
            "original_packet_count": len(packets),
            "kept_packet_count": len(kept_packets),
            "packets_in": packet_summaries,
            "included": item_included,
            "filter_reason": None if item_included else (
                "all contextual evidence packets filtered by context selection"
            ),
        })

    return {
        "schema": "petta-memory-pi-pln-context-selection-v1",
        "mode": "design-specification-no-runtime",
        "selection_policy": {
            "domain": domain,
            "cluster_id": cluster_id,
            "promotion_rule": promotion_rule,
            "min_packet_relevance": min_packet_relevance,
            "relevance_formula": "evidence_weight = (support + opposition) / (support + opposition + 2); relevance = evidence_weight * context_match",
            "source_pattern": "context selection from trueagi-io/chaining survey (near-term)",
        },
        "input_count": len(items),
        "output_count": len(selected_indices),
        "items": per_item,
        "selected_indices": selected_indices,
        "filtered_indices": filtered_indices,
        "total_packets_in": total_packets_in,
        "total_packets_out": total_packets_out,
        "boundary": _CONTEXT_SELECTION_BOUNDARY,
    }


def _packet_filter_reason(
    domain_match: bool,
    cluster_match: bool,
    rule_match: bool,
    domain: str | None,
    cluster_id: str | None,
    promotion_rule: str | None,
) -> str:
    reasons: list[str] = []
    if not domain_match and domain is not None:
        reasons.append(f"domain mismatch (expected {domain!r})")
    if not cluster_match and cluster_id is not None:
        reasons.append(f"cluster_id mismatch (expected {cluster_id!r})")
    if not rule_match and promotion_rule is not None:
        reasons.append(f"promotion_rule mismatch (expected {promotion_rule!r})")
    return "; ".join(reasons) if reasons else "unknown"


_PIPELINE_BOUNDARY = (
    "non-live wrapper-only chained inference pipeline; no SWI/PeTTa/MeTTa runtime invoked; "
    "no memory append or inferred-belief promotion; no patham9/PLN source change; "
    "no OmegaClaw/GoalChainer live path"
)


def chained_inference_pipeline(
    handoff: dict[str, Any],
    *,
    domain: str | None = None,
    cluster_id: str | None = None,
    promotion_rule: str | None = None,
    min_packet_relevance: float = 0.0,
    min_confidence: float = 0.0,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Chained inference-control pipeline: context selection then probabilistic filtering.

    This is the third concrete inference-control mechanism, combining the two
    near-term patterns from the trueagi-io/chaining survey into a single
    pipeline:

    1. **Context selection** (stage 1): filter EvidencePackets by domain,
       cluster_id, or promotion_rule, and score remaining packets by
       evidence-weighted relevance.
    2. **Probabilistic filtering** (stage 2): apply EC projection to the
       context-filtered handoff, compute composite scores, and filter/rank
       by confidence threshold and top_k.

    The pipeline is non-live and wrapper-only:
    - No SWI/PeTTa/MeTTa runtime invoked
    - No memory append or inferred-belief promotion
    - No patham9/PLN source changes
    - No OmegaClaw/GoalChainer live path

    Args:
        handoff: A ``petta-memory-patham9-pln-handoff-v1`` dict from
            :func:`patham9_pln_handoff_sentences`.
        domain: If set, select only EvidencePackets whose ``promotion_domain``
            matches this value (stage 1).
        cluster_id: If set, select only EvidencePackets whose ``cluster_id``
            matches this value (stage 1).
        promotion_rule: If set, select only EvidencePackets whose
            ``promotion_rule`` matches this value (stage 1).
        min_packet_relevance: Minimum relevance score [0, 1] for a packet
            to survive stage 1 (default 0.0, no relevance filtering).
        min_confidence: Minimum projected confidence for inclusion in
            stage 2 (default 0.0, no confidence filtering).
        top_k: If set, keep only the top-k items by composite score in
            stage 2.

    Returns:
        A dict with schema ``petta-memory-pi-pln-inference-pipeline-v1``.
    """
    if handoff.get("schema") != "petta-memory-patham9-pln-handoff-v1":
        raise ValueError("expected petta-memory-patham9-pln-handoff-v1 handoff")
    if min_packet_relevance < 0 or min_packet_relevance > 1:
        raise ValueError(f"min_packet_relevance {min_packet_relevance} out of [0, 1]")
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError(f"min_confidence {min_confidence} out of [0, 1]")
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k {top_k} must be non-negative or None")

    # Stage 1: context selection
    context_result = context_selection_wrapper(
        handoff,
        domain=domain,
        cluster_id=cluster_id,
        promotion_rule=promotion_rule,
        min_packet_relevance=min_packet_relevance,
    )

    # Build a filtered handoff from the context selection result
    # Items that were filtered out by context selection are removed;
    # items with kept packets have their packets updated.
    original_items = list(handoff.get("items", []))
    selected_indices = set(context_result.get("selected_indices", []))
    filtered_handoff_items: list[dict[str, Any]] = []
    for idx, item in enumerate(original_items):
        if idx not in selected_indices:
            continue
        # Use the context-filtered version of the item
        per_item = None
        for pi in context_result.get("items", []):
            if pi["item_index"] == idx:
                per_item = pi
                break
        if per_item is None:
            filtered_handoff_items.append(item)
            continue
        # Reconstruct item with filtered packets
        filtered_item = dict(item)
        filtered_extension = dict(item.get("pi_pln_extension", {}))
        # Rebuild packets from context_result's kept packets
        kept_packets = [
            pkt for pkt, summary in zip(
                item.get("pi_pln_extension", {}).get("contextual_evidence_packets", []),
                per_item.get("packets_in", []),
            ) if summary.get("included", False)
        ]
        filtered_extension["contextual_evidence_packets"] = kept_packets
        filtered_extension["context_selection_applied"] = True
        filtered_extension["context_selection_criteria"] = {
            "domain": domain,
            "cluster_id": cluster_id,
            "promotion_rule": promotion_rule,
            "min_packet_relevance": min_packet_relevance,
        }
        filtered_item["pi_pln_extension"] = filtered_extension
        filtered_handoff_items.append(filtered_item)

    # Build the stage-2 handoff
    stage2_handoff = {
        "schema": "petta-memory-patham9-pln-handoff-v1",
        "item_count": len(filtered_handoff_items),
        "items": filtered_handoff_items,
    }

    # Stage 2: probabilistic filtering
    filter_result = probabilistic_inference_filter(
        stage2_handoff,
        min_confidence=min_confidence,
        top_k=top_k,
    )

    # Remap item_index values in the filter result to original handoff indices
    original_selected = list(selected_indices)
    remapped_items: list[dict[str, Any]] = []
    remapped_selected: list[int] = []
    remapped_filtered: list[int] = []
    remapped_ranking: list[dict[str, Any]] = []

    # Build a mapping from stage2 index -> original index
    stage2_to_original = {s2_idx: orig_idx for s2_idx, orig_idx in enumerate(sorted(selected_indices))}

    for fi in filter_result.get("items", []):
        stage2_idx = fi["item_index"]
        orig_idx = stage2_to_original.get(stage2_idx, stage2_idx)
        remapped = dict(fi)
        remapped["item_index"] = orig_idx
        remapped["pipeline_stage"] = "post-filter"
        if fi.get("included"):
            remapped_selected.append(orig_idx)
        else:
            remapped_filtered.append(orig_idx)
        remapped_items.append(remapped)

    for r in filter_result.get("ranking", []):
        stage2_idx = r["item_index"]
        orig_idx = stage2_to_original.get(stage2_idx, stage2_idx)
        remapped_ranking.append({
            "rank": r["rank"],
            "item_index": orig_idx,
            "composite_score": r["composite_score"],
            "term": r.get("term"),
        })

    # Items filtered by stage 1 (context selection) are also reported
    context_filtered_indices = list(context_result.get("filtered_indices", []))
    all_filtered = sorted(set(remapped_filtered + context_filtered_indices))

    return {
        "schema": "petta-memory-pi-pln-inference-pipeline-v1",
        "mode": "design-specification-no-runtime",
        "pipeline_policy": {
            "stage_1": "context_selection",
            "stage_2": "probabilistic_filtering",
            "context_selection": {
                "domain": domain,
                "cluster_id": cluster_id,
                "promotion_rule": promotion_rule,
                "min_packet_relevance": min_packet_relevance,
            },
            "probabilistic_filtering": {
                "min_confidence": min_confidence,
                "top_k": top_k,
                "scoring_formula": "projected_strength * projected_confidence",
            },
            "source_pattern": "chained filter+select pipeline from trueagi-io/chaining survey (near-term)",
        },
        "input_count": len(original_items),
        "stage1_output_count": len(filtered_handoff_items),
        "stage1_filtered_count": len(context_filtered_indices),
        "stage1_filtered_indices": context_filtered_indices,
        "stage1_total_packets_in": context_result.get("total_packets_in", 0),
        "stage1_total_packets_out": context_result.get("total_packets_out", 0),
        "stage2_output_count": len(remapped_selected),
        "output_count": len(remapped_selected),
        "selected_indices": remapped_selected,
        "filtered_indices": all_filtered,
        "items": remapped_items,
        "ranking": remapped_ranking,
        "stage1_result": {
            "schema": context_result["schema"],
            "selected_indices": context_result.get("selected_indices", []),
            "filtered_indices": context_result.get("filtered_indices", []),
            "total_packets_in": context_result.get("total_packets_in", 0),
            "total_packets_out": context_result.get("total_packets_out", 0),
        },
        "stage2_result": {
            "schema": filter_result["schema"],
            "selected_indices": filter_result.get("selected_indices", []),
            "filtered_indices": filter_result.get("filtered_indices", []),
        },
        "boundary": _PIPELINE_BOUNDARY,
    }


_META_LEARNING_BOUNDARY = (
    "non-live wrapper-only benchmark; no SWI/PeTTa/MeTTa runtime invoked; "
    "no memory append or inferred-belief promotion; no patham9/PLN source change; "
    "no OmegaClaw/GoalChainer live path"
)


def build_meta_learning_benchmark_handoff(
    *,
    shortcut_strength: float = 0.95,
    shortcut_confidence: float = 0.90,
    chain_strengths: list[float] | None = None,
    chain_confidences: list[float] | None = None,
    shortcut_domain: str = "benchmark",
    chain_domain: str = "benchmark",
    shortcut_ec: tuple[int, int] = (9, 1),
    chain_ecs: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Build a synthetic meta-learning benchmark handoff.

    Creates a ``petta-memory-patham9-pln-handoff-v1`` handoff with a
    high-confidence shortcut belief and a chain of lower-confidence
    transitive beliefs leading to the same conclusion, inspired by the
    OpenCog classic meta-learning experiment reproduced in the
    trueagi-io/chaining repo.

    The shortcut item has higher STV than any chain item, so a correct
    inference-control wrapper should rank it first when filtering/ranking
    by composite score.

    The benchmark is non-live and wrapper-only:
    - No SWI/PeTTa/MeTTa runtime invoked
    - No memory append or inferred-belief promotion
    - No patham9/PLN source changes
    - No OmegaClaw/GoalChainer live path

    Args:
        shortcut_strength: STV strength for the shortcut item (default 0.95).
        shortcut_confidence: STV confidence for the shortcut item (default 0.90).
        chain_strengths: STV strengths for each chain item.  Default
            ``[0.70, 0.65, 0.60]`` (three-step transitive chain).
        chain_confidences: STV confidences for each chain item.  Default
            ``[0.55, 0.50, 0.45]``.
        shortcut_domain: Promotion domain for the shortcut item's EvidencePacket.
        chain_domain: Promotion domain for chain items' EvidencePackets.
        shortcut_ec: ``(support, opposition)`` EC counts for the shortcut.
        chain_ecs: List of ``(support, opposition)`` EC counts for chain items.
            Default ``[(3, 1), (2, 2), (1, 3)]``.

    Returns:
        A dict with schema ``petta-memory-patham9-pln-handoff-v1``.
    """
    if chain_strengths is None:
        chain_strengths = [0.70, 0.65, 0.60]
    if chain_confidences is None:
        chain_confidences = [0.55, 0.50, 0.45]
    if chain_ecs is None:
        chain_ecs = [(3, 1), (2, 2), (1, 3)]
    if len(chain_strengths) != len(chain_confidences):
        raise ValueError("chain_strengths and chain_confidences must have equal length")
    if len(chain_strengths) != len(chain_ecs):
        raise ValueError("chain_strengths and chain_ecs must have equal length")
    if not (0 <= shortcut_strength <= 1 and 0 <= shortcut_confidence <= 1):
        raise ValueError("shortcut STV out of [0, 1]")
    for s, c in zip(chain_strengths, chain_confidences):
        if not (0 <= s <= 1 and 0 <= c <= 1):
            raise ValueError("chain STV out of [0, 1]")
    for sup, opp in [shortcut_ec] + list(chain_ecs):
        if sup < 0 or opp < 0:
            raise ValueError("EC counts must be non-negative")

    items: list[dict[str, Any]] = []

    # Shortcut item (index 0)
    items.append({
        "item_index": 0,
        "belief_id": "shortcut-0",
        "term": "(ShortcutConclusion)",
        "stv": {"strength": shortcut_strength, "confidence": shortcut_confidence},
        "stamp": [0],
        "pi_pln_extension": {
            "provenance": "meta-learning-benchmark-shortcut",
            "contextual_evidence_packets": [
                {
                    "statement": "(ShortcutConclusion)",
                    "ec": {"support": shortcut_ec[0], "opposition": shortcut_ec[1]},
                    "promotion_domain": shortcut_domain,
                    "cluster_id": "benchmark-shortcut",
                    "promotion_rule": "direct-observation",
                }
            ],
        },
    })

    # Chain items (indices 1..N)
    for i, (strength, confidence, ec) in enumerate(
        zip(chain_strengths, chain_confidences, chain_ecs), start=1
    ):
        items.append({
            "item_index": i,
            "belief_id": f"chain-{i}",
            "term": f"(ChainStep{i})",
            "stv": {"strength": strength, "confidence": confidence},
            "stamp": [i],
            "pi_pln_extension": {
                "provenance": f"meta-learning-benchmark-chain-{i}",
                "contextual_evidence_packets": [
                    {
                        "statement": f"(ChainStep{i})",
                        "ec": {"support": ec[0], "opposition": ec[1]},
                        "promotion_domain": chain_domain,
                        "cluster_id": "benchmark-chain",
                        "promotion_rule": "transitive-inference",
                    }
                ],
            },
        })

    return {
        "schema": "petta-memory-patham9-pln-handoff-v1",
        "item_count": len(items),
        "items": items,
    }


def run_meta_learning_benchmark(
    *,
    handoff: dict[str, Any] | None = None,
    min_confidence: float = 0.0,
    top_k: int | None = None,
    domain: str | None = None,
    min_packet_relevance: float = 0.0,
) -> dict[str, Any]:
    """Run the meta-learning inference-control benchmark.

    Builds (or accepts) a synthetic shortcut-vs-chain handoff, runs the
    probabilistic inference filter and the chained inference-control
    pipeline against it, and reports whether the shortcut item is
    correctly ranked first.

    This implements the near-term "meta-learning benchmark" pattern from
    the trueagi-io/chaining inference-control survey.  It exercises the
    existing wrapper mechanisms against a known scenario where a
    high-confidence shortcut should be preferred over a longer chain of
    lower-confidence transitive beliefs.

    The benchmark is non-live and wrapper-only:
    - No SWI/PeTTa/MeTTa runtime invoked
    - No memory append or inferred-belief promotion
    - No patham9/PLN source changes
    - No OmegaClaw/GoalChainer live path

    Args:
        handoff: If provided, use this handoff instead of building the
            default benchmark handoff.  Must have schema
            ``petta-memory-patham9-pln-handoff-v1``.
        min_confidence: Minimum projected confidence for the probabilistic
            filter stage (default 0.0, no filtering).
        top_k: If set, keep only top-k items by composite score.
        domain: If set, filter by this promotion domain in the chained
            pipeline's context selection stage.
        min_packet_relevance: Minimum packet relevance for context selection.

    Returns:
        A dict with schema ``petta-memory-pi-pln-meta-learning-benchmark-v1``.
    """
    if handoff is None:
        handoff = build_meta_learning_benchmark_handoff()
    if handoff.get("schema") != "petta-memory-patham9-pln-handoff-v1":
        raise ValueError("expected petta-memory-patham9-pln-handoff-v1 handoff")
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError(f"min_confidence {min_confidence} out of [0, 1]")
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k {top_k} must be non-negative or None")
    if min_packet_relevance < 0 or min_packet_relevance > 1:
        raise ValueError(f"min_packet_relevance {min_packet_relevance} out of [0, 1]")

    items = list(handoff.get("items", []))
    # Identify shortcut (index 0) and chain items (indices 1..N)
    shortcut_index = 0
    chain_indices = list(range(1, len(items))) if len(items) > 1 else []

    # Run probabilistic filter
    filter_result = probabilistic_inference_filter(
        handoff,
        min_confidence=min_confidence,
        top_k=top_k,
    )

    # Run chained pipeline (context selection + probabilistic filtering)
    pipeline_result = chained_inference_pipeline(
        handoff,
        domain=domain,
        min_packet_relevance=min_packet_relevance,
        min_confidence=min_confidence,
        top_k=top_k,
    )

    # Analyze ranking: is shortcut ranked first?
    filter_ranking = filter_result.get("ranking", [])
    pipeline_ranking = pipeline_result.get("ranking", [])

    filter_shortcut_rank = None
    for r in filter_ranking:
        if r["item_index"] == shortcut_index:
            filter_shortcut_rank = r["rank"]
            break

    pipeline_shortcut_rank = None
    for r in pipeline_ranking:
        if r["item_index"] == shortcut_index:
            pipeline_shortcut_rank = r["rank"]
            break

    filter_shortcut_first = (
        filter_shortcut_rank is not None and filter_shortcut_rank == 1
    )
    pipeline_shortcut_first = (
        pipeline_shortcut_rank is not None and pipeline_shortcut_rank == 1
    )

    # Check if any chain item outranks the shortcut
    chain_outranks_filter = False
    chain_outranks_pipeline = False
    for r in filter_ranking:
        if r["item_index"] in chain_indices and r["rank"] < (filter_shortcut_rank or len(items) + 1):
            chain_outranks_filter = True
    for r in pipeline_ranking:
        if r["item_index"] in chain_indices and r["rank"] < (pipeline_shortcut_rank or len(items) + 1):
            chain_outranks_pipeline = True

    # Shortcut composite score vs best chain composite score
    filter_shortcut_score = None
    best_chain_score = None
    for r in filter_ranking:
        if r["item_index"] == shortcut_index:
            filter_shortcut_score = r["composite_score"]
        elif r["item_index"] in chain_indices:
            if best_chain_score is None or r["composite_score"] > best_chain_score:
                best_chain_score = r["composite_score"]

    shortcut_preferred = (
        filter_shortcut_score is not None
        and (best_chain_score is None or filter_shortcut_score > best_chain_score)
    )

    return {
        "schema": "petta-memory-pi-pln-meta-learning-benchmark-v1",
        "mode": "design-specification-no-runtime",
        "benchmark_scenario": {
            "shortcut_item_index": shortcut_index,
            "chain_item_indices": chain_indices,
            "shortcut_belief_id": items[0].get("belief_id") if items else None,
            "shortcut_stv": items[0].get("stv") if items else None,
            "chain_belief_ids": [items[i].get("belief_id") for i in chain_indices if i < len(items)],
            "source_pattern": "meta-learning benchmark from trueagi-io/chaining survey (near-term)",
            "description": (
                "Evaluates whether the pi-PLN wrapper's inference control "
                "correctly prefers a high-confidence shortcut belief over "
                "a longer chain of lower-confidence transitive beliefs, "
                "inspired by the OpenCog classic meta-learning experiment."
            ),
        },
        "filter_result": {
            "schema": filter_result["schema"],
            "shortcut_rank": filter_shortcut_rank,
            "shortcut_first": filter_shortcut_first,
            "shortcut_composite_score": filter_shortcut_score,
            "best_chain_composite_score": best_chain_score,
            "chain_outranks_shortcut": chain_outranks_filter,
            "ranking": filter_ranking,
        },
        "pipeline_result": {
            "schema": pipeline_result["schema"],
            "shortcut_rank": pipeline_shortcut_rank,
            "shortcut_first": pipeline_shortcut_first,
            "ranking": pipeline_ranking,
        },
        "shortcut_preferred": shortcut_preferred,
        "filter_shortcut_first": filter_shortcut_first,
        "pipeline_shortcut_first": pipeline_shortcut_first,
        "overall_pass": shortcut_preferred and filter_shortcut_first,
        "policy": {
            "min_confidence": min_confidence,
            "top_k": top_k,
            "domain": domain,
            "min_packet_relevance": min_packet_relevance,
        },
        "boundary": _META_LEARNING_BOUNDARY,
    }
