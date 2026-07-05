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
