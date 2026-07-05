from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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
