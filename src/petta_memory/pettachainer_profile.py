from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable

from .sexpr import parse_one_list, symbol_text, to_source
from .store import MediumMemoryStore


def build_promoted_cluster(index: int, *, support: float = 3.0, opposition: float = 1.0) -> str:
    """Return one promoted-belief cluster shaped for PeTTaChainer profiling.

    The generated workload is intentionally narrow and OmegaClaw-like: each
    cluster promotes one memory-readiness belief with explicit STV plus EC counts,
    so the same journal can exercise proof-statement export and EvidencePacket
    export without inventing evidence counts from truth values.
    """
    suffix = f"profile-{index:03d}"
    strength = min(0.99, 0.70 + (index % 10) / 100)
    confidence = min(0.95, 0.60 + (index % 7) / 100)
    trust = min(0.90, 0.55 + (index % 5) / 100)
    return f"""
(MemoryCluster mc-{suffix})
(SchemaVersion mc-{suffix} medium-memory-v1)
(ClusterType mc-{suffix} belief-promotion)
(ClusterOpenedAt mc-{suffix} "2026-07-02 profile fixture")
(ClusterSource mc-{suffix} local-profile-generator)
(Contains mc-{suffix} pe-{suffix})
(Contains mc-{suffix} b-{suffix})
(ClusterStatus mc-{suffix} active)
(PromotionEvent pe-{suffix})
(PromotesFrom pe-{suffix} qc-{suffix})
(PromotesTo pe-{suffix} b-{suffix})
(PromotionRule pe-{suffix} explicit-profile-workload)
(PromotionTrust pe-{suffix} {trust:.2f})
(PromotionDomain pe-{suffix} omegaclaw-memory)
(DerivedBelief b-{suffix})
(BeliefContent b-{suffix} (Requires MemoryTarget{index} PLNReadyViews))
(TruthValue b-{suffix} (stv {strength:.2f} {confidence:.2f}))
(EvidenceFor b-{suffix} qc-{suffix})
(EvidenceSupportCount b-{suffix} {support + index:.1f})
(EvidenceOppositionCount b-{suffix} {opposition + (index % 3):.1f})
"""


def build_profile_store(path: str | Path, count: int) -> MediumMemoryStore:
    if count < 0:
        raise ValueError("count must be non-negative")
    store = MediumMemoryStore(path)
    for index in range(count):
        store.append_cluster(build_promoted_cluster(index))
    return store


def _time_call(label: str, fn: Callable[[], object]) -> dict[str, object]:
    started = time.perf_counter()
    result = fn()
    return {"label": label, "seconds": round(time.perf_counter() - started, 6), "status": "ok", "result": result}


def _run_isolated_stage(
    label: str,
    target: Callable[..., dict[str, object]],
    args: tuple[object, ...],
    *,
    stage_timeout_sec: float,
) -> dict[str, object]:
    """Run one noisy/slow PeTTaChainer stage in a subprocess.

    The PeTTaChainer runtime can emit large compile traces or spend a long time
    compiling rules. Profiling should record that as a bounded result instead of
    letting a cron worker hang, so each optional runtime stage gets a hard wall
    clock timeout and captured stdout/stderr.
    """
    if stage_timeout_sec <= 0:
        raise ValueError("stage_timeout_sec must be positive")
    started = time.perf_counter()
    result_queue: mp.Queue[dict[str, object]] = mp.Queue(maxsize=1)
    process = mp.Process(target=_isolated_stage_worker, args=(target, args, result_queue))
    process.start()
    process.join(stage_timeout_sec)
    elapsed = round(time.perf_counter() - started, 6)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return {"label": label, "seconds": elapsed, "status": "timeout", "timeout_sec": stage_timeout_sec}
    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        payload = {"status": "error", "error": f"child exited with code {process.exitcode} without a result"}
    payload.setdefault("status", "ok" if process.exitcode == 0 else "error")
    payload.update({"label": label, "seconds": elapsed})
    return payload


def _isolated_stage_worker(
    target: Callable[..., dict[str, object]],
    args: tuple[object, ...],
    result_queue: "mp.Queue[dict[str, object]]",
) -> None:
    # PeTTaChainer/SWI-Prolog writes below Python's sys.stdout layer, so capture
    # OS file descriptors rather than only using contextlib.redirect_stdout.
    stdout_original = os.dup(1)
    stderr_original = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)
            try:
                payload = target(*args)
                payload.setdefault("status", "ok")
            except BaseException as exc:  # pragma: no cover - exercised through parent status in tests.
                payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stdout_original, 1)
                os.dup2(stderr_original, 2)
        finally:
            os.close(stdout_original)
            os.close(stderr_original)
        stdout_file.seek(0, os.SEEK_END)
        stderr_file.seek(0, os.SEEK_END)
        payload["stdout_chars"] = stdout_file.tell()
        payload["stderr_chars"] = stderr_file.tell()
    result_queue.put(payload)


def _check_statements_stage(statements: list[str]) -> dict[str, object]:
    from pettachainer import check_stmt

    return {"result": [check_stmt(stmt) for stmt in statements]}


def _pettachainer_init_stage() -> dict[str, object]:
    from pettachainer import PeTTaChainer

    init_event = _time_call("construct_pettachainer", lambda: PeTTaChainer())
    init_event["result"] = "initialized"
    return {"result": "initialized", "stages": [init_event]}


def _proof_add_stage(statements: list[str]) -> dict[str, object]:
    from pettachainer import PeTTaChainer

    handler = PeTTaChainer()
    return {"stages": [_time_call("add_proof_statements_no_check", lambda: handler.add_atoms_no_check(statements))]}


def _compileadd_probe_expressions(statement: str, kb: str) -> dict[str, str]:
    stmt1 = f"(materialize-stmt-lambdas {statement})"
    atoms = f"(collapse (mm2compile {kb} {stmt1}))"
    iatoms = f"(list_to_set (map-flat internalize-proof-structure {atoms}))"
    return {
        "materialize_stmt_lambdas": stmt1,
        "mm2compile_collapse": atoms,
        "internalize_proof_structure": iatoms,
        "externalize_proof_structure": f"(map-flat externalize-proof-structure {iatoms})",
        "index_source_implication": f"(index-source-implication {kb} {stmt1})",
        "add_internalized_atoms": f"(collapse (foreach-flat add-to-kb {iatoms}))",
        "maybe_process_on_add": f"(maybe-process-on-add {kb} {stmt1})",
    }


def _compileadd_probe_call_text(statement: str, kb: str, probe: str, invocation: str) -> str:
    """Return the exact MeTTa call used for an internal ``compileadd`` probe.

    ``compileadd`` itself invokes its subforms directly inside a ``let*``. The
    first probe version wrapped each subform in ``eval``, which may evaluate the
    materialized statement rather than just timing the subform. Keeping both call
    modes available lets profile artifacts distinguish genuine subform cost from
    an eval/probe artifact.
    """
    expressions = _compileadd_probe_expressions(statement, kb)
    if probe not in expressions:
        raise ValueError(f"unknown compileadd probe: {probe}")
    if invocation == "direct":
        return f"!{expressions[probe]}"
    if invocation == "eval":
        return f"!(eval {expressions[probe]})"
    raise ValueError(f"unknown compileadd probe invocation: {invocation}")


def _compileadd_probe_stage(statement: str, probe: str, invocation: str = "direct") -> dict[str, object]:
    """Time one internal expression from PeTTaChainer's ``compileadd`` path.

    Each probe is intentionally isolated in its own subprocess by the caller. If
    one internal form hangs, the surrounding profile still records which substep
    hit the hard timeout instead of collapsing the whole add path into a single
    opaque ``compileadd`` timeout.
    """
    from pettachainer import PeTTaChainer

    handler = PeTTaChainer()
    call_text = _compileadd_probe_call_text(statement, handler.kb, probe, invocation)
    return {
        "probe": probe,
        "invocation": invocation,
        "stages": [_time_call(f"{probe}_{invocation}", lambda: handler.handler.process_metta_string(call_text))],
    }


def _sexpr_equal_allow_numeric_rendering(left: object, right: object) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _sexpr_equal_allow_numeric_rendering(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, tuple) or isinstance(right, tuple):
        return False
    if left == right:
        return True
    try:
        return float(str(left)) == float(str(right))
    except ValueError:
        return False


def _materialize_identity_matches(statement: str, outputs: object) -> bool:
    output_items = outputs if isinstance(outputs, list) else [outputs]
    output_text = "\n".join(str(item) for item in output_items)
    if statement in output_text:
        return True
    expected = parse_one_list(statement)
    for item in output_items:
        try:
            if _sexpr_equal_allow_numeric_rendering(expected, parse_one_list(str(item))):
                return True
        except ValueError:
            continue
    return False


def _materialize_identity_stage(statement: str) -> dict[str, object]:
    """Run only ``materialize-stmt-lambdas`` and compare output structurally.

    This is narrower than the generic compileadd probe: it is meant for a
    lambda-free statement whose source inspection predicts identity
    materialization, so the stage records whether the runtime output matches the
    original statement.  Numeric formatting differences such as ``0.70`` versus
    ``0.7`` are treated as identity-preserving because PeTTa's renderer may
    normalize floats.  The caller still runs the stage in a bounded subprocess.
    """
    from pettachainer import PeTTaChainer

    handler = PeTTaChainer()
    call_text = f"!(materialize-stmt-lambdas {statement})"

    def run() -> list[str]:
        result = handler.handler.process_metta_string(call_text)
        if isinstance(result, list):
            return [str(item) for item in result]
        return [str(result)]

    event = _time_call("materialize_stmt_lambdas_identity", run)
    outputs = event.get("result", [])
    return {
        "call_text": call_text,
        "expected_statement": statement,
        "identity_output_present": _materialize_identity_matches(statement, outputs),
        "stages": [event],
    }


def _compileadd_probe_specs(statements: list[str]) -> list[tuple[str, str, str]]:
    if not statements:
        return []
    # Direct probes mirror PeTTaChainer's compileadd let* path. The two legacy
    # eval probes are retained narrowly as controls for the previous timeout
    # artifact hypothesis, without doubling every expensive subform.
    return [
        ("compileadd_probe_materialize_direct", "materialize_stmt_lambdas", "direct"),
        ("compileadd_probe_materialize_eval_control", "materialize_stmt_lambdas", "eval"),
        ("compileadd_probe_mm2compile_direct", "mm2compile_collapse", "direct"),
        ("compileadd_probe_mm2compile_eval_control", "mm2compile_collapse", "eval"),
        ("compileadd_probe_internalize_direct", "internalize_proof_structure", "direct"),
        ("compileadd_probe_externalize_direct", "externalize_proof_structure", "direct"),
        ("compileadd_probe_index_source_direct", "index_source_implication", "direct"),
        ("compileadd_probe_add_internalized_direct", "add_internalized_atoms", "direct"),
        ("compileadd_probe_maybe_process_on_add_direct", "maybe_process_on_add", "direct"),
    ]


def _proof_runtime_stage(statements: list[str], steps: int, timeout_sec: float) -> dict[str, object]:
    from pettachainer import PeTTaChainer

    handler = PeTTaChainer()
    stages: list[dict[str, object]] = []
    stages.append(_time_call("add_proof_statements_no_check", lambda: handler.add_atoms_no_check(statements)))
    query = "(: $proof (Requires MemoryTarget0 PLNReadyViews) $tv)"
    stages.append(_time_call("query_first_target", lambda: handler.query(query, steps=steps, timeout_sec=timeout_sec)))
    return {"stages": stages}


def _contextual_add_stage(packets: list[str]) -> dict[str, object]:
    from pettachainer import PeTTaChainer

    handler = PeTTaChainer()
    return {"stages": [_time_call("add_evidence_packets_no_check", lambda: handler.add_atoms_no_check(packets))]}


def _contextual_runtime_stage(packets: list[str], steps: int, timeout_sec: float) -> dict[str, object]:
    from pettachainer import PeTTaChainer

    handler = PeTTaChainer()
    stages: list[dict[str, object]] = []
    stages.append(_time_call("add_evidence_packets_no_check", lambda: handler.add_atoms_no_check(packets)))
    query = "(: $proof (Requires MemoryTarget0 PLNReadyViews) $tv)"
    stages.append(
        _time_call(
            "contextual_query_first_target",
            lambda: handler.contextual_query(query, steps=steps, timeout_sec=timeout_sec).answers,
        )
    )
    return {"stages": stages}


def _configure_local_runtime(project_root: Path) -> None:
    pettachainer = project_root / "repos" / "PeTTaChainer"
    petta = project_root / "repos" / "PeTTa"
    workspace = project_root.parents[1]
    swi_prefix = workspace / "projects" / "omegaclaw" / "local" / "swipl-9.3.36"
    venvs = sorted((pettachainer / ".venv" / "lib").glob("python*/site-packages"))
    missing = [
        str(path)
        for path in (pettachainer / "pettachainer", petta / "python", swi_prefix)
        if not path.exists()
    ]
    if not venvs:
        missing.append(str(pettachainer / ".venv" / "lib" / "python*/site-packages"))
    if missing:
        raise RuntimeError("local PeTTaChainer/SWI runtime is unavailable: " + ", ".join(missing))
    os.environ.setdefault("SWIPL_HOME", str(swi_prefix))
    os.environ.setdefault("SWI_HOME_DIR", str(swi_prefix / "lib" / "swipl"))
    os.environ["PATH"] = f"{swi_prefix / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"{swi_prefix / 'lib' / 'swipl' / 'lib' / 'x86_64-linux'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    for path in [str(venvs[-1]), str(pettachainer), str(petta / "python")]:
        if path not in sys.path:
            sys.path.insert(0, path)


def profile_sizes(
    sizes: Iterable[int],
    *,
    steps: int,
    timeout_sec: float,
    project_root: Path,
    stage_timeout_sec: float = 30.0,
    include_runtime_add: bool = False,
    include_contextual: bool = False,
) -> dict[str, object]:
    _configure_local_runtime(project_root)

    results: list[dict[str, object]] = []
    for count in sizes:
        with tempfile.TemporaryDirectory() as td:
            store_path = Path(td) / "medium_memory.metta"
            row: dict[str, object] = {
                "clusters": count,
                "steps": steps,
                "timeout_sec": timeout_sec,
                "stage_timeout_sec": stage_timeout_sec,
            }
            events: list[dict[str, object]] = []
            build_event = _time_call("build_store_and_exports", lambda: _build_export_payload(store_path, count))
            payload = build_event.pop("result")
            events.append(build_event)
            statements = payload["statements"]
            packets = payload["packets"]
            row.update({"statement_count": len(statements), "packet_count": len(packets)})
            events.append(
                _run_isolated_stage(
                    "check_stmt_all",
                    _check_statements_stage,
                    (statements,),
                    stage_timeout_sec=stage_timeout_sec,
                )
            )
            if include_runtime_add:
                events.append(
                    _run_isolated_stage(
                        "pettachainer_init_only",
                        _pettachainer_init_stage,
                        (),
                        stage_timeout_sec=stage_timeout_sec,
                    )
                )
                for label, probe, invocation in _compileadd_probe_specs(statements):
                    events.append(
                        _run_isolated_stage(
                            label,
                            _compileadd_probe_stage,
                            (statements[0], probe, invocation),
                            stage_timeout_sec=stage_timeout_sec,
                        )
                    )
                events.append(
                    _run_isolated_stage(
                        "proof_runtime_add_only",
                        _proof_add_stage,
                        (statements,),
                        stage_timeout_sec=stage_timeout_sec,
                    )
                )
                events.append(
                    _run_isolated_stage(
                        "proof_runtime_add_and_query",
                        _proof_runtime_stage,
                        (statements, steps, timeout_sec),
                        stage_timeout_sec=stage_timeout_sec,
                    )
                )
                if include_contextual:
                    events.append(
                        _run_isolated_stage(
                            "contextual_packet_add_only",
                            _contextual_add_stage,
                            (packets,),
                            stage_timeout_sec=stage_timeout_sec,
                        )
                    )
                    events.append(
                        _run_isolated_stage(
                            "contextual_runtime_add_and_query",
                            _contextual_runtime_stage,
                            (packets, steps, timeout_sec),
                            stage_timeout_sec=stage_timeout_sec,
                        )
                    )
            row["events"] = events
            results.append(row)
    return {"workload": "petta-memory promoted-belief proof/packet profile", "results": results}


def _build_export_payload(store_path: Path, count: int) -> dict[str, list[str]]:
    store = build_profile_store(store_path, count)
    return {
        "statements": [line for line in store.pettachainer_evidence_view().splitlines() if line],
        "packets": [line for line in store.pettachainer_evidence_packet_view().splitlines() if line],
    }


def inspect_pettachainer_add_api(repo_path: str | Path) -> dict[str, object]:
    """Inspect a PeTTaChainer checkout for add-path API options.

    This is a source-level, no-runtime probe.  It records whether the checked-out
    PeTTaChainer exposes a public precompiled-add/cache API or only routes public
    add calls through ``compileadd``/``compileadd-mine``.  Keeping this as a pure
    filesystem inspection lets project records justify the current non-live
    precompiled handoff gate without rerunning the noisy SWI/MeTTa runtime.
    """
    repo = Path(repo_path)
    py_path = repo / "pettachainer" / "pettachainer.py"
    metta_path = repo / "pettachainer" / "metta" / "petta_chainer.metta"
    missing = [str(path) for path in (py_path, metta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing PeTTaChainer source files: " + ", ".join(missing))

    py_source = py_path.read_text(encoding="utf-8")
    metta_source = metta_path.read_text(encoding="utf-8")
    tree = ast.parse(py_source)
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PeTTaChainer"
        ),
        None,
    )
    if class_node is None:
        raise ValueError("PeTTaChainer class not found")
    public_methods = [
        node.name
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    add_methods = [name for name in public_methods if "add" in name.lower()]
    add_method_sources = {
        node.name: ast.get_source_segment(py_source, node) or ""
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in add_methods
    }
    add_method_compile_calls = {
        name: sorted(set(re.findall(r"compileadd(?:-mine)?", source)))
        for name, source in add_method_sources.items()
    }
    precompiled_terms = sorted(
        set(re.findall(r"\b[\w-]*(?:precompile|precompiled|cache|handoff)[\w-]*\b", py_source + "\n" + metta_source, re.IGNORECASE))
    )
    compileadd_defs = sorted(set(re.findall(r"\(= \((compileadd(?:-mine)?)\b", metta_source)))
    compileadd_subforms = re.findall(r"\$[\w-]+ \(([^\s()]+)", metta_source)
    selected_subforms = [
        name
        for name in compileadd_subforms
        if name
        in {
            "materialize-stmt-lambdas",
            "collapse",
            "list_to_set",
            "map-flat",
            "index-source-implication",
            "maybe-process-on-add",
            "process-on-add-items",
        }
    ]
    exposes_precompiled_add_api = any(
        token.lower() in {"precompile", "precompiled", "precompiled-add", "precompiled-cache", "handoff"}
        for token in precompiled_terms
    )
    return {
        "source": "pettachainer source inspection",
        "repo_path": str(repo),
        "public_add_methods": add_methods,
        "add_method_compile_calls": add_method_compile_calls,
        "compileadd_definitions": compileadd_defs,
        "compileadd_subforms_seen": selected_subforms,
        "precompiled_add_terms_seen": precompiled_terms,
        "exposes_precompiled_add_api": exposes_precompiled_add_api,
        "recommended_boundary": (
            "no public precompiled-add API found; keep petta-memory's handoff cache non-live and continue "
            "upstream materialize-stmt-lambdas/mm2compile instrumentation"
            if not exposes_precompiled_add_api
            else "review discovered precompiled/cache terms manually before adopting any API"
        ),
        "gates": [
            "Source inspection only; does not invoke PeTTaChainer compileadd/query.",
            "Do not infer beliefs or enable OmegaClaw writes from this inspection.",
            "Adopt an upstream API only after a separate non-live gate verifies semantics and provenance.",
        ],
    }


def _extract_metta_definition(source: str, symbol: str) -> dict[str, object] | None:
    """Return a bounded source excerpt for a top-level MeTTa definition.

    This is intentionally a source inspector, not a MeTTa evaluator.  It is used
    to keep the current PeTTaChainer bottleneck work grounded in checked-out
    upstream text while avoiding another noisy ``compileadd`` runtime call.
    """
    lines = source.splitlines()
    pattern = re.compile(rf"^\(= \({re.escape(symbol)}(?:\s|\))")
    start_index = next((index for index, line in enumerate(lines) if pattern.search(line)), None)
    if start_index is None:
        return None
    depth = 0
    end_index = start_index
    for index in range(start_index, len(lines)):
        line = lines[index]
        depth += line.count("(") - line.count(")")
        end_index = index
        if index > start_index and depth <= 0:
            break
    snippet = "\n".join(lines[start_index : end_index + 1])
    return {
        "symbol": symbol,
        "line_start": start_index + 1,
        "line_end": end_index + 1,
        "line_count": end_index - start_index + 1,
        "calls": sorted(set(re.findall(r"\(([A-Za-z0-9_+*/<>=?!|.-]+)(?:\s|\))", snippet))),
        "recursive": bool(re.search(rf"\({re.escape(symbol)}(?:\s|\))", snippet.split("\n", 1)[-1] if "\n" in snippet else "")),
        "snippet": snippet,
    }


def _sexpr_walk(value: object) -> Iterable[object]:
    yield value
    if isinstance(value, tuple):
        for item in value:
            yield from _sexpr_walk(item)


def inspect_materialize_stmt_lambdas_for_statement(repo_path: str | Path, statement: str) -> dict[str, object]:
    """Source-level analysis of ``materialize-stmt-lambdas`` for one statement.

    Runtime probes show the first ``compileadd`` binding can time out even for
    the tiny petta-memory promoted-belief statement.  This helper keeps the next
    probe bounded and non-live: it reads the checked-out source definition and
    statically checks whether the statement contains any ``|->`` lambda forms
    that would trigger the definition's only ``eval`` branch.  For lambda-free
    statements, the source definition should act as a structural identity walk,
    so a runtime timeout points at evaluator/recursive traversal overhead rather
    than user lambda execution.
    """
    repo = Path(repo_path)
    root_metta_path = repo / "pettachainer" / "metta" / "petta_chainer.metta"
    if not root_metta_path.exists():
        raise FileNotFoundError(f"missing PeTTaChainer source file: {root_metta_path}")

    form = parse_one_list(statement)
    walked = list(_sexpr_walk(form))
    expr_count = sum(1 for item in walked if isinstance(item, tuple))
    atom_count = len(walked) - expr_count
    lambda_forms = [to_source(item) for item in walked if isinstance(item, tuple) and item and symbol_text(item[0]) == "|->"]
    source = root_metta_path.read_text(encoding="utf-8")
    definition = _extract_metta_definition(source, "materialize-stmt-lambdas")
    if definition is not None:
        definition["file"] = "pettachainer/metta/petta_chainer.metta"
    return {
        "source": "pettachainer materialize-stmt-lambdas source statement inspection",
        "repo_path": str(repo),
        "statement": statement,
        "statement_stats": {
            "expression_nodes": expr_count,
            "atom_nodes": atom_count,
            "total_nodes": len(walked),
            "lambda_form_count": len(lambda_forms),
            "lambda_forms": lambda_forms,
        },
        "materialize_expected_identity": not lambda_forms,
        "expected_materialized_statement": statement if not lambda_forms else None,
        "definition": definition,
        "interpretation": (
            "No |-> lambda forms occur in this statement, so source-level materialization should only walk and rebuild the same tree; "
            "the observed timeout is therefore more likely in the PeTTa/MeTTa evaluator recursion/materialization machinery than in user lambda execution."
            if not lambda_forms
            else "Statement contains |-> lambda forms, so materialize-stmt-lambdas may invoke eval and must be runtime-gated separately."
        ),
        "next_probe": {
            "kind": "non-live materialize identity runtime gate",
            "preconditions": [
                "Use an isolated subprocess and short stage timeout.",
                "Compare materialized output to the original statement for lambda-free promoted beliefs.",
                "Do not proceed to mm2compile/compileadd unless this identity gate completes cleanly.",
            ],
        },
        "gates": [
            "Source inspection only; no PeTTaChainer runtime, compileadd, query, GoalChainer, or OmegaClaw path is invoked.",
            "Do not treat the identity expectation as an inferred belief or runtime success.",
            "Keep live writes and full add/query behind separate non-live gates.",
        ],
    }


def run_materialize_identity_gate(
    statement: str,
    *,
    project_root: Path,
    stage_timeout_sec: float = 10.0,
) -> dict[str, object]:
    """Run a bounded non-live identity gate for ``materialize-stmt-lambdas``.

    Source inspection first verifies that the statement is lambda-free.  Only
    then does this function configure the local PeTTaChainer runtime and run the
    single materialization form in an isolated subprocess.  It does not call
    ``mm2compile``, ``compileadd``, query, GoalChainer, or OmegaClaw paths.
    """
    repo = project_root / "repos" / "PeTTaChainer"
    inspection = inspect_materialize_stmt_lambdas_for_statement(repo, statement)
    if not inspection["materialize_expected_identity"]:
        return {
            "source": "non-live materialize-stmt-lambdas identity gate",
            "status": "skipped",
            "reason": "statement contains |-> lambda forms; identity materialization is not expected",
            "inspection": inspection,
            "gates": [
                "No runtime execution attempted because source inspection did not predict identity materialization.",
                "Do not proceed to mm2compile/compileadd/query from a skipped identity gate.",
            ],
        }

    _configure_local_runtime(project_root)
    event = _run_materialize_identity_event(statement, stage_timeout_sec=stage_timeout_sec)
    status = "passed" if event.get("status") == "ok" and event.get("identity_output_present") else "blocked"
    return {
        "source": "non-live materialize-stmt-lambdas identity gate",
        "status": status,
        "inspection": inspection,
        "runtime_event": event,
        "interpretation": (
            "Runtime materialization returned the original lambda-free statement; mm2compile can be gated separately."
            if status == "passed"
            else "Identity materialization did not complete with matching output under the bound; keep mm2compile/compileadd/query gated."
        ),
        "gates": [
            "Single materialize-stmt-lambdas call only; no mm2compile, compileadd, query, GoalChainer, or OmegaClaw path is invoked.",
            "Temporary/non-live probe only; no petta-memory journal writes and no inferred-belief claims.",
            "Proceed to mm2compile instrumentation only after this identity gate passes with matching output.",
        ],
    }


def _run_materialize_identity_event(statement: str, *, stage_timeout_sec: float) -> dict[str, object]:
    return _run_isolated_stage(
        "materialize_stmt_lambdas_identity",
        _materialize_identity_stage,
        (statement,),
        stage_timeout_sec=stage_timeout_sec,
    )


def materialize_identity_proof_shape_rungs(statement: str) -> list[str]:
    """Return bounded materialization rungs for one ``(: proof type tv)`` atom.

    The generic ladder accepts caller-selected forms.  This helper gives the
    blocked PeTTaChainer proof shape a reproducible progression: materialize the
    independent type and truth-value subforms first, then synthetic top-level
    proof prefixes, sentinel full-arity proof atoms, then the exact full proof atom.
    ``materialize-stmt-lambdas`` should be purely structural for lambda-free
    expressions, so prefix and sentinel rungs help distinguish a subform problem
    from a top-level list-shape/evaluator problem without invoking ``mm2compile``
    or ``compileadd``.
    """
    form = parse_one_list(statement)
    if len(form) != 4 or symbol_text(form[0]) != ":":
        raise ValueError("statement must be a PeTTaChainer proof atom: (: proof type tv)")
    proof_id = to_source(form[1])
    statement_type = to_source(form[2])
    truth_value = to_source(form[3])
    sentinel_type = "ProofShapeSentinel"
    sentinel_truth_value = "(STV 1.0 1.0)"
    return [
        statement_type,
        truth_value,
        f"(: {proof_id})",
        f"(: {proof_id} {statement_type})",
        f"(: {proof_id} {sentinel_type} {sentinel_truth_value})",
        f"(: {proof_id} {statement_type} {sentinel_truth_value})",
        f"(: {proof_id} {sentinel_type} {truth_value})",
        statement,
    ]


def materialize_nested_type_proof_rungs(statement: str) -> list[str]:
    """Return proof materialization rungs that decompose the nested Type field.

    The proof-shape ladder narrowed the current timeout to a full ``(: proof
    type tv)`` atom whose ``type`` field is itself a nested statement such as
    ``(Requires MemoryTarget0 PLNReadyViews)``.  This ladder keeps the top-level
    proof shape and sentinel STV fixed while gradually rebuilding the nested type
    expression with sentinel arguments.  It helps distinguish a generic nested
    expression/arity problem from a specific predicate or argument token problem,
    without invoking ``mm2compile`` or ``compileadd``.
    """
    form = parse_one_list(statement)
    if len(form) != 4 or symbol_text(form[0]) != ":":
        raise ValueError("statement must be a PeTTaChainer proof atom: (: proof type tv)")
    statement_type = form[2]
    if not isinstance(statement_type, tuple) or len(statement_type) < 2:
        raise ValueError("proof Type must be a nested expression with at least one argument")
    proof_id = to_source(form[1])
    type_head = to_source(statement_type[0])
    args = [to_source(part) for part in statement_type[1:]]
    sentinel_truth_value = "(STV 1.0 1.0)"
    sentinel_args = [f"TypeArgSentinel{index}" for index in range(len(args))]

    rungs = [
        f"(: {proof_id} {type_head} {sentinel_truth_value})",
        f"(: {proof_id} ({type_head}) {sentinel_truth_value})",
    ]
    for width in range(1, len(args) + 1):
        partial_args = args[:width]
        rungs.append(f"(: {proof_id} ({' '.join([type_head, *partial_args])}) {sentinel_truth_value})")
    if args != sentinel_args:
        rungs.append(f"(: {proof_id} ({' '.join([type_head, *sentinel_args])}) {sentinel_truth_value})")
        for index in range(len(args)):
            mixed_args = list(sentinel_args)
            mixed_args[index] = args[index]
            rung = f"(: {proof_id} ({' '.join([type_head, *mixed_args])}) {sentinel_truth_value})"
            if rung not in rungs:
                rungs.append(rung)
    return rungs


def materialize_nested_type_arity_matrix_rungs(statement: str) -> list[str]:
    """Return nested-Type materialization rungs ordered to test arity first.

    The first nested-Type ladder stopped at the original two-argument Type before
    reaching the all-sentinel and mixed-argument controls.  This matrix keeps the
    same full proof shape and sentinel STV, but schedules all-sentinel arity
    rungs before any original argument tokens.  If ``(Requires S0 S1)`` blocks,
    the materializer problem is likely generic to two-argument nested Type
    expressions inside proof atoms; if it passes, the mixed/original rows localize
    the issue to a specific argument token or token combination.
    """
    form = parse_one_list(statement)
    if len(form) != 4 or symbol_text(form[0]) != ":":
        raise ValueError("statement must be a PeTTaChainer proof atom: (: proof type tv)")
    statement_type = form[2]
    if not isinstance(statement_type, tuple) or len(statement_type) < 2:
        raise ValueError("proof Type must be a nested expression with at least one argument")

    proof_id = to_source(form[1])
    type_head = to_source(statement_type[0])
    args = [to_source(part) for part in statement_type[1:]]
    sentinel_truth_value = "(STV 1.0 1.0)"
    sentinel_args = [f"TypeArgSentinel{index}" for index in range(len(args))]

    rungs = [
        f"(: {proof_id} ({type_head}) {sentinel_truth_value})",
    ]
    for width in range(1, len(args) + 1):
        rungs.append(f"(: {proof_id} ({' '.join([type_head, *sentinel_args[:width]])}) {sentinel_truth_value})")
    for index in range(len(args)):
        mixed_args = list(sentinel_args)
        mixed_args[index] = args[index]
        rung = f"(: {proof_id} ({' '.join([type_head, *mixed_args])}) {sentinel_truth_value})"
        if rung not in rungs:
            rungs.append(rung)
    original_rung = f"(: {proof_id} ({' '.join([type_head, *args])}) {sentinel_truth_value})"
    if original_rung not in rungs:
        rungs.append(original_rung)
    return rungs


def materialize_nested_type_context_matrix_rungs(statement: str) -> list[str]:
    """Return rungs that test the blocked nested Type in nearby contexts.

    The arity matrix showed that a full proof atom blocks as soon as its Type
    field is a two-argument nested expression, even with synthetic sentinel
    tokens.  This context matrix keeps that all-sentinel nested expression fixed
    and moves it through minimal surrounding list shapes before returning to the
    exact PeTTaChainer ``(: proof type tv)`` context.  The ordering distinguishes
    a generic nested-expression-in-four-field-list problem from something more
    specific to the PeTTaChainer ``:`` proof atom shape.
    """
    form = parse_one_list(statement)
    if len(form) != 4 or symbol_text(form[0]) != ":":
        raise ValueError("statement must be a PeTTaChainer proof atom: (: proof type tv)")
    statement_type = form[2]
    if not isinstance(statement_type, tuple) or len(statement_type) < 3:
        raise ValueError("proof Type must be a nested expression with at least two arguments")

    proof_id = to_source(form[1])
    type_head = to_source(statement_type[0])
    sentinel_args = [f"TypeArgSentinel{index}" for index in range(len(statement_type) - 1)]
    nested_type = f"({' '.join([type_head, *sentinel_args])})"
    sentinel_truth_value = "(STV 1.0 1.0)"
    return [
        nested_type,
        f"(: {proof_id} {nested_type})",
        f"(ProofEnvelope {proof_id} {nested_type})",
        f"(ProofEnvelope {proof_id} {nested_type} {sentinel_truth_value})",
        f"(: {proof_id} {nested_type} {sentinel_truth_value})",
    ]


def run_materialize_nested_type_arity_matrix_gate(
    statement: str,
    *,
    project_root: Path,
    stage_timeout_sec: float = 10.0,
) -> dict[str, object]:
    """Run the nested-Type materialization matrix without mm2compile/add/query."""
    rungs = materialize_nested_type_arity_matrix_rungs(statement)
    result = run_materialize_identity_ladder_gate(
        rungs,
        project_root=project_root,
        stage_timeout_sec=stage_timeout_sec,
    )
    result.update(
        {
            "source": "non-live materialize-stmt-lambdas nested-type arity matrix gate",
            "proof_statement": statement,
            "nested_type_arity_matrix_rungs": rungs,
            "interpretation": (
                "All sentinel/mixed/original nested-Type matrix rungs materialized as identity; mm2compile can be gated separately."
                if result.get("status") == "passed"
                else "A nested-Type arity/token matrix rung failed or timed out; keep mm2compile/compileadd/query gated and use the first blocked rung to distinguish generic arity from token-specific cost."
            ),
        }
    )
    result["gates"] = [
        "Nested-Type arity/token matrix only; each rung invokes materialize-stmt-lambdas in an isolated subprocess.",
        "No mm2compile, compileadd, query, GoalChainer, OmegaClaw path, journal write, or inferred-belief claim is invoked.",
        "Synthetic sentinel/mixed rungs are diagnostics for the materializer/evaluator and are not PLN premises.",
    ]
    return result


def run_materialize_nested_type_context_matrix_gate(
    statement: str,
    *,
    project_root: Path,
    stage_timeout_sec: float = 10.0,
) -> dict[str, object]:
    """Run the nested-Type context matrix without mm2compile/add/query."""
    rungs = materialize_nested_type_context_matrix_rungs(statement)
    result = run_materialize_identity_ladder_gate(
        rungs,
        project_root=project_root,
        stage_timeout_sec=stage_timeout_sec,
    )
    result.update(
        {
            "source": "non-live materialize-stmt-lambdas nested-type context matrix gate",
            "proof_statement": statement,
            "nested_type_context_matrix_rungs": rungs,
            "interpretation": (
                "All context-matrix rungs materialized as identity; return to mm2compile gating."
                if result.get("status") == "passed"
                else "A context-matrix rung failed or timed out; keep mm2compile/compileadd/query gated and use the first blocked context to distinguish generic full-list nesting from ':' proof-shape-specific cost."
            ),
        }
    )
    result["gates"] = [
        "Nested-Type context matrix only; each rung invokes materialize-stmt-lambdas in an isolated subprocess.",
        "No mm2compile, compileadd, query, GoalChainer, OmegaClaw path, journal write, or inferred-belief claim is invoked.",
        "Synthetic ProofEnvelope/context rungs are diagnostics for the materializer/evaluator and are not PLN premises.",
    ]
    return result


def run_materialize_nested_type_ladder_gate(
    statement: str,
    *,
    project_root: Path,
    stage_timeout_sec: float = 10.0,
) -> dict[str, object]:
    """Run a non-live materialization ladder over the nested proof Type field."""
    rungs = materialize_nested_type_proof_rungs(statement)
    result = run_materialize_identity_ladder_gate(
        rungs,
        project_root=project_root,
        stage_timeout_sec=stage_timeout_sec,
    )
    result.update(
        {
            "source": "non-live materialize-stmt-lambdas nested-type proof ladder gate",
            "proof_statement": statement,
            "nested_type_rungs": rungs,
            "interpretation": (
                "Nested Type materialization completed for all sentinel/partial proof rungs; return to mm2compile gating."
                if result.get("status") == "passed"
                else "A nested Type proof rung failed or timed out; keep mm2compile/compileadd/query gated and instrument this shape next."
            ),
        }
    )
    result["gates"] = [
        "Nested-Type ladder only; each rung invokes materialize-stmt-lambdas in an isolated subprocess.",
        "No mm2compile, compileadd, query, GoalChainer, OmegaClaw path, journal write, or inferred-belief claim is invoked.",
        "Synthetic sentinel rungs are diagnostics for the materializer/evaluator and are not PLN premises.",
    ]
    return result


def run_materialize_proof_shape_ladder_gate(
    statement: str,
    *,
    project_root: Path,
    stage_timeout_sec: float = 10.0,
) -> dict[str, object]:
    """Run the materialization ladder specialized for a proof atom shape.

    This is a non-live instrumentation gate for the current PeTTaChainer
    bottleneck.  It delegates every runtime call to
    ``run_materialize_identity_ladder_gate`` and adds only the deterministic rung
    construction/provenance around the result.
    """
    rungs = materialize_identity_proof_shape_rungs(statement)
    result = run_materialize_identity_ladder_gate(
        rungs,
        project_root=project_root,
        stage_timeout_sec=stage_timeout_sec,
    )
    result.update(
        {
            "source": "non-live materialize-stmt-lambdas proof-shape ladder gate",
            "proof_statement": statement,
            "proof_shape_rungs": rungs,
            "interpretation": (
                "Proof-shape materialization completed for subforms, prefixes, and the full statement; mm2compile can be gated separately."
                if result.get("status") == "passed"
                else "A proof-shape materialization rung failed or timed out; keep mm2compile/compileadd/query gated and instrument the first blocked shape."
            ),
        }
    )
    result["gates"] = [
        "Proof-shape ladder only; each rung invokes materialize-stmt-lambdas in an isolated subprocess.",
        "No mm2compile, compileadd, query, GoalChainer, OmegaClaw path, journal write, or inferred-belief claim is invoked.",
        "Synthetic prefix rungs are diagnostics for the materializer/evaluator and are not PLN premises.",
    ]
    return result


def run_materialize_identity_ladder_gate(
    statements: Iterable[str],
    *,
    project_root: Path,
    stage_timeout_sec: float = 10.0,
) -> dict[str, object]:
    """Run materialization identity probes from small forms up to a full proof.

    The previous single-statement gate showed that the tiny promoted-belief proof
    can time out before ``mm2compile``.  This ladder keeps the same non-live
    boundary but makes the bottleneck sharper by testing caller-supplied
    lambda-free subforms independently before the full proof statement.  Each
    rung is source-checked and then run as one isolated ``materialize-stmt-lambdas``
    call; no compiled add/query or integration path is invoked.
    """
    items = list(statements)
    if not items:
        raise ValueError("materialize ladder requires at least one statement")
    repo = project_root / "repos" / "PeTTaChainer"
    inspections = [inspect_materialize_stmt_lambdas_for_statement(repo, item) for item in items]
    skipped = [item for item, inspection in zip(items, inspections) if not inspection["materialize_expected_identity"]]
    if skipped:
        return {
            "source": "non-live materialize-stmt-lambdas identity ladder gate",
            "status": "skipped",
            "reason": "one or more statements contain |-> lambda forms; identity materialization is not expected",
            "skipped_statements": skipped,
            "inspections": inspections,
            "gates": [
                "No runtime execution attempted because source inspection did not predict identity materialization for every rung.",
                "Do not proceed to mm2compile/compileadd/query from a skipped ladder gate.",
            ],
        }

    _configure_local_runtime(project_root)
    events: list[dict[str, object]] = []
    for index, item in enumerate(items):
        event = _run_materialize_identity_event(item, stage_timeout_sec=stage_timeout_sec)
        event["rung_index"] = index
        event["rung_statement"] = item
        events.append(event)
        if event.get("status") != "ok" or not event.get("identity_output_present"):
            break
    all_passed = len(events) == len(items) and all(
        event.get("status") == "ok" and event.get("identity_output_present") for event in events
    )
    return {
        "source": "non-live materialize-stmt-lambdas identity ladder gate",
        "status": "passed" if all_passed else "blocked",
        "rung_count_requested": len(items),
        "rung_count_executed": len(events),
        "inspections": inspections,
        "runtime_events": events,
        "first_blocked_rung": None if all_passed else events[-1].get("rung_index") if events else None,
        "interpretation": (
            "All lambda-free materialization rungs returned identity output; mm2compile can be gated separately."
            if all_passed
            else "A lambda-free materialization rung failed or timed out; keep mm2compile/compileadd/query gated and instrument this rung next."
        ),
        "gates": [
            "Each rung invokes only materialize-stmt-lambdas in an isolated subprocess; no mm2compile, compileadd, query, GoalChainer, or OmegaClaw path is invoked.",
            "Temporary/non-live probe only; no petta-memory journal writes and no inferred-belief claims.",
            "Stop at the first blocked rung to bound noisy PeTTaChainer runtime work.",
        ],
    }


def inspect_compileadd_bottleneck_sources(repo_path: str | Path) -> dict[str, object]:
    """Inspect source definitions on the current ``compileadd`` bottleneck path.

    Runtime probes already show that ``materialize-stmt-lambdas`` and
    ``mm2compile`` time out for the tiny promoted-belief workload while later
    index/process hooks can complete.  This helper records the exact upstream
    definitions and import/file locations that should be instrumented next,
    without invoking SWI, PeTTaChainer, ``compileadd``, or query.
    """
    repo = Path(repo_path)
    root_metta_path = repo / "pettachainer" / "metta" / "petta_chainer.metta"
    compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
    mining_path = repo / "pettachainer" / "metta" / "chainer" / "mining.metta"
    missing = [str(path) for path in (root_metta_path, compile_path, mining_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing PeTTaChainer MeTTa source files: " + ", ".join(missing))

    sources = {
        "pettachainer/metta/petta_chainer.metta": root_metta_path.read_text(encoding="utf-8"),
        "pettachainer/metta/chainer/compile.metta": compile_path.read_text(encoding="utf-8"),
        "pettachainer/metta/chainer/mining.metta": mining_path.read_text(encoding="utf-8"),
    }
    symbol_files = {
        "compileadd": "pettachainer/metta/petta_chainer.metta",
        "compileadd-mine": "pettachainer/metta/petta_chainer.metta",
        "materialize-stmt-lambdas": "pettachainer/metta/petta_chainer.metta",
        "mm2compile": "pettachainer/metta/chainer/compile.metta",
        "compile": "pettachainer/metta/chainer/compile.metta",
        "compile_": "pettachainer/metta/chainer/compile.metta",
        "index-source-implication": "pettachainer/metta/chainer/compile.metta",
        "maybe-process-on-add": "pettachainer/metta/chainer/mining.metta",
    }
    definitions: dict[str, object] = {}
    for symbol, file_name in symbol_files.items():
        definition = _extract_metta_definition(sources[file_name], symbol)
        if definition is not None:
            definition["file"] = file_name
            definitions[symbol] = definition
    root_imports = re.findall(r"!\(import! &self ([^)]+)\)", sources["pettachainer/metta/petta_chainer.metta"])
    return {
        "source": "pettachainer compileadd bottleneck source inspection",
        "repo_path": str(repo),
        "root_imports": root_imports,
        "definitions": definitions,
        "runtime_provenance": (
            "Use with prior profile artifacts showing materialize-stmt-lambdas/mm2compile timeouts; "
            "this helper does not rerun PeTTaChainer."
        ),
        "next_instrumentation_targets": [
            {
                "symbol": "materialize-stmt-lambdas",
                "reason": "First compileadd binding; recursively traverses every term and evals |-> lambda forms before mm2compile.",
            },
            {
                "symbol": "mm2compile",
                "reason": "Second compileadd binding; clears ctx, invokes the broad compile dispatcher, then reads generated ctx atoms.",
            },
            {
                "symbol": "compile_",
                "reason": "Large upstream dispatcher beneath mm2compile; source-level line span provides the next finer instrumentation boundary.",
            },
        ],
        "gates": [
            "Source inspection only; no compileadd/query/runtime execution.",
            "Do not adopt or modify upstream PeTTaChainer semantics from this artifact alone.",
            "Keep petta-memory handoff cache non-live until a separate runtime gate passes.",
        ],
    }


def inspect_compile_dispatch_for_statement(repo_path: str | Path, statement: str) -> dict[str, object]:
    """Map one exported PeTTaChainer statement to a source-level ``compile_`` branch.

    This is deliberately not a runtime probe.  It answers a narrower question
    after the ``compileadd`` bottleneck source map: for the tiny promoted-belief
    STV statement that petta-memory exports, which branch of PeTTaChainer's
    ``compile_`` dispatcher should be reached *after* materialization and
    ``mm2compile``?  The result keeps follow-up instrumentation focused without
    invoking SWI, ``compileadd``, query, or GoalChainer.
    """
    repo = Path(repo_path)
    compile_path = repo / "pettachainer" / "metta" / "chainer" / "compile.metta"
    logic_config_path = repo / "pettachainer" / "metta" / "chainer" / "logic_config.metta"
    missing = [str(path) for path in (compile_path, logic_config_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing PeTTaChainer source files: " + ", ".join(missing))

    form = parse_one_list(statement)
    if len(form) != 4 or symbol_text(form[0]) != ":":
        raise ValueError("statement must be a PeTTaChainer proof atom: (: proof type tv)")
    proof_id = to_source(form[1])
    statement_type = form[2]
    truth_value = to_source(form[3])
    type_head = symbol_text(statement_type[0]) if isinstance(statement_type, tuple) and statement_type else symbol_text(statement_type)
    is_variable_type = bool(type_head and type_head.startswith("$"))

    logic_source = logic_config_path.read_text(encoding="utf-8")
    bidirectional_heads = sorted(
        token
        for token in set(re.findall(r"!\(set-bidirectional-implication-form\s+([^)\s]+)\)", logic_source))
        if not token.startswith("$")
    )
    if is_variable_type:
        selected_branch = "variable-type-empty"
        reason = "compile_ first rejects variable Type values by returning empty."
    elif type_head == "Implication":
        selected_branch = "implication-rule"
        reason = "Type head is Implication, so compile_ enters build-implication-plan and rule compilation."
    elif type_head in bidirectional_heads:
        selected_branch = "bidirectional-implication-rule"
        reason = "Type head is configured as bidirectional and compile_ expands it into forward/backward implications."
    else:
        selected_branch = "fact-assertion"
        reason = "Type is a concrete non-Implication expression, so compile_ should use compile-fact-kb plus compile-outputs."

    compile_source = compile_path.read_text(encoding="utf-8")
    compile_definition = _extract_metta_definition(compile_source, "compile_")
    if compile_definition is not None:
        compile_definition["file"] = "pettachainer/metta/chainer/compile.metta"
    return {
        "source": "pettachainer compile_ dispatch source inspection",
        "repo_path": str(repo),
        "statement": statement,
        "parsed_statement": {
            "proof_id": proof_id,
            "type": to_source(statement_type),
            "type_head": type_head,
            "truth_value": truth_value,
        },
        "configured_bidirectional_heads": bidirectional_heads,
        "selected_compile_branch": selected_branch,
        "reason": reason,
        "compile_definition": compile_definition,
        "next_instrumentation_targets": [
            {
                "symbol": "materialize-stmt-lambdas",
                "reason": "Still the first timed-out compileadd binding before any compile_ branch can be reached.",
            },
            {
                "symbol": "mm2compile",
                "reason": "Owns context clearing, broad compile dispatch, and generated ctx atom collection.",
            },
            {
                "symbol": "compile_ fact-assertion branch",
                "reason": "The petta-memory promoted-belief STV shape should avoid implication/rule branches; future probes can instrument this fact path specifically.",
            },
        ],
        "gates": [
            "Source inspection only; no PeTTaChainer runtime, compileadd, query, GoalChainer, or OmegaClaw path is invoked.",
            "Do not treat branch mapping as inferred belief evidence; it only narrows instrumentation targets.",
            "Keep handoff caches non-live until a separate add/query runtime gate passes.",
        ],
    }


def _static_import_safe_symbol(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", text).strip("_").lower()
    if not safe:
        return "atom"
    if re.match(r"^[0-9]", safe):
        return f"n_{safe}"
    return safe


def _static_import_statement_key(statement_type: object) -> str:
    if isinstance(statement_type, tuple) and statement_type:
        head = symbol_text(statement_type[0]) or "stmt"
        args = [symbol_text(part) if not isinstance(part, tuple) else to_source(part) for part in statement_type[1:]]
        return _static_import_safe_symbol("_".join([head, *[arg or "arg" for arg in args]]))
    return _static_import_safe_symbol(to_source(statement_type))


def design_static_import_microbenchmark_atoms(sample_atoms: Iterable[str]) -> dict[str, object]:
    """Design Prolog-safe scratch atoms for a future PeTTa ``static-import!`` gate.

    The prior source inspection showed current PeTTaChainer exports are unsafe for
    PeTTa's line-oriented converter.  This pure helper sketches a bounded
    alternative for a later temporary-directory benchmark: lower/underscore atom
    names, flat three-argument top-level records matching the loader's declared
    space-predicate arity, and enough mapping metadata to compare loaded facts
    back to the original STV/EvidencePacket exports.  It does not run SWI,
    qcompile, consult, or PeTTaChainer.
    """
    records: list[dict[str, object]] = []
    for atom in sample_atoms:
        form = parse_one_list(atom)
        normalized_atom: str | None = None
        kind = "unsupported"
        if len(form) == 4 and symbol_text(form[0]) == ":":
            proof_id = _static_import_safe_symbol(to_source(form[1]))
            statement_key = _static_import_statement_key(form[2])
            tv = form[3]
            strength = to_source(tv[1]) if isinstance(tv, tuple) and len(tv) >= 3 else "unknown"
            confidence = to_source(tv[2]) if isinstance(tv, tuple) and len(tv) >= 3 else "unknown"
            normalized_atom = (
                f"(pm_stv_statement {proof_id} "
                f"(pm_stv_payload {statement_key} {strength} {confidence}))"
            )
            kind = "stv_statement"
        elif len(form) == 5 and symbol_text(form[0]) == "EvidencePacket":
            statement_key = _static_import_statement_key(form[1])
            ec = form[2]
            support = to_source(ec[1]) if isinstance(ec, tuple) and len(ec) >= 3 else "0"
            opposition = to_source(ec[2]) if isinstance(ec, tuple) and len(ec) >= 3 else "0"
            provenance = _static_import_safe_symbol(to_source(form[4]))
            normalized_atom = (
                f"(pm_evidence_packet {statement_key} "
                f"(pm_ec_payload {support} {opposition} {provenance}))"
            )
            kind = "evidence_packet"
        records.append(
            {
                "kind": kind,
                "original_atom": atom,
                "normalized_atom": normalized_atom,
                "converted_prolog_fact": _petta_static_import_convert_line(normalized_atom) if normalized_atom else None,
                "safe_for_current_converter": bool(
                    normalized_atom
                    and not re.search(r"[A-Z-]", normalized_atom)
                    and len(parse_one_list(normalized_atom)) == 3
                ),
            }
        )
    return {
        "source": "petta static-import microbenchmark atom design",
        "records": records,
        "all_records_safe_for_current_converter": all(record["safe_for_current_converter"] for record in records),
        "benchmark_gate": [
            "Write normalized atoms only to a temporary scratch .metta file; do not append them to petta-memory journals.",
            "Run static-import! only in a bounded non-live runtime gate, then query the generated space predicate and compare to these expected facts.",
            "Treat this as a loader semantics benchmark, not as PeTTaChainer compileadd/query success or inferred belief evidence.",
        ],
    }


def _petta_static_import_convert_line(atom: str, space: str = "gckb") -> str:
    inner = atom[1:-1]
    inner = inner.replace("(", "[").replace(")", "]").replace(" ", ",")
    return f"'{space}'({inner})."


def _petta_static_import_fact_goal(fact: str) -> str:
    """Return a Prolog goal for checking one generated static-import fact.

    The loader benchmark already compares the generated ``scratch.pl`` text, but
    the runtime gate should also prove that each expected fact is actually
    consultable in SWI after ``static-import!``.  The generated facts are complete
    Prolog clauses like ``'pmbench'(...).``; as a query goal they are the same
    term without the trailing full stop.
    """
    stripped = fact.strip()
    if not stripped.endswith("."):
        raise ValueError("static-import fact must end with a full stop")
    return stripped[:-1]


def inspect_petta_static_import_source(petta_repo_path: str | Path, sample_atoms: Iterable[str] | None = None) -> dict[str, object]:
    """Inspect PeTTa's ``static-import!`` bulk-load source for petta-memory use.

    Ben pointed at ``trueagi-io/PeTTa`` ``lib/lib_import.pl`` as a possible fast
    static atom path.  This helper keeps that exploration source-level and
    non-live: it reads the checked-out Prolog loader, models its documented
    line-by-line transformation for a few sample atoms, and reports whether the
    current petta-memory/PeTTaChainer exports look safe to feed through it.
    It does not invoke SWI, create ``.pl``/``.qlf`` files, consult code, or
    change any memory/integration state.
    """
    repo = Path(petta_repo_path)
    import_path = repo / "lib" / "lib_import.pl"
    if not import_path.exists():
        raise FileNotFoundError(f"missing PeTTa static import source: {import_path}")

    source = import_path.read_text(encoding="utf-8")
    default_atoms = [
        "(: b-profile-000 (Requires MemoryTarget0 PLNReadyViews) (STV 0.70 0.55))",
        "(EvidencePacket (Requires MemoryTarget0 PLNReadyViews) (EC 3.0 1.0) ((domain omegaclaw-memory) (promotion-rule explicit-profile-workload)) pe-profile-000)",
    ]
    atoms = list(sample_atoms) if sample_atoms is not None else default_atoms

    def converted_line(atom: str, space: str = "gckb") -> str:
        # Mirrors lib_import.pl convert_line/3 for one already-line-oriented atom.
        return _petta_static_import_convert_line(atom, space=space)

    def token_warnings(atom: str) -> list[str]:
        tokens = re.findall(r'"(?:[^"\\]|\\.)*"|[^\s()]+', atom)
        warnings: list[str] = []
        for token in tokens:
            if token.startswith('"'):
                warnings.append(f"string literal {token!r} would be passed through without Prolog escaping semantics")
            elif re.match(r"^[A-Z_]", token):
                warnings.append(f"token {token!r} would be read by Prolog as a variable unless quoted")
            elif re.search(r"[^A-Za-z0-9_:.]", token):
                warnings.append(f"token {token!r} contains punctuation that is not a plain unquoted Prolog atom")
        return warnings

    source_features = {
        "defines_static_import": bool(re.search(r"'static-import!'\s*\(", source)),
        "uses_metta_file_to_prolog": "metta_file_to_prolog" in source,
        "uses_qcompile": "qcompile" in source,
        "consults_qlf": "consult(QlfFile)" in source,
        "line_oriented_converter": "read_line_to_string" in source and "convert_line" in source,
        "declares_space_predicate_arity_3": bool(re.search(r"multifile '~w'/3", source)),
    }
    conversion_limitations = [
        "Input is documented as S-expression data only: no code and no bangs.",
        "Conversion is line-oriented and strips the first/last character of each line, so multiline atoms/comments are not safe.",
        "The converter replaces parentheses with Prolog lists and spaces with commas without token quoting.",
        "Current petta-memory proof atoms contain uppercase symbols and hyphenated identifiers that are unsafe as unquoted Prolog terms.",
        "The generated predicate stores converted atoms under a space predicate; this is a bulk data loader, not a PeTTaChainer compileadd/indexing API.",
    ]
    samples = [
        {
            "input": atom,
            "converted_prolog_fact": converted_line(atom),
            "warnings": token_warnings(atom),
        }
        for atom in atoms
    ]
    sample_safe = all(not sample["warnings"] for sample in samples)
    recommendation = (
        "Do not use static-import! directly for current petta-memory PeTTaChainer exports. "
        "It is worth a later non-live benchmark only after either the export format or PeTTa converter quotes/escapes symbols safely "
        "and after query/index semantics are compared against compileadd-derived atoms."
    )
    return {
        "source": "PeTTa lib_import.pl static-import source inspection",
        "repo_path": str(repo),
        "source_file": str(import_path),
        "source_features": source_features,
        "sample_conversions": samples,
        "sample_atoms_safe_for_current_converter": sample_safe,
        "conversion_limitations": conversion_limitations,
        "recommendation": recommendation,
        "next_probe": {
            "kind": "non-live static-import microbenchmark",
            "preconditions": [
                "Use a temporary directory only and no OmegaClaw/live memory writes.",
                "Generate Prolog-safe quoted/lowercase normalized atoms or patch the converter in a scratch copy.",
                "Compare loaded predicate contents and read-only query behavior against expected normalized atoms before considering PeTTaChainer integration.",
            ],
        },
        "gates": [
            "Source inspection only; no SWI qcompile/consult/static-import execution.",
            "Do not treat static-import! as a supported PeTTaChainer precompiled-add API.",
            "Keep compileadd/query/live OmegaClaw paths gated until a separate runtime semantics test passes.",
        ],
    }



def _static_import_microbenchmark_stage(
    normalized_atoms: list[str],
    expected_facts: list[str],
    space: str = "gckb",
) -> dict[str, object]:
    """Run PeTTa ``static-import!`` on normalized atoms in a subprocess.

    This stage is designed to be called via ``_run_isolated_stage`` so that the
    SWI/PeTTa runtime noise and time are bounded.  It writes normalized atoms to
    a temporary ``.metta`` file, loads PeTTa's ``lib_import.pl`` to register
    the ``static-import!`` Prolog predicate, sets ``working_dir`` to the temp
    directory, calls ``static-import!`` directly via janus, then queries the
    loaded space predicate and compares results against expected facts.
    """
    import tempfile as _tempfile

    from petta import PeTTa

    # Initialize PeTTa so janus_swi is configured.
    PeTTa(verbose=False)

    import janus_swi as janus

    with _tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scratch_metta = td_path / "scratch.metta"
        scratch_metta.write_text("\n".join(normalized_atoms) + "\n", encoding="utf-8")

        # Load lib_import.pl to register the static-import! Prolog predicate.
        import_path = str(Path(__file__).resolve().parents[4] / "repos" / "PeTTa" / "lib" / "lib_import.pl")
        janus.query_once(f"consult('{import_path}')")

        # Set working_dir so static-import! finds scratch.metta in the temp dir.
        td_str = str(td).replace("\\", "/")
        janus.query_once(f"retractall(working_dir(_))")
        janus.query_once(f"assertz(working_dir('{td_str}'))")

        # Call static-import! directly via janus (not through process_metta_string,
        # which would need the MeTTa-level import_prolog_functions_from_file wrapper).
        files_before = set(p.name for p in td_path.iterdir())
        try:
            janus.query_once(f"'static-import!'({space}, scratch, true)")
            import_status = "called"
        except Exception as exc:
            import_status = f"error: {type(exc).__name__}: {exc}"
        files_after = set(p.name for p in td_path.iterdir())
        generated_files = sorted(files_after - files_before)

        # Query the loaded space predicate to retrieve all loaded facts.
        # janus_swi's query_once returns the first solution; for counting
        # and verification, use aggregate_all and check the first solution.
        loaded_facts_raw: list = []
        fact_count = 0
        try:
            count_result = janus.query_once(
                f"aggregate_all(count, {space}(_, _, _), Count)"
            )
            fact_count = count_result.get("Count", 0) if count_result else 0
        except Exception as exc:
            import_status += f"; count_error: {type(exc).__name__}: {exc}"

        first_solution: dict[str, object] = {}
        try:
            first_result = janus.query_once(f"{space}(A, B, C)")
            if first_result and first_result.get("truth"):
                first_solution = {
                    "A": str(first_result.get("A", "")),
                    "B": str(first_result.get("B", "")),
                    "C": str(first_result.get("C", "")),
                }
        except Exception as exc:
            import_status += f"; first_query_error: {type(exc).__name__}: {exc}"

        # Read the generated .pl file to compare against expected facts.
        pl_content = ""
        pl_path = td_path / "scratch.pl"
        if pl_path.exists():
            pl_content = pl_path.read_text(encoding="utf-8")
        # Extract fact lines (skip directives) for comparison.
        pl_fact_lines = sorted(
            line.strip() for line in pl_content.splitlines()
            if line.strip() and not line.strip().startswith(":-")
        )
        expected_sorted = sorted(expected_facts)
        facts_match = pl_fact_lines == expected_sorted

        # Verify the expected facts against the consulted runtime predicate too,
        # not just against the generated scratch.pl text.  This catches false
        # positives where conversion worked but qcompile/consult or the selected
        # named space did not actually make the facts queryable.
        runtime_fact_checks: list[dict[str, object]] = []
        for fact in expected_sorted:
            goal = _petta_static_import_fact_goal(fact)
            try:
                query_result = janus.query_once(goal)
                present = bool(query_result and query_result.get("truth"))
                runtime_fact_checks.append({"fact": fact, "goal": goal, "present": present})
            except Exception as exc:
                runtime_fact_checks.append(
                    {"fact": fact, "goal": goal, "present": False, "error": f"{type(exc).__name__}: {exc}"}
                )
        runtime_expected_facts_present = all(check["present"] for check in runtime_fact_checks)

        return {
            "result": "loaded" if fact_count > 0 else "empty",
            "space": space,
            "import_status": import_status,
            "generated_files": generated_files,
            "loaded_fact_count": fact_count,
            "expected_fact_count": len(expected_sorted),
            "facts_match": facts_match,
            "runtime_expected_facts_present": runtime_expected_facts_present,
            "runtime_fact_checks": runtime_fact_checks,
            "pl_fact_lines": pl_fact_lines,
            "expected_facts": expected_sorted,
            "first_solution": first_solution,
        }


def run_static_import_microbenchmark(
    sample_atoms: list[str],
    *,
    project_root: Path,
    stage_timeout_sec: float = 30.0,
    space: str = "gckb",
) -> dict[str, object]:
    """Run a non-live PeTTa ``static-import!`` microbenchmark over normalized atoms.

    This consumes the output of ``design_static_import_microbenchmark_atoms``
    and runs the actual ``static-import!`` loader in a bounded subprocess to
    verify that (1) the converter accepts the normalized atoms, (2) the loaded
    space predicate contains the expected facts, and (3) the conversion timing
    is bounded.  It does not append to petta-memory journals, does not invoke
    PeTTaChainer ``compileadd``/query, and does not touch OmegaClaw.
    """
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", space):
        raise ValueError("space must be a Prolog-safe predicate symbol")
    design = design_static_import_microbenchmark_atoms(sample_atoms)
    if not design["all_records_safe_for_current_converter"]:
        return {
            "source": "non-live static-import microbenchmark",
            "status": "skipped",
            "reason": "not all normalized atoms are safe for the current PeTTa static-import converter",
            "design": design,
            "gates": [
                "No runtime execution attempted; unsafe atoms were not loaded.",
                "Do not treat this as PeTTaChainer compileadd/query success or inferred belief evidence.",
            ],
        }

    normalized_atoms = [r["normalized_atom"] for r in design["records"] if r["normalized_atom"]]
    expected_facts = [_petta_static_import_convert_line(atom, space=space) for atom in normalized_atoms]

    _configure_local_runtime(project_root)
    event = _run_isolated_stage(
        "static_import_load_and_query",
        _static_import_microbenchmark_stage,
        (normalized_atoms, expected_facts, space),
        stage_timeout_sec=stage_timeout_sec,
    )
    return {
        "source": "non-live static-import microbenchmark",
        "design": design,
        "runtime_event": event,
        "gates": [
            "Temporary directory only; no petta-memory journal writes or OmegaClaw integration.",
            "This is a loader semantics benchmark, not PeTTaChainer compileadd/query success.",
            "Do not treat loaded facts as inferred beliefs or PLN premises.",
            "Keep compileadd/query/live OmegaClaw paths gated until a separate runtime semantics test passes.",
        ],
    }


def summarize_compileadd_strategy(profile: dict[str, object]) -> dict[str, object]:
    """Summarize the bounded PeTTaChainer add-path decision from a profile.

    The profile artifacts are intentionally noisy and stage-oriented.  This
    pure helper extracts the decision-relevant statuses so cron slices and
    project records can make the add-path choice reproducibly without rerunning
    PeTTaChainer.  It does not change export semantics or call the runtime.
    """
    results = profile.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("profile must contain at least one result row")
    row = results[0]
    if not isinstance(row, dict):
        raise ValueError("profile result row must be an object")
    events = row.get("events")
    if not isinstance(events, list):
        raise ValueError("profile result row must contain events")
    by_label = {event.get("label"): event for event in events if isinstance(event, dict) and event.get("label")}

    def status(label: str) -> str:
        event = by_label.get(label, {})
        value = event.get("status") if isinstance(event, dict) else None
        return str(value) if value is not None else "missing"

    materialize_direct = status("compileadd_probe_materialize_direct")
    materialize_eval = status("compileadd_probe_materialize_eval_control")
    mm2compile_direct = status("compileadd_probe_mm2compile_direct")
    mm2compile_eval = status("compileadd_probe_mm2compile_eval_control")
    proof_add = status("proof_runtime_add_only")
    check_stmt = status("check_stmt_all")
    init_only = status("pettachainer_init_only")
    fast_later_probes = [
        label
        for label in ("compileadd_probe_index_source_direct", "compileadd_probe_maybe_process_on_add_direct")
        if status(label) == "ok"
    ]

    direct_and_eval_timeout = all(
        value == "timeout"
        for value in (materialize_direct, materialize_eval, mm2compile_direct, mm2compile_eval)
    )
    add_path_blocked = proof_add == "timeout" or direct_and_eval_timeout
    if check_stmt == "ok" and init_only == "ok" and add_path_blocked:
        recommendation = "precompiled_statement_cache_gate"
        rationale = (
            "Validated STV/EvidencePacket exports and PeTTaChainer construction are healthy, "
            "but compileadd materialization/mm2compile and proof add remain timeout-bound. "
            "The next bounded petta-memory path should cache checked promoted statements/packets "
            "as a non-live PLN-ready handoff artifact while leaving full PeTTaChainer add/query "
            "behind an explicit gate and reserving upstream materialization instrumentation for follow-up."
        )
    elif check_stmt == "ok" and not add_path_blocked:
        recommendation = "continue_runtime_add_query_gate"
        rationale = "The checked profile does not show the compileadd/add timeout pattern; continue with runtime add/query smoke gates."
    else:
        recommendation = "runtime_setup_or_export_debug"
        rationale = "Basic statement validation or constructor setup is not healthy enough to choose an add optimization path."

    return {
        "recommended_next_add_path": recommendation,
        "rationale": rationale,
        "observed_statuses": {
            "check_stmt_all": check_stmt,
            "pettachainer_init_only": init_only,
            "compileadd_probe_materialize_direct": materialize_direct,
            "compileadd_probe_materialize_eval_control": materialize_eval,
            "compileadd_probe_mm2compile_direct": mm2compile_direct,
            "compileadd_probe_mm2compile_eval_control": mm2compile_eval,
            "proof_runtime_add_only": proof_add,
        },
        "fast_later_probes": fast_later_probes,
        "gates": [
            "Do not enable live OmegaClaw integration from this artifact.",
            "Do not treat cached statements as inferred beliefs; they are checked handoff inputs only.",
            "Revisit full compileadd/query after upstream materialize/mm2compile instrumentation or a precompiled add API exists.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile narrow PeTTaChainer workloads generated from petta-memory exports")
    parser.add_argument("--sizes", default="1,3", help="Comma-separated promoted-belief cluster counts")
    parser.add_argument("--steps", type=int, default=5, help="PeTTaChainer query step bound")
    parser.add_argument("--timeout-sec", type=float, default=5.0, help="Per-query timeout")
    parser.add_argument("--stage-timeout-sec", type=float, default=30.0, help="Hard subprocess timeout for each PeTTaChainer stage")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--include-runtime-add", action="store_true", help="Also time PeTTaChainer compileadd/query in isolated subprocesses; can be noisy/slow")
    parser.add_argument("--include-contextual", action="store_true", help="Also time EvidencePacket contextual projection; implies --include-runtime-add")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args(argv)
    sizes = [int(part) for part in args.sizes.split(",") if part.strip()]
    profile = profile_sizes(
        sizes,
        steps=args.steps,
        timeout_sec=args.timeout_sec,
        project_root=args.project_root,
        stage_timeout_sec=args.stage_timeout_sec,
        include_runtime_add=args.include_runtime_add or args.include_contextual,
        include_contextual=args.include_contextual,
    )
    text = json.dumps(profile, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
