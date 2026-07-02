from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable

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
    return {"label": label, "seconds": round(time.perf_counter() - started, 6), "result": result}


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
    include_runtime_add: bool = False,
    include_contextual: bool = False,
) -> dict[str, object]:
    _configure_local_runtime(project_root)
    from pettachainer import check_stmt

    if include_runtime_add:
        from pettachainer import PeTTaChainer

    results: list[dict[str, object]] = []
    for count in sizes:
        with tempfile.TemporaryDirectory() as td:
            store_path = Path(td) / "medium_memory.metta"
            row: dict[str, object] = {"clusters": count, "steps": steps, "timeout_sec": timeout_sec}
            events: list[dict[str, object]] = []
            build_event = _time_call("build_store_and_exports", lambda: _build_export_payload(store_path, count))
            payload = build_event.pop("result")
            events.append(build_event)
            statements = payload["statements"]
            packets = payload["packets"]
            row.update({"statement_count": len(statements), "packet_count": len(packets)})
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                events.append(_time_call("check_stmt_all", lambda: [check_stmt(stmt) for stmt in statements]))
                if include_runtime_add:
                    proof_handler = PeTTaChainer()
                    events.append(_time_call("add_proof_statements_no_check", lambda: proof_handler.add_atoms_no_check(statements)))
                    query = "(: $proof (Requires MemoryTarget0 PLNReadyViews) $tv)"
                    events.append(_time_call("query_first_target", lambda: proof_handler.query(query, steps=steps, timeout_sec=timeout_sec)))
                    if include_contextual:
                        packet_handler = PeTTaChainer()
                        events.append(_time_call("add_evidence_packets_no_check", lambda: packet_handler.add_atoms_no_check(packets)))
                        events.append(
                            _time_call(
                                "contextual_query_first_target",
                                lambda: packet_handler.contextual_query(query, steps=steps, timeout_sec=timeout_sec).answers,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile narrow PeTTaChainer workloads generated from petta-memory exports")
    parser.add_argument("--sizes", default="1,3", help="Comma-separated promoted-belief cluster counts")
    parser.add_argument("--steps", type=int, default=5, help="PeTTaChainer query step bound")
    parser.add_argument("--timeout-sec", type=float, default=5.0, help="Per-query timeout")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--include-runtime-add", action="store_true", help="Also time PeTTaChainer compileadd/query; can be noisy/slow")
    parser.add_argument("--include-contextual", action="store_true", help="Also time EvidencePacket contextual projection; implies --include-runtime-add")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args(argv)
    sizes = [int(part) for part in args.sizes.split(",") if part.strip()]
    profile = profile_sizes(
        sizes,
        steps=args.steps,
        timeout_sec=args.timeout_sec,
        project_root=args.project_root,
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
