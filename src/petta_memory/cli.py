from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .goalchainer_smoke import run_goalchainer_handoff_smoke, run_goalchainer_precompiled_handoff_smoke
from .patham9_pln import (
    context_selection_wrapper,
    patham9_pi_pln_extension_spec,
    patham9_pln_handoff_sentences,
    probabilistic_inference_filter,
    run_patham9_pln_derivation_smoke,
    run_patham9_pln_derivation_ec_projection_smoke,
    run_patham9_pln_ec_projection_smoke,
    run_patham9_pln_ec_projection_conflicting_smoke,
    run_patham9_pln_multi_sentence_derivation_smoke,
    run_patham9_pln_query_smoke,
    survey_trueagi_chaining_inference_control,
)
from .store import MediumMemoryStore, ValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PeTTa intermediate memory store CLI")
    parser.add_argument("--store", default="medium_memory.metta", help="Path to append-only .metta journal")
    sub = parser.add_subparsers(dest="cmd", required=True)

    append = sub.add_parser("append", help="Append a MemoryCluster from a file or stdin")
    append.add_argument("file", nargs="?", help="Cluster .metta file; stdin when omitted")

    query = sub.add_parser("query", help="Query clusters")
    query.add_argument("kind", choices=["id", "type", "about", "status", "role", "cluster"])
    query.add_argument("value")
    query.add_argument("--limit", type=int, default=20)

    prompt = sub.add_parser("prompt-view", help="Print bounded prompt view")
    prompt.add_argument("--limit-chars", type=int, default=4000)
    prompt.add_argument("--topic", action="append", default=[], help="Prefer clusters with this About target")
    prompt.add_argument("--status", action="append", default=[], help="Prefer clusters with this status")

    index = sub.add_parser("index-view", help="Print generated MM-index retrieval view")
    index.add_argument("--limit-chars", type=int, default=8000)

    pln = sub.add_parser("pln-view", help="Print PLN-safe atom view")
    pln.add_argument("--exclude", action="append", default=[], help="Additional predicate to exclude")
    pln.add_argument("--normalized", action="store_true", help="Include normalized PLN premise mapping atoms")
    pln.add_argument("--limit-chars", type=int, help="Bound output while preserving complete atom lines")

    pettachainer = sub.add_parser("pettachainer-view", help="Print promoted beliefs as PeTTaChainer proof statements")
    pettachainer.add_argument("--limit-chars", type=int, help="Bound output while preserving complete atom lines")

    packets = sub.add_parser(
        "pettachainer-packets-view",
        help="Print promoted beliefs with explicit evidence counts as PeTTaChainer EvidencePacket atoms",
    )
    packets.add_argument("--limit-chars", type=int, help="Bound output while preserving complete atom lines")

    handoff = sub.add_parser(
        "pettachainer-handoff-cache",
        help="Print non-live JSON cache of PLN-ready PeTTaChainer handoff inputs",
    )
    handoff.add_argument("--cache-id", default="petta-memory-pettachainer-handoff")

    goalchainer = sub.add_parser(
        "goalchainer-handoff-cache",
        help="Print non-live JSON cache mapping promoted evidence to GoalChainer appraisal inputs",
    )
    goalchainer.add_argument("--cache-id", default="petta-memory-goalchainer-handoff")

    patham9_pln = sub.add_parser(
        "patham9-pln-handoff",
        help="Print non-live patham9/PLN Sentence inputs from promoted handoff evidence",
    )
    patham9_pln.add_argument("--cache-id", default="petta-memory-patham9-pln-handoff")

    patham9_smoke = sub.add_parser(
        "patham9-pln-smoke",
        help="Run a bounded read-only patham9/PLN query smoke from promoted handoff evidence",
    )
    patham9_smoke.add_argument("--cache-id", default="petta-memory-patham9-pln-smoke")
    patham9_smoke.add_argument("--pln-repo", default="../patham9-pln", help="Path to local patham9/PLN checkout")
    patham9_smoke.add_argument("--env-script", help="Path to local PeTTa/SWI environment activation script")
    patham9_smoke.add_argument("--timeout-sec", type=float, default=30.0)

    patham9_derivation = sub.add_parser(
        "patham9-pln-derivation-smoke",
        help="Run a bounded read-only patham9/PLN two-premise derivation smoke from promoted handoff evidence",
    )
    patham9_derivation.add_argument("--cache-id", default="petta-memory-patham9-pln-derivation-smoke")
    patham9_derivation.add_argument("--pln-repo", default="../patham9-pln", help="Path to local patham9/PLN checkout")
    patham9_derivation.add_argument("--env-script", help="Path to local PeTTa/SWI environment activation script")
    patham9_derivation.add_argument("--timeout-sec", type=float, default=30.0)

    ec_projection = sub.add_parser(
        "patham9-pln-ec-projection-smoke",
        help="Run a bounded non-live EC projection comparison smoke (direct vs projected STV)",
    )
    ec_projection.add_argument("--cache-id", default="petta-memory-patham9-pln-ec-projection-smoke")
    ec_projection.add_argument("--pln-repo", default="../patham9-pln", help="Path to local patham9/PLN checkout")
    ec_projection.add_argument("--env-script", help="Path to local PeTTa/SWI environment activation script")
    ec_projection.add_argument("--timeout-sec", type=float, default=30.0)

    ec_conflicting = sub.add_parser(
        "patham9-pln-ec-conflicting-smoke",
        help="Run a bounded non-live conflicting-EC projection comparison smoke (strong STV + opposing EC)",
    )
    ec_conflicting.add_argument("--cache-id", default="petta-memory-patham9-pln-ec-conflicting-smoke")
    ec_conflicting.add_argument("--pln-repo", default="../patham9-pln", help="Path to local patham9/PLN checkout")
    ec_conflicting.add_argument("--env-script", help="Path to local PeTTa/SWI environment activation script")
    ec_conflicting.add_argument("--timeout-sec", type=float, default=30.0)
    ec_conflicting.add_argument("--support", type=float, default=1.0, help="Conflicting EC support count")
    ec_conflicting.add_argument("--opposition", type=float, default=9.0, help="Conflicting EC opposition count")

    derivation_ec = sub.add_parser(
        "patham9-pln-derivation-ec-smoke",
        help="Run a bounded non-live derivation EC projection comparison smoke (direct vs projected)",
    )
    derivation_ec.add_argument("--cache-id", default="petta-memory-patham9-pln-derivation-ec-smoke")
    derivation_ec.add_argument("--pln-repo", default="../patham9-pln", help="Path to local patham9/PLN checkout")
    derivation_ec.add_argument("--env-script", help="Path to local PeTTa/SWI environment activation script")
    derivation_ec.add_argument("--timeout-sec", type=float, default=30.0)

    multi_derivation = sub.add_parser(
        "patham9-pln-multi-derivation-smoke",
        help="Run a bounded non-live multi-Sentence derivation smoke from all promoted handoff items",
    )
    multi_derivation.add_argument("--cache-id", default="petta-memory-patham9-pln-multi-derivation-smoke")
    multi_derivation.add_argument("--pln-repo", default="../patham9-pln", help="Path to local patham9/PLN checkout")
    multi_derivation.add_argument("--env-script", help="Path to local PeTTa/SWI environment activation script")
    multi_derivation.add_argument("--timeout-sec", type=float, default=30.0)
    multi_derivation.add_argument("--bridge-term", help="Custom derived term for the bridge implication")

    pi_pln_spec = sub.add_parser(
        "patham9-pi-pln-spec",
        help="Print the concrete pi-PLN extension layer specification from handoff evidence",
    )
    pi_pln_spec.add_argument("--cache-id", default="petta-memory-patham9-pi-pln-spec")

    inf_ctl_survey = sub.add_parser(
        "trueagi-inf-ctl-survey",
        help="Survey inference-control patterns from the trueagi-io/chaining repo",
    )
    inf_ctl_survey.add_argument(
        "--chaining-repo",
        default="../trueagi-chaining",
        help="Path to local trueagi-io/chaining checkout",
    )

    inf_filter = sub.add_parser(
        "pi-pln-inference-filter",
        help="Apply probabilistic inference-control filter to handoff Sentences",
    )
    inf_filter.add_argument("--cache-id", default="petta-memory-pi-pln-inference-filter")
    inf_filter.add_argument("--min-confidence", type=float, default=0.0, help="Minimum projected confidence for inclusion")
    inf_filter.add_argument("--top-k", type=int, default=None, help="Keep only top-k items by composite score")

    context_select = sub.add_parser(
        "pi-pln-context-select",
        help="Apply context-selection wrapper to filter EvidencePackets before PLN invocation",
    )
    context_select.add_argument("--cache-id", default="petta-memory-pi-pln-context-select")
    context_select.add_argument("--domain", help="Select only EvidencePackets with this promotion_domain")
    context_select.add_argument("--cluster-id", help="Select only EvidencePackets from this cluster")
    context_select.add_argument("--promotion-rule", help="Select only EvidencePackets with this promotion_rule")
    context_select.add_argument("--min-relevance", type=float, default=0.0, help="Minimum packet relevance score [0, 1]")

    goal_smoke = sub.add_parser(
        "goalchainer-smoke",
        help="Run a bounded non-live GoalChainer decision smoke from promoted handoff evidence",
    )
    goal_smoke.add_argument("--cache-id", default="petta-memory-goalchainer-handoff-smoke")
    goal_smoke.add_argument("--goalchainer-repo", help="Path to local OmegaClaw-GoalChainer checkout")
    goal_smoke.add_argument("--request", help="Incident/request text for GoalChainer demo --json")
    goal_smoke.add_argument("--timeout-sec", type=float, default=20.0)
    goal_smoke.add_argument(
        "--external-cli",
        action="store_true",
        help="Use GoalChainer demo --json subprocess instead of the precompiled-cache bypass",
    )

    audit = sub.add_parser("audit-view", help="Print bounded complete MemoryCluster records for audit")
    audit.add_argument("--limit-chars", type=int, default=20000)

    tail = sub.add_parser("tail", help="Print journal tail")
    tail.add_argument("--chars", type=int, default=4000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = MediumMemoryStore(Path(args.store))
    try:
        if args.cmd == "append":
            text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
            cluster = store.append_cluster(text)
            print(cluster.cluster_id)
            return 0
        if args.cmd == "query":
            if args.kind == "id":
                clusters = store.query_id(args.value, limit=args.limit)
            elif args.kind == "type":
                clusters = store.query_type(args.value, limit=args.limit)
            elif args.kind == "about":
                clusters = store.query_about(args.value, limit=args.limit)
            elif args.kind == "status":
                clusters = store.query_status(args.value, limit=args.limit)
            elif args.kind == "role":
                clusters = store.query_role(args.value, limit=args.limit)
            else:
                cluster = store.query_cluster(args.value)
                clusters = [cluster] if cluster else []
            print("\n".join(c.text for c in clusters if c))
            return 0
        if args.cmd == "prompt-view":
            topics = set(args.topic) if args.topic else None
            statuses = set(args.status) if args.status else None
            print(store.prompt_view(limit_chars=args.limit_chars, topics=topics, statuses=statuses), end="")
            return 0
        if args.cmd == "index-view":
            print(store.index_view(limit_chars=args.limit_chars), end="")
            return 0
        if args.cmd == "pln-view":
            excluded = set(args.exclude) if args.exclude else None
            print(
                store.pln_view(
                    excluded_predicates=excluded,
                    normalized=args.normalized,
                    limit_chars=args.limit_chars,
                ),
                end="",
            )
            return 0
        if args.cmd == "pettachainer-view":
            print(store.pettachainer_evidence_view(limit_chars=args.limit_chars), end="")
            return 0
        if args.cmd == "pettachainer-packets-view":
            print(store.pettachainer_evidence_packet_view(limit_chars=args.limit_chars), end="")
            return 0
        if args.cmd == "pettachainer-handoff-cache":
            print(json.dumps(store.pettachainer_handoff_cache(cache_id=args.cache_id), indent=2, sort_keys=True))
            return 0
        if args.cmd == "goalchainer-handoff-cache":
            print(json.dumps(store.goalchainer_handoff_cache(cache_id=args.cache_id), indent=2, sort_keys=True))
            return 0
        if args.cmd == "patham9-pln-handoff":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            print(json.dumps(patham9_pln_handoff_sentences(cache), indent=2, sort_keys=True))
            return 0
        if args.cmd == "patham9-pln-smoke":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            kwargs = {"pln_repo": args.pln_repo, "timeout_sec": args.timeout_sec}
            if args.env_script:
                kwargs["env_script"] = args.env_script
            print(json.dumps(run_patham9_pln_query_smoke(handoff, **kwargs), indent=2, sort_keys=True))
            return 0
        if args.cmd == "patham9-pln-derivation-smoke":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            kwargs = {"pln_repo": args.pln_repo, "timeout_sec": args.timeout_sec}
            if args.env_script:
                kwargs["env_script"] = args.env_script
            print(json.dumps(run_patham9_pln_derivation_smoke(handoff, **kwargs), indent=2, sort_keys=True))
            return 0
        if args.cmd == "patham9-pln-ec-projection-smoke":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            kwargs = {"pln_repo": args.pln_repo, "timeout_sec": args.timeout_sec}
            if args.env_script:
                kwargs["env_script"] = args.env_script
            print(json.dumps(run_patham9_pln_ec_projection_smoke(handoff, **kwargs), indent=2, sort_keys=True))
            return 0
        if args.cmd == "patham9-pln-ec-conflicting-smoke":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            kwargs = {"pln_repo": args.pln_repo, "timeout_sec": args.timeout_sec}
            if args.env_script:
                kwargs["env_script"] = args.env_script
            kwargs["conflicting_support"] = args.support
            kwargs["conflicting_opposition"] = args.opposition
            print(json.dumps(run_patham9_pln_ec_projection_conflicting_smoke(handoff, **kwargs), indent=2, sort_keys=True))
            return 0
        if args.cmd == "patham9-pln-derivation-ec-smoke":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            kwargs = {"pln_repo": args.pln_repo, "timeout_sec": args.timeout_sec}
            if args.env_script:
                kwargs["env_script"] = args.env_script
            print(json.dumps(run_patham9_pln_derivation_ec_projection_smoke(handoff, **kwargs), indent=2, sort_keys=True))
            return 0
        if args.cmd == "patham9-pln-multi-derivation-smoke":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            kwargs = {"pln_repo": args.pln_repo, "timeout_sec": args.timeout_sec}
            if args.env_script:
                kwargs["env_script"] = args.env_script
            if args.bridge_term:
                kwargs["bridge_term"] = args.bridge_term
            print(json.dumps(run_patham9_pln_multi_sentence_derivation_smoke(handoff, **kwargs), indent=2, sort_keys=True))
            return 0
        if args.cmd == "patham9-pi-pln-spec":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            print(json.dumps(patham9_pi_pln_extension_spec(handoff), indent=2, sort_keys=True))
            return 0
        if args.cmd == "trueagi-inf-ctl-survey":
            print(json.dumps(survey_trueagi_chaining_inference_control(args.chaining_repo), indent=2, sort_keys=True))
            return 0
        if args.cmd == "pi-pln-inference-filter":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            result = probabilistic_inference_filter(
                handoff,
                min_confidence=args.min_confidence,
                top_k=args.top_k,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.cmd == "pi-pln-context-select":
            cache = store.pettachainer_handoff_cache(cache_id=args.cache_id)
            handoff = patham9_pln_handoff_sentences(cache)
            result = context_selection_wrapper(
                handoff,
                domain=args.domain,
                cluster_id=args.cluster_id,
                promotion_rule=args.promotion_rule,
                min_packet_relevance=args.min_relevance,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.cmd == "goalchainer-smoke":
            cache = store.goalchainer_handoff_cache(cache_id=args.cache_id)
            kwargs = {}
            if args.goalchainer_repo:
                kwargs["goalchainer_repo"] = args.goalchainer_repo
            if args.request:
                kwargs["request"] = args.request
            if args.external_cli:
                kwargs["timeout_sec"] = args.timeout_sec
                result = run_goalchainer_handoff_smoke(cache, **kwargs)
            else:
                result = run_goalchainer_precompiled_handoff_smoke(cache, **kwargs)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.cmd == "audit-view":
            print(store.audit_view(limit_chars=args.limit_chars), end="")
            return 0
        if args.cmd == "tail":
            print(store.tail(args.chars), end="")
            return 0
    except (OSError, ValidationError) as exc:
        print(f"petta-memory: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
