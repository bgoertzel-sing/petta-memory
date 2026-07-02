from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
