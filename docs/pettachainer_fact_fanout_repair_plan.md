# PeTTaChainer fact-path fan-out repair plan

Status: first isolated source repair admitted; no upstream change or set collapse is approved.

The source-gated probes now close the 256-copy public `compile` result for one
concrete fact:

| Boundary | Factor | Closed count |
| --- | ---: | ---: |
| literal fact clause | 1 | 1 |
| `compile-fact-kb` | 8x | 8 |
| `bidirectional-implication-type?` | 4x | 32 |
| annotated `@` definition head | 2x | 64 |
| duplicate compiler registration | 2x | 128 |
| public `compile` wrapper/evaluator | 2x | 256 |

After compiler output is replaced by one canonical clause for diagnosis,
`mm2stmt`'s overlapping empty-premise arms add 2x and the surrounding
clear/convert/collect expression adds another 2x, closing its observed four
copies.

`build_pettachainer_fact_fanout_repair_plan()` validates this arithmetic and
fails closed when any measured count drifts. The repair order is:

1. Test removal of duplicate compiler registration in an isolated pinned
   PeTTaChainer checkout. **Completed 2026-07-17:** an exact one-line removal
   reduced direct `compile_` from 128 duplicate-equivalent outputs to one with
   the same normalized clause. The factors above are therefore diagnostic
   attributions under the duplicated runtime, not independent repair effects.
2. Make the zero-premise `mm2stmt` case exclusive while proving unchanged
   behavior for non-empty premise lists.
3. Investigate the bidirectional classifier, annotated head, `compile-fact-kb`,
   and public wrapper until each source-gated rung is single-output.
4. Consider byte-identical set collapse only as an experimental fallback, with
   separate fact/rule semantic parity tests. Do not use it to conceal matcher or
   import defects.

Before applying another repair, rerun the existing public wrapper, fact-KB,
predicate-ladder, annotation, `mm2stmt`, and collection gates against this
single-import candidate. Keep `compileadd`, query/result admission, manifest construction, promotion,
memory writes, and live OmegaClaw/GoalChainer integration closed until a pinned
isolated repair passes the existing rungs.
