"""GOLEM-Iter Phase G3: Attention-ranked context injection.

Renders a bounded (<=500 char) attention-ranked belief index for injection
into Iter's system context.  Each entry shows:

  belief_id  claim_state  STI_rank  evidence_count  last_stimulus_class

The transformation is a pure function of bridge state — no side effects.
It does NOT recommend actions; it only surfaces ranked observations.

Activation is gated by environment variable ITER_PETTA_ECAN=1 (default OFF).
When unset, ``render_context`` returns an empty string.

Design ref: GOLEM-Iter §6 Phase G3.
"""

from __future__ import annotations

import os
from typing import Optional

from .ecan_bridge import ECANBridge
from .stimulus import StimulusInjector

_ENV_GATE = "ITER_PETTA_ECAN"
_MAX_CHARS = 500


def _is_enabled() -> bool:
    return os.environ.get(_ENV_GATE, "").strip() in ("1", "true", "True", "yes", "on")


def render_context(
    bridge: ECANBridge,
    injector: Optional[StimulusInjector] = None,
    limit: int = 20,
    max_chars: int = _MAX_CHARS,
) -> str:
    """Render a compact attention-ranked belief index for system context.

    Args:
        bridge: An ECANBridge that has been synced from the store.
        injector: Optional StimulusInjector for steadfastness metadata
            (last_event_class).  If None, that field is rendered as "-".
        limit: Maximum number of beliefs to include.
        max_chars: Hard character budget for the rendered string.

    Returns:
        A compact text block, or empty string when ITER_PETTA_ECAN is unset
        or an exception occurs.  The block is prefixed with a header line.
    """
    if not _is_enabled():
        return ""

    try:
        claims = bridge.get_claim_states()
        evidence_map = bridge.get_evidence_map()

        if injector is not None and injector.enabled:
            # Use steadfastness-aware prioritization (G4 gate)
            scored = injector.prioritized_with_steadfastness(limit=limit)
            entries = []
            for bid, sti, _last_inj, _total, event_cls in scored:
                ev_count = len(evidence_map.get(bid, []))
                state = claims.get(bid, "SelfReported")
                cls_name = event_cls if event_cls else "-"
                entries.append((bid, state, ev_count, cls_name))
        else:
            # Fall back to plain STI ranking
            scored = bridge.get_prioritized_beliefs(limit=limit)
            entries = []
            for bid, sti in scored:
                ev_count = len(evidence_map.get(bid, []))
                state = claims.get(bid, "SelfReported")
                entries.append((bid, state, ev_count, "-"))

        if not entries:
            return ""

        lines = ["## Attention Focus (ECAN)"]
        for bid, state, ev_count, cls_name in entries:
            line = f"  {bid} | {state} | ev={ev_count} | stim={cls_name}"
            # Respect char budget: check accumulated length
            if len("\n".join(lines) + "\n" + line) > max_chars:
                lines.append("  ...")
                break
            lines.append(line)

        return "\n".join(lines)
    except Exception:
        # Fail silently — never break the agent loop
        return ""


def context_block(bridge: ECANBridge, injector: Optional[StimulusInjector] = None) -> str:
    """Convenience wrapper: render with default limit and char budget."""
    return render_context(bridge, injector=injector)
