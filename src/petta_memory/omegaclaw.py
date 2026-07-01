from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Optional

from .sexpr import StringAtom, to_source
from .store import MediumMemoryStore, ValidationError


_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")


class LiveWriteDisabled(RuntimeError):
    """Raised when OmegaClaw integration attempts a live/autonomous write."""


@dataclass(frozen=True)
class OmegaClawMemoryPolicy:
    """Feature flags for the future OmegaClaw memory boundary.

    Defaults are deliberately inert. Prompt-view reads must be explicitly enabled
    by integration code, and live/autonomous writes are rejected in this prototype.
    """

    prompt_view_reads_enabled: bool = False
    autonomous_writes_enabled: bool = False
    prompt_view_limit_chars: int = 4000
    view_id: str = "oc-prompt-memory-view"
    prompt_topics: frozenset[str] = frozenset()
    prompt_statuses: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.autonomous_writes_enabled:
            raise LiveWriteDisabled("autonomous OmegaClaw writes are not enabled in petta-memory v0")
        if self.prompt_view_limit_chars < 0:
            raise ValidationError("prompt_view_limit_chars must be non-negative")
        if not _ID_RE.match(self.view_id):
            raise ValidationError(f"view_id must be a valid symbol id: {self.view_id}")


class OmegaClawMemoryBridge:
    """Read-only wrapper sketch for future OmegaClaw prompt context plumbing.

    This class is intentionally not wired into OmegaClaw. It demonstrates the
    planned read/write boundary over the local MediumMemoryStore:

    - prompt-view reads may be enabled by policy and return bounded MeTTa atoms;
    - autonomous writes always fail here until a later, reviewed integration;
    - no external systems, remotes, schedulers, or live agent state are touched.
    """

    def __init__(self, store: MediumMemoryStore, policy: Optional[OmegaClawMemoryPolicy] = None) -> None:
        self.store = store
        self.policy = policy or OmegaClawMemoryPolicy()

    def prompt_view_metta(self, *, generated_at: Optional[str] = None) -> str:
        """Return a bounded read-only MeTTa wrapper for OmegaClaw prompt assembly.

        When prompt-view reads are disabled, return an empty string so an
        integration caller can safely concatenate the result without changing
        prompt behavior.
        """

        if not self.policy.prompt_view_reads_enabled:
            return ""
        timestamp = generated_at or datetime.now(timezone.utc).isoformat()
        body = self.store.prompt_view(
            limit_chars=self.policy.prompt_view_limit_chars,
            topics=set(self.policy.prompt_topics),
            statuses=set(self.policy.prompt_statuses),
        ).strip()
        header = [
            f";;; BEGIN OmegaClawPromptView {self.policy.view_id}",
            f"(OmegaClawPromptView {self.policy.view_id})",
            f"(PromptViewSource {self.policy.view_id} petta-memory)",
            f"(PromptViewMode {self.policy.view_id} read-only)",
            f"(PromptViewGeneratedAt {self.policy.view_id} {_quoted_string(timestamp)})",
        ]
        if body:
            header.append(body)
        header.append(f";;; END OmegaClawPromptView {self.policy.view_id}")
        return "\n".join(header) + "\n"

    def append_from_omegaclaw(self, cluster_text: str) -> None:
        """Reject live OmegaClaw writes until the integration is explicitly reviewed."""

        raise LiveWriteDisabled(
            "OmegaClaw autonomous writes are outside v0; use MediumMemoryStore.append_cluster "
            "only in local tests or a reviewed manual migration path"
        )


def _quoted_string(value: str) -> str:
    return to_source(StringAtom(value))
