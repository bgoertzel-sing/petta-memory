"""
Semantic Parser Contract - Item 6 from Ben review.

Defines an interface for parsing natural-language text into structured
semantic representations suitable for MeTTa/PLN inference, evidence
tracking, and persistent memory.

Includes a PeTTaChainer-format parser that follows Man Hin's translation
instructions: input is formatted as CONTEXT/TODAY/DOMAIN/BACKGROUND/TEXT
blocks, output is parsed line-by-line into MeTTa atoms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol
import json
import os
import re
import logging

log = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "data", "pettachainer_prompt.txt"
)


def load_pettachainer_prompt() -> str:
    """Load the Man Hin PeTTaChainer translation prompt."""
    with open(_PROMPT_PATH, "r") as f:
        return f.read()


@dataclass
class ParsedSemantics:
    """Structured output from a semantic parse."""
    concepts: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    evidence_type: str = "observation"
    attribution: str = "unknown"
    scope: str = "general"
    raw_text: str = ""
    metta_atoms: list[str] = field(default_factory=list)
    confidence: float = 1.0
    # PeTTaChainer-specific fields
    context_atoms: list[str] = field(default_factory=list)
    today: str = ""
    domain: str = ""
    background: str = ""
    is_query: bool = False

    def to_metta_atoms(self) -> list[str]:
        atoms = list(self.metta_atoms)
        for c in self.concepts:
            atoms.append(f"(Concept {c})")
        for subj, pred, obj in self.relations:
            atoms.append(f"({pred} {subj} {obj})")
        return atoms

    def to_dict(self) -> dict:
        return {
            "concepts": self.concepts,
            "relations": [list(r) for r in self.relations],
            "evidence_type": self.evidence_type,
            "attribution": self.attribution,
            "scope": self.scope,
            "raw_text": self.raw_text,
            "metta_atoms": self.metta_atoms,
            "confidence": self.confidence,
            "context_atoms": self.context_atoms,
            "today": self.today,
            "domain": self.domain,
            "background": self.background,
            "is_query": self.is_query,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedSemantics":
        return cls(
            concepts=d.get("concepts", []),
            relations=[tuple(r) for r in d.get("relations", [])],
            evidence_type=d.get("evidence_type", "observation"),
            attribution=d.get("attribution", "unknown"),
            scope=d.get("scope", "general"),
            raw_text=d.get("raw_text", ""),
            metta_atoms=d.get("metta_atoms", []),
            confidence=d.get("confidence", 1.0),
            context_atoms=d.get("context_atoms", []),
            today=d.get("today", ""),
            domain=d.get("domain", ""),
            background=d.get("background", ""),
            is_query=d.get("is_query", False),
        )


class SemanticParser(Protocol):
    """Interface contract for semantic parsing."""

    def parse(
        self,
        text: str,
        *,
        attribution: str = "unknown",
        scope: str = "general",
        evidence_type: str = "observation",
        context: Optional[str] = None,
    ) -> ParsedSemantics:
        """Parse natural-language text into structured semantics."""
        ...


class MockSemanticParser:
    """Deterministic parser for tests. Uses simple keyword matching."""

    _RELATION_PATTERNS = [
        (r'(\w+)\s+is\s+a\s+(\w+)', "IsA"),
        (r'(\w+)\s+uses\s+(\w+)', "Uses"),
        (r'(\w+)\s+produces\s+(\w+)', "Produces"),
        (r'(\w+)\s+supports\s+(\w+)', "Supports"),
        (r'(\w+)\s+contradicts\s+(\w+)', "Contradicts"),
        (r'(\w+)\s+derives\s+(\w+)', "Derives"),
        (r'(\w+)\s+evidences\s+(\w+)', "EvidenceFor"),
    ]

    _STOP = frozenset(
        "the a an is are was were be been being have has had do does did "
        "will would shall should can could may might must of to in on at "
        "by for with from as that this these those it its and or not but".split()
    )

    def parse(
        self,
        text: str,
        *,
        attribution: str = "unknown",
        scope: str = "general",
        evidence_type: str = "observation",
        context: Optional[str] = None,
    ) -> ParsedSemantics:
        words = re.findall(r'[A-Za-z][A-Za-z0-9_-]+', text)
        concepts = [
            w.lower() for w in words
            if w.lower() not in self._STOP and len(w) >= 3
        ]
        seen = set()
        unique_concepts = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)

        relations = []
        for pattern, pred in self._RELATION_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                relations.append((m.group(1).lower(), pred, m.group(2).lower()))

        return ParsedSemantics(
            concepts=unique_concepts,
            relations=relations,
            evidence_type=evidence_type,
            attribution=attribution,
            scope=scope,
            raw_text=text,
            confidence=1.0,
        )


class LLMSemanticParser:
    """LLM-backed parser that expects JSON output.

    Falls back to MockSemanticParser on error or bad JSON.
    """

    def __init__(self, llm_call: Callable[[str, str], str]):
        self._llm_call = llm_call
        self._fallback = MockSemanticParser()

    _SYSTEM = "You are a semantic parser. Output ONLY valid JSON."

    def parse(
        self,
        text: str,
        *,
        attribution: str = "unknown",
        scope: str = "general",
        evidence_type: str = "observation",
        context: Optional[str] = None,
    ) -> ParsedSemantics:
        prompt = f"Parse this text into semantic JSON: {text}"
        if context:
            prompt += f"\nContext: {context}"

        try:
            raw = self._llm_call(prompt, self._SYSTEM)
            raw = raw.strip()
            bt = chr(96) * 3
            if raw.startswith(bt):
                raw = re.sub(r'^' + bt + r'\w*\n?', '', raw)
                raw = re.sub(r'\n?' + bt + r'$', '', raw)
                raw = raw.strip()

            data = json.loads(raw)
            return ParsedSemantics(
                concepts=data.get("concepts", []),
                relations=[tuple(r) for r in data.get("relations", [])],
                evidence_type=data.get("evidence_type", evidence_type),
                attribution=data.get("attribution", attribution),
                scope=data.get("scope", scope),
                raw_text=text,
                confidence=data.get("confidence", 1.0),
            )
        except Exception as e:
            log.warning("LLMSemanticParser failed (%s), falling back", e)
            return self._fallback.parse(
                text,
                attribution=attribution,
                scope=scope,
                evidence_type=evidence_type,
            )


# ---------------------------------------------------------------------------
# PeTTaChainer-format parser
# ---------------------------------------------------------------------------

# Regex for assertion lines: (: <name> <content> (STV <strength> <confidence>))
_ASSERT_RE = re.compile(
    r'^\(:\s+(\S+)\s+(.+?)\s+\(STV\s+([\d.]+)\s+([\d.]+)\)\)\s*$'
)


def parse_pettachainer_output(raw: str) -> tuple[list[str], bool, float]:
    """Parse PeTTaChainer line-by-line atom output.

    Returns (atoms, is_query, avg_confidence).
    Each assertion line looks like::

        (: e_adopt (Member sk_adopt_1 adopt) (STV 1.0 0.99))

    Query lines are bare MeTTa atoms without the (: ... (STV ...)) wrapper.
    """
    atoms: list[str] = []
    is_query = False
    confidences: list[float] = []

    for line in raw.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Strip comments
        comment_idx = stripped.find(";")
        if comment_idx >= 0:
            stripped = stripped[:comment_idx].strip()
        if not stripped:
            continue
        # Skip markdown fences
        bt3 = chr(96) * 3
        if stripped.startswith(bt3):
            continue
        # Check for assertion with STV
        m = _ASSERT_RE.match(stripped)
        if m:
            atoms.append(stripped)
            confidences.append(float(m.group(4)))
            continue
        # Check for assertion without STV: (: name content)
        m2 = re.match(r'^\(:\s+(\S+)\s+(.+)\)\s*$', stripped)
        if m2:
            atoms.append(stripped)
            confidences.append(1.0)
            continue
        # Bare S-expression (query or standalone atom)
        if stripped.startswith("(") and stripped.endswith(")"):
            atoms.append(stripped)
            is_query = True
            continue
        # Non-atom line -- could be a query marker or prose
        if stripped.upper().startswith("QUERY"):
            is_query = True
            continue
        log.debug("parse_pettachainer_output skipping non-atom line: %s", stripped)

    avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
    return atoms, is_query, avg_conf


class PeTTaChainerParser:
    """LLM-backed parser using Man Hin's PeTTaChainer translation prompt.

    Formats input as CONTEXT/TODAY/DOMAIN/BACKGROUND/TEXT blocks per the
    prompt specification, sends to the LLM, and parses the line-by-line
    MeTTa atom output into a ParsedSemantics.

    The LLM call signature is ``(user_msg: str, system: str) -> str``.
    """

    def __init__(self, llm_call: Callable[[str, str], str]):
        self._llm_call = llm_call
        self._prompt = load_pettachainer_prompt()

    # -- public API --

    def format_input(
        self,
        text: str,
        *,
        context_atoms: Optional[list[str]] = None,
        today: str = "",
        domain: str = "",
        background: str = "",
    ) -> str:
        """Format input according to the PeTTaChainer prompt spec.

        When context_atoms/today/domain/background are all empty the input
        is a bare TEXT block (as the prompt allows).
        """
        lines: list[str] = []

        has_context = bool(context_atoms or today or domain or background)
        if not has_context:
            return text

        lines.append("CONTEXT:")
        if context_atoms:
            for atom in context_atoms:
                lines.append(atom)
        if today:
            lines.append(f"TODAY: {today}")
        if domain:
            lines.append(f"DOMAIN: {domain}")
        if background:
            lines.append(f"BACKGROUND: {background}")
        lines.append("")
        lines.append("TEXT:")
        lines.append(text)

        return "\n".join(lines)

    def parse(
        self,
        text: str,
        *,
        attribution: str = "unknown",
        scope: str = "general",
        evidence_type: str = "observation",
        context: Optional[str] = None,
        today: str = "",
        domain: str = "",
        background: str = "",
    ) -> ParsedSemantics:
        """Parse text using PeTTaChainer format.

        The *context* parameter may be a raw PeTTaChainer context block
        (atoms + optional TODAY/DOMAIN/BACKGROUND lines). Alternatively,
        today/domain/background can be passed directly as keyword args.
        When none are provided, bare text is sent to the LLM.
        """
        context_atoms: list[str] = []

        if context:
            ctx_atoms, ctx_today, ctx_domain, ctx_bg = self._split_context(context)
            context_atoms = ctx_atoms
            if not today:
                today = ctx_today
            if not domain:
                domain = ctx_domain
            if not background:
                background = ctx_bg

        formatted = self.format_input(
            text,
            context_atoms=context_atoms or None,
            today=today,
            domain=domain,
            background=background,
        )

        try:
            raw_output = self._llm_call(formatted, self._prompt)
        except Exception as e:
            log.warning("PeTTaChainerParser LLM call failed (%s)", e)
            return ParsedSemantics(
                raw_text=text,
                attribution=attribution,
                scope=scope,
                evidence_type=evidence_type,
                confidence=0.0,
                context_atoms=context_atoms,
                today=today,
                domain=domain,
                background=background,
            )

        atoms, is_query, avg_conf = parse_pettachainer_output(raw_output)

        return ParsedSemantics(
            raw_text=text,
            attribution=attribution,
            scope=scope,
            evidence_type=evidence_type,
            metta_atoms=atoms,
            confidence=avg_conf,
            context_atoms=context_atoms,
            today=today,
            domain=domain,
            background=background,
            is_query=is_query,
        )

    def parse_with_context(
        self,
        text: str,
        *,
        context_atoms: Optional[list[str]] = None,
        today: str = "",
        domain: str = "",
        background: str = "",
        attribution: str = "unknown",
        scope: str = "general",
        evidence_type: str = "observation",
    ) -> ParsedSemantics:
        """Parse with structured context parameters.

        This is a convenience wrapper that accepts the context fields
        directly rather than as a pre-formatted string.
        """
        formatted = self.format_input(
            text,
            context_atoms=context_atoms,
            today=today,
            domain=domain,
            background=background,
        )

        try:
            raw_output = self._llm_call(formatted, self._prompt)
        except Exception as e:
            log.warning("PeTTaChainerParser LLM call failed (%s)", e)
            return ParsedSemantics(
                raw_text=text,
                attribution=attribution,
                scope=scope,
                evidence_type=evidence_type,
                confidence=0.0,
                context_atoms=context_atoms or [],
                today=today,
                domain=domain,
                background=background,
            )

        atoms, is_query, avg_conf = parse_pettachainer_output(raw_output)

        return ParsedSemantics(
            raw_text=text,
            attribution=attribution,
            scope=scope,
            evidence_type=evidence_type,
            metta_atoms=atoms,
            confidence=avg_conf,
            context_atoms=context_atoms or [],
            today=today,
            domain=domain,
            background=background,
            is_query=is_query,
        )

    # -- internals --

    @staticmethod
    def _split_context(raw: str) -> tuple[list[str], str, str, str]:
        """Split a raw context block into atoms, today, domain, background.

        Expects lines like::

            (: e_xxx (Member sk_xxx yyy) (STV 1.0 0.99))
            TODAY: Monday 2026-07-07
            DOMAIN: biology
            BACKGROUND: A lab setting
        """
        atoms: list[str] = []
        today = ""
        domain = ""
        background_lines: list[str] = []
        in_background = False

        for line in raw.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("CONTEXT:"):
                continue
            if stripped.upper().startswith("TODAY:"):
                today = stripped[len("TODAY:"):].strip()
                in_background = False
                continue
            if stripped.upper().startswith("DOMAIN:"):
                domain = stripped[len("DOMAIN:"):].strip()
                in_background = False
                continue
            if stripped.upper().startswith("BACKGROUND:"):
                background_lines.append(stripped[len("BACKGROUND:"):].strip())
                in_background = True
                continue
            if stripped.upper().startswith("TEXT:"):
                break
            if in_background:
                background_lines.append(stripped)
                continue
            atoms.append(stripped)

        background = " ".join(background_lines).strip()
        return atoms, today, domain, background
