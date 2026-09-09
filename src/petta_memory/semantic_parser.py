"""
Semantic Parser Contract - Item 6 from Ben review.

Defines an interface for parsing natural-language text into structured
semantic representations suitable for MeTTa/PLN inference, evidence
tracking, and persistent memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Any
import json
import re
import logging

log = logging.getLogger(__name__)



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
        }

    @classmethod
    def from_dict(cls, d: dict) -> ParsedSemantics:
        return cls(
            concepts=d.get("concepts", []),
            relations=[tuple(r) for r in d.get("relations", [])],
            evidence_type=d.get("evidence_type", "observation"),
            attribution=d.get("attribution", "unknown"),
            scope=d.get("scope", "general"),
            raw_text=d.get("raw_text", ""),
            metta_atoms=d.get("metta_atoms", []),
            confidence=d.get("confidence", 1.0),
        )



class SemanticParser(Protocol):
    """Interface contract for semantic parsing.

    Any object with a compatible parse method satisfies this protocol.
    Implementations may use LLMs, symbolic parsers, neural models, or
    hybrid approaches - the contract is the same.
    """

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
        lower_text = text.lower()
        for pattern, pred in self._RELATION_PATTERNS:
            for m in re.finditer(pattern, lower_text):
                subj = m.group(1)
                obj = m.group(2)
                if subj not in self._STOP and obj not in self._STOP:
                    relations.append((subj, pred, obj))

        return ParsedSemantics(
            concepts=unique_concepts,
            relations=relations,
            evidence_type=evidence_type,
            attribution=attribution,
            scope=scope,
            raw_text=text,
            confidence=0.9,
        )



_LLM_SYSTEM_PROMPT = (
    "You are a semantic parser for a cognitive memory system. "
    "Extract structured semantics from the given text. "
    "Return ONLY a JSON object with these keys: "
    '{"concepts": [list], "relations": [["s","p","o"]], '
    '"evidence_type": "observation", "scope": "", "confidence": 0.9}. '
    "Use lowercase for concepts, camelCase for predicates. Return ONLY JSON."
)

_LLM_USER_TEMPLATE = (
    "Attribution: {attribution}\n"
    "Scope: {scope}\n"
    "Evidence type: {evidence_type}\n"
    "Context: {context}\n\n"
    "Text to parse:\n{text}"
)


class LLMSemanticParser:
    """LLM-backed semantic parser behind the SemanticParser contract."""

    def __init__(self, llm_call, *, fallback=None):
        self.llm_call = llm_call
        self.fallback = fallback or MockSemanticParser()

    def parse(self, text, *, attribution="unknown", scope="general", evidence_type="observation", context=None):
        prompt = _LLM_USER_TEMPLATE.format(
            attribution=attribution, scope=scope,
            evidence_type=evidence_type, context=context or "", text=text)
        try:
            raw_response = self.llm_call(prompt, _LLM_SYSTEM_PROMPT)
            data = self._extract_json(raw_response)
            if data is None:
                log.warning("LLM returned unparseable JSON, using fallback")
                return self.fallback.parse(text, attribution=attribution, scope=scope, evidence_type=evidence_type, context=context)
            return ParsedSemantics(
                concepts=data.get("concepts", []),
                relations=[tuple(r) for r in data.get("relations", [])],
                evidence_type=data.get("evidence_type", evidence_type),
                attribution=attribution,
                scope=data.get("scope", scope),
                raw_text=text,
                confidence=data.get("confidence", 0.8))
        except Exception as e:
            log.warning(f"LLM parse failed: {e}, using fallback")
            return self.fallback.parse(text, attribution=attribution, scope=scope, evidence_type=evidence_type, context=context)

    def _extract_json(self, text: str) -> dict | None:
        """Try to find a JSON object in the LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    return None
            return None
