import pytest
from petta_memory.semantic_parser import (
    ParsedSemantics, MockSemanticParser, LLMSemanticParser, SemanticParser,
)


def test_mock_parser_concepts():
    p = MockSemanticParser()
    result = p.parse("The agent uses memory", attribution="test", scope="memory")
    assert "agent" in result.concepts
    assert "memory" in result.concepts
    assert result.attribution == "test"
    assert result.scope == "memory"


def test_mock_parser_relations():
    p = MockSemanticParser()
    result = p.parse("agent uses memory")
    assert ("agent", "Uses", "memory") in result.relations


def test_parsed_semantics_roundtrip():
    ps = ParsedSemantics(
        concepts=["cat", "dog"],
        relations=[("cat", "chases", "dog")],
        evidence_type="observation",
        attribution="test",
        scope="general",
        raw_text="cat chases dog",
        confidence=0.95,
    )
    d = ps.to_dict()
    ps2 = ParsedSemantics.from_dict(d)
    assert ps2.concepts == ["cat", "dog"]
    assert ps2.relations == [("cat", "chases", "dog")]
    assert ps2.confidence == 0.95


def test_to_metta_atoms():
    ps = ParsedSemantics(concepts=["water"], relations=[("ice", "melts", "water")])
    atoms = ps.to_metta_atoms()
    assert "(Concept water)" in atoms
    assert "(melts ice water)" in atoms


def test_llm_parser_success():
    def fake_llm(prompt, system):
        return '{"concepts": ["gravity", "mass"], "relations": [["gravity", "attracts", "mass"]], "evidence_type": "inference", "scope": "physics", "confidence": 0.85}'
    parser = LLMSemanticParser(fake_llm)
    result = parser.parse("gravity attracts mass")
    assert result.concepts == ["gravity", "mass"]
    assert result.relations == [("gravity", "attracts", "mass")]
    assert result.scope == "physics"
    assert result.confidence == 0.85


def test_llm_parser_fallback_on_bad_json():
    def bad_llm(prompt, system):
        return "this is not json"
    parser = LLMSemanticParser(bad_llm)
    result = parser.parse("agent uses memory")
    assert "agent" in result.concepts
    assert ("agent", "Uses", "memory") in result.relations


def test_llm_parser_fallback_on_exception():
    def crashing_llm(prompt, system):
        raise RuntimeError("LLM unavailable")
    parser = LLMSemanticParser(crashing_llm)
    result = parser.parse("agent uses memory")
    assert "agent" in result.concepts


def test_llm_parser_strips_markdown_fences():
    bt = chr(96) * 3
    def fenced_llm(prompt, system):
        return bt + "json\n" + '{"concepts": ["x"], "relations": [], "confidence": 0.5}' + "\n" + bt
    parser = LLMSemanticParser(fenced_llm)
    result = parser.parse("test")
    assert result.concepts == ["x"]


def test_protocol_satisfaction():
    p = MockSemanticParser()
    assert hasattr(p, "parse")
    llm_p = LLMSemanticParser(lambda p, s: "{}")
    assert hasattr(llm_p, "parse")
