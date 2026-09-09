"""Tests for PeTTaChainerParser."""

import pytest
from petta_memory.semantic_parser import (
    ParsedSemantics,
    PeTTaChainerParser,
    load_pettachainer_prompt,
)


class FakeLLM:
    """Fake LLM that returns canned MeTTa atom output."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None
        self.last_system = None
        self.call_count = 0

    def __call__(self, prompt: str, system: str) -> str:
        self.last_prompt = prompt
        self.last_system = system
        self.call_count += 1
        return self.response


class TestPeTTaChainerParser:

    def test_prompt_loads(self):
        prompt = load_pettachainer_prompt()
        assert len(prompt) > 100
        assert "PeTTaChainer" in prompt or "MeTTa" in prompt

    def test_system_prompt_is_pettachainer(self):
        fake = FakeLLM("(: e1 (Member sk_1 dog) (STV 1.0 0.99))")
        parser = PeTTaChainerParser(fake)
        parser.parse("A dog barks.")
        assert "PeTTaChainer" in fake.last_system or "atom" in fake.last_system.lower()

    def test_assertion_atoms_parsed(self):
        response = (
            "(: e_dog (Member sk_dog_1 dog) (STV 1.0 0.99))\n"
            "(: e_bark (Agent sk_bark_1 sk_dog_1) (STV 1.0 0.99))"
        )
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse("A dog barks.")
        assert len(result.metta_atoms) == 2
        assert result.metta_atoms[0] == "(: e_dog (Member sk_dog_1 dog) (STV 1.0 0.99))"
        assert result.metta_atoms[1] == "(: e_bark (Agent sk_bark_1 sk_dog_1) (STV 1.0 0.99))"
        assert result.is_query is False
        assert result.confidence == pytest.approx(0.99)

    def test_query_atoms_parsed(self):
        response = "(Agent sk_bark_1 who)"
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse("Who barked?")
        assert len(result.metta_atoms) == 1
        assert result.is_query is True

    def test_mixed_assertion_and_query(self):
        response = (
            "(: e1 (Member sk_1 cat) (STV 1.0 0.99))\n"
            "(Agent sk_2 who)"
        )
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse("A cat sat. Who sat?")
        assert len(result.metta_atoms) == 2
        assert result.is_query is True

    def test_context_formatting(self):
        response = "(: e1 (Member sk_1 dog) (STV 1.0 0.99))"
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse(
            "It barks.",
            context="(: e_dog (Member sk_dog_1 dog) (STV 1.0 0.99))",
            today="Tuesday 2026-07-07",
            domain="pets",
            background="A household with animals",
        )
        # Check the prompt contains labeled blocks
        assert "CONTEXT:" in fake.last_prompt
        assert "TODAY: Tuesday 2026-07-07" in fake.last_prompt
        assert "DOMAIN: pets" in fake.last_prompt
        assert "BACKGROUND: A household with animals" in fake.last_prompt
        assert "TEXT:" in fake.last_prompt
        assert "It barks." in fake.last_prompt
        # Check result carries context info
        assert len(result.context_atoms) == 1
        assert result.today == "Tuesday 2026-07-07"
        assert result.domain == "pets"
        assert result.background == "A household with animals"

    def test_bare_text_no_labels(self):
        response = "(: e1 (Member sk_1 cat) (STV 1.0 0.99))"
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        parser.parse("A cat.")
        # No metadata -> bare text, no CONTEXT/TEXT labels
        assert "CONTEXT:" not in fake.last_prompt
        assert "TEXT:" not in fake.last_prompt
        assert fake.last_prompt == "A cat."

    def test_markdown_fences_stripped(self):
        response = (
            "```metta\n"
            "(: e1 (Member sk_1 dog) (STV 1.0 0.99))\n"
            "```"
        )
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse("A dog.")
        assert len(result.metta_atoms) == 1
        assert "```" not in result.metta_atoms[0]

    def test_comments_stripped(self):
        response = (
            "(: e1 (Member sk_1 dog) (STV 1.0 0.99)) ; this is a comment\n"
            "; standalone comment line\n"
            "(: e2 (Agent sk_bark_1 sk_dog_1) (STV 1.0 0.99))"
        )
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse("A dog barks.")
        assert len(result.metta_atoms) == 2
        assert "comment" not in result.metta_atoms[0]

    def test_llm_failure_returns_empty(self):
        class FailLLM:
            def __call__(self, prompt, system):
                raise RuntimeError("API down")
        parser = PeTTaChainerParser(FailLLM())
        result = parser.parse("A dog.")
        assert result.metta_atoms == []
        assert result.confidence == 0.0

    def test_to_dict_roundtrip(self):
        response = "(: e1 (Member sk_1 dog) (STV 1.0 0.99))"
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse("A dog.", domain="animals")
        d = result.to_dict()
        assert d["metta_atoms"] == ["(: e1 (Member sk_1 dog) (STV 1.0 0.99))"]
        assert d["domain"] == "animals"
        assert d["is_query"] is False
        restored = ParsedSemantics.from_dict(d)
        assert restored.metta_atoms == result.metta_atoms
        assert restored.domain == result.domain

    def test_blank_lines_ignored(self):
        response = "\n\n(: e1 (Member sk_1 dog) (STV 1.0 0.99))\n\n\n"
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse("A dog.")
        assert len(result.metta_atoms) == 1

    def test_assertion_without_stv(self):
        response = "(: e1 (Member sk_1 dog))"
        fake = FakeLLM(response)
        parser = PeTTaChainerParser(fake)
        result = parser.parse("A dog.")
        assert len(result.metta_atoms) == 1
        assert result.confidence == 1.0
