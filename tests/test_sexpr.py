"""Direct tests for the S-expression parser (petta_memory.sexpr).

The parser is foundational (used by store.py, omegaclaw.py, patham9_pln.py,
pettachainer_profile.py, pipln_models.py, usability_bundle.py) but had no
dedicated test file. These tests exercise every public function and the
internal _Parser class across normal and error paths.
"""
import pytest
from petta_memory.sexpr import (
    SExpressionSyntaxError,
    StringAtom,
    parse_top_level_lists,
    parse_one_list,
    to_source,
    symbol_text,
)


# ── parse_top_level_lists ────────────────────────────────────────────

class TestParseTopLevelLists:
    def test_single_atom(self):
        result = parse_top_level_lists("(MemoryCluster mc1)")
        assert len(result) == 1
        assert result[0] == ("MemoryCluster", "mc1")

    def test_multiple_atoms(self):
        result = parse_top_level_lists("(HasStatus e1 active)\n(ClusterType tc Observation)")
        assert len(result) == 2
        assert result[0] == ("HasStatus", "e1", "active")
        assert result[1] == ("ClusterType", "tc", "Observation")

    def test_empty_text_returns_empty(self):
        result = parse_top_level_lists("")
        assert result == ()

    def test_whitespace_only_returns_empty(self):
        result = parse_top_level_lists("   \n\t  \n")
        assert result == ()

    def test_comments_ignored(self):
        result = parse_top_level_lists("; this is a comment\n(MemoryCluster mc1)")
        assert len(result) == 1

    def test_inline_comment_ignored(self):
        result = parse_top_level_lists("(MemoryCluster mc1) ; trailing comment")
        assert len(result) == 1

    def test_nested_lists(self):
        result = parse_top_level_lists("(Outer (Inner a b) c)")
        assert result[0] == ("Outer", ("Inner", "a", "b"), "c")

    def test_deeply_nested(self):
        result = parse_top_level_lists("(A (B (C (D e))))")
        assert result[0] == ("A", ("B", ("C", ("D", "e"))))

    def test_string_atom_value(self):
        result = parse_top_level_lists('(HasName e1 "hello world")')
        assert len(result) == 1
        assert result[0][0] == "HasName"
        assert isinstance(result[0][2], StringAtom)
        assert result[0][2].value == "hello world"

    def test_string_with_escaped_quote(self):
        result = parse_top_level_lists('(Said "he said \\"hi\\"")')
        assert result[0][1].value == 'he said "hi"'

    def test_string_with_escaped_backslash(self):
        result = parse_top_level_lists('(Path "C:\\\\dir")')
        assert result[0][1].value == 'C:\\dir'

    def test_string_with_escaped_newline(self):
        result = parse_top_level_lists('(Text "line1\\nline2")')
        assert result[0][1].value == "line1\nline2"

    def test_string_with_escaped_tab(self):
        result = parse_top_level_lists('(Text "a\\tb")')
        assert result[0][1].value == "a\tb"

    def test_string_with_unknown_escape(self):
        """Unknown escape chars are kept literally."""
        result = parse_top_level_lists('(Text "a\\zb")')
        assert result[0][1].value == "azb"

    def test_top_level_bare_atom_rejected(self):
        with pytest.raises(SExpressionSyntaxError, match="not a list"):
            parse_top_level_lists("bareword")

    def test_top_level_string_rejected(self):
        with pytest.raises(SExpressionSyntaxError, match="not a list"):
            parse_top_level_lists('"just a string"')

    def test_empty_list_rejected(self):
        with pytest.raises(SExpressionSyntaxError, match="empty top-level list"):
            parse_top_level_lists("()")

    def test_unclosed_paren_raises(self):
        with pytest.raises(SExpressionSyntaxError, match="unclosed"):
            parse_top_level_lists("(MemoryCluster mc1")

    def test_unexpected_closing_paren_raises(self):
        with pytest.raises(SExpressionSyntaxError, match="unexpected '\\)'"):
            parse_top_level_lists(")")

    def test_unclosed_string_raises(self):
        with pytest.raises(SExpressionSyntaxError, match="unclosed string"):
            parse_top_level_lists('(Text "no end quote)')

    def test_unfinished_escape_raises(self):
        with pytest.raises(SExpressionSyntaxError, match="unclosed"):
            parse_top_level_lists('(Text "trailing backslash\\")')

    def test_empty_string_atom(self):
        result = parse_top_level_lists('(HasName e1 "")')
        assert isinstance(result[0][2], StringAtom)
        assert result[0][2].value == ""


# ── parse_one_list ──────────────────────────────────────────────────

class TestParseOneList:
    def test_single_list(self):
        result = parse_one_list("(MemoryCluster mc1)")
        assert result == ("MemoryCluster", "mc1")

    def test_zero_lists_raises(self):
        with pytest.raises(SExpressionSyntaxError, match="exactly one"):
            parse_one_list("")

    def test_two_lists_raises(self):
        with pytest.raises(SExpressionSyntaxError, match="exactly one"):
            parse_one_list("(A b)(C d)")


# ── to_source ────────────────────────────────────────────────────────

class TestToSource:
    def test_round_trip_simple(self):
        src = "(MemoryCluster mc1)"
        parsed = parse_top_level_lists(src)
        assert to_source(parsed[0]) == src

    def test_round_trip_nested(self):
        src = "(Outer (Inner a b) c)"
        parsed = parse_top_level_lists(src)
        assert to_source(parsed[0]) == src

    def test_string_round_trip(self):
        src = '(HasName e1 "hello")'
        parsed = parse_top_level_lists(src)
        assert to_source(parsed[0]) == src

    def test_string_with_special_chars(self):
        src = '(Text "line1\\nline2\\t\\"quote\\"")'
        parsed = parse_top_level_lists(src)
        assert to_source(parsed[0]) == src

    def test_empty_string_to_source(self):
        result = parse_top_level_lists('(S "")')
        assert to_source(result[0]) == '(S "")'

    def test_backslash_in_string(self):
        result = parse_top_level_lists('(Path "C:\\\\folder")')
        assert to_source(result[0]) == '(Path "C:\\\\folder")'


# ── symbol_text ──────────────────────────────────────────────────────

class TestSymbolText:
    def test_symbol_returns_itself(self):
        assert symbol_text("hello") == "hello"

    def test_string_atom_returns_none(self):
        assert symbol_text(StringAtom("value")) is None

    def test_tuple_returns_none(self):
        assert symbol_text(("a", "b")) is None

    def test_empty_string_symbol(self):
        """Empty string is not a valid symbol but symbol_text should still return it."""
        assert symbol_text("") == ""


# ── StringAtom ───────────────────────────────────────────────────────

class TestStringAtom:
    def test_value_stored(self):
        sa = StringAtom("test")
        assert sa.value == "test"

    def test_equality(self):
        assert StringAtom("a") == StringAtom("a")
        assert StringAtom("a") != StringAtom("b")

    def test_frozen(self):
        sa = StringAtom("x")
        with pytest.raises(Exception):
            sa.value = "y"
