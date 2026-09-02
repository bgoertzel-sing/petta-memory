from __future__ import annotations

from dataclasses import dataclass


class SExpressionSyntaxError(ValueError):
    """Raised when text is not valid S-expression syntax."""


@dataclass(frozen=True)
class StringAtom:
    """A quoted string token, stored with its decoded Python value."""

    value: str


SExpr = str | StringAtom | tuple["SExpr", ...]


_SYMBOL_FORBIDDEN = set('()";')


def parse_top_level_lists(text: str) -> tuple[tuple[SExpr, ...], ...]:
    """Parse text as one or more top-level S-expression lists.

    The parser is intentionally small but real: it tracks balanced parentheses,
    quoted strings with backslash escapes, and semicolon comments outside strings.
    Top-level atoms must be lists because `.metta` memory journal entries are
    predicate applications such as `(MemoryCluster mc1)`.
    """

    parser = _Parser(text)
    forms = parser.parse_all()
    for form in forms:
        if not isinstance(form, tuple):
            raise SExpressionSyntaxError(f"top-level form is not a list: {to_source(form)}")
        if not form:
            raise SExpressionSyntaxError("empty top-level list is not a memory atom")
    return tuple(forms)  # type: ignore[return-value]


def parse_one_list(text: str) -> tuple[SExpr, ...]:
    forms = parse_top_level_lists(text)
    if len(forms) != 1:
        raise SExpressionSyntaxError("expected exactly one top-level atom")
    return forms[0]


def to_source(expr: SExpr) -> str:
    if isinstance(expr, tuple):
        return "(" + " ".join(to_source(part) for part in expr) + ")"
    if isinstance(expr, StringAtom):
        return '"' + _escape_string(expr.value) + '"'
    return expr


def symbol_text(expr: SExpr) -> str | None:
    return expr if isinstance(expr, str) else None


def _escape_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0

    def parse_all(self) -> list[SExpr]:
        out: list[SExpr] = []
        while True:
            self._skip_ws_and_comments()
            if self.i >= len(self.text):
                return out
            out.append(self._parse_expr())

    def _parse_expr(self) -> SExpr:
        self._skip_ws_and_comments()
        if self.i >= len(self.text):
            raise SExpressionSyntaxError("unexpected end of input")
        ch = self.text[self.i]
        if ch == "(":
            return self._parse_list()
        if ch == ")":
            raise SExpressionSyntaxError("unexpected ')' outside list")
        if ch == '"':
            return self._parse_string()
        return self._parse_symbol()

    def _parse_list(self) -> tuple[SExpr, ...]:
        self.i += 1  # '('
        items: list[SExpr] = []
        while True:
            self._skip_ws_and_comments()
            if self.i >= len(self.text):
                raise SExpressionSyntaxError("unclosed '('")
            if self.text[self.i] == ")":
                self.i += 1
                return tuple(items)
            items.append(self._parse_expr())

    def _parse_string(self) -> StringAtom:
        self.i += 1  # opening quote
        chars: list[str] = []
        while self.i < len(self.text):
            ch = self.text[self.i]
            self.i += 1
            if ch == '"':
                return StringAtom("".join(chars))
            if ch == "\\":
                if self.i >= len(self.text):
                    raise SExpressionSyntaxError("unfinished string escape")
                esc = self.text[self.i]
                self.i += 1
                chars.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(esc, esc))
            else:
                chars.append(ch)
        raise SExpressionSyntaxError("unclosed string literal")

    def _parse_symbol(self) -> str:
        start = self.i
        while self.i < len(self.text):
            ch = self.text[self.i]
            if ch.isspace() or ch in _SYMBOL_FORBIDDEN:
                break
            self.i += 1
        if self.i == start:
            raise SExpressionSyntaxError(f"unexpected character: {self.text[self.i]!r}")
        return self.text[start : self.i]

    def _skip_ws_and_comments(self) -> None:
        while self.i < len(self.text):
            ch = self.text[self.i]
            if ch.isspace():
                self.i += 1
                continue
            if ch == ";":
                while self.i < len(self.text) and self.text[self.i] not in "\r\n":
                    self.i += 1
                continue
            break
