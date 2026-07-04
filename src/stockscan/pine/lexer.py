"""Pine v6 lexer.

Produces *logical lines* of tokens. Pine's layout rules:
- a statement starts at an indent that is a multiple of 4 (block depth = indent // 4)
- a physical line indented by a non-multiple of 4 continues the previous
  logical line (TradingView's line-wrapping convention)
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PineSyntaxError

KEYWORDS = {
    "var", "varip", "if", "else", "for", "to", "by", "while", "switch",
    "and", "or", "not", "true", "false", "na",
    "import", "export", "method", "type", "enum",
}

# multi-char operators first
OPERATORS = [
    "=>", "==", "!=", "<=", ">=", ":=", "?", ":", ",", "(", ")", "[", "]",
    "+", "-", "*", "/", "%", "<", ">", "=", ".",
]


@dataclass(frozen=True)
class Token:
    kind: str  # NUM, STR, ID, KW, OP, COLOR
    value: str | float
    line: int


@dataclass
class LogicalLine:
    depth: int
    tokens: list[Token]
    line: int


def tokenize(source: str) -> list[LogicalLine]:
    logical: list[LogicalLine] = []
    for lineno, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("//"):
            continue
        indent = len(raw) - len(raw.lstrip(" \t"))
        if "\t" in raw[:indent]:
            indent = len(raw[:indent].expandtabs(4))
        tokens = _tokenize_line(raw, lineno)
        if not tokens:
            continue
        if indent % 4 != 0 and logical:
            logical[-1].tokens.extend(tokens)
        else:
            logical.append(LogicalLine(depth=indent // 4, tokens=tokens, line=lineno))
    return logical


def _tokenize_line(text: str, lineno: int) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            break  # comment to end of line
        if ch == '"' or ch == "'":
            value, i = _read_string(text, i, lineno)
            tokens.append(Token("STR", value, lineno))
            continue
        if ch == "#":
            j = i + 1
            while j < n and text[j] in "0123456789abcdefABCDEF":
                j += 1
            tokens.append(Token("COLOR", text[i:j], lineno))
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            seen_dot = False
            while j < n and (text[j].isdigit() or (text[j] == "." and not seen_dot)):
                if text[j] == ".":
                    # ".." range? not valid Pine; but guard member access `1.x`
                    if j + 1 < n and not text[j + 1].isdigit():
                        break
                    seen_dot = True
                j += 1
            if j < n and text[j] in "eE" and j + 1 < n and (
                text[j + 1].isdigit() or text[j + 1] in "+-"
            ):
                j += 2
                while j < n and text[j].isdigit():
                    j += 1
            tokens.append(Token("NUM", float(text[i:j]), lineno))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            tokens.append(Token("KW" if word in KEYWORDS else "ID", word, lineno))
            i = j
            continue
        for op in OPERATORS:
            if text.startswith(op, i):
                tokens.append(Token("OP", op, lineno))
                i += len(op)
                break
        else:
            raise PineSyntaxError(f"unexpected character {ch!r}", lineno)
    return tokens


def _read_string(text: str, start: int, lineno: int) -> tuple[str, int]:
    quote = text[start]
    out: list[str] = []
    i = start + 1
    escapes = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'"}
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(escapes.get(text[i + 1], text[i + 1]))
            i += 2
            continue
        if ch == quote:
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise PineSyntaxError("unterminated string", lineno)
