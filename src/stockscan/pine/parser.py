"""Pine v6 parser — logical lines → AST.

Supported statements: variable declarations (typed / var / varip), `:=`
reassignment, tuple destructuring, `if`/`else if`/`else` blocks, user function
definitions (`name(params) =>`), and expression statements. `for`, `while`,
`switch`, `type`, and `import` are outside the subset and raise
PineUnsupportedError with the line number.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .errors import PineSyntaxError, PineUnsupportedError
from .lexer import LogicalLine, Token, tokenize

_node_ids = itertools.count()

TYPE_NAMES = {
    "int", "float", "bool", "string", "color", "table", "label", "line",
    "box", "array", "matrix", "map", "series", "simple",
}


@dataclass
class Node:
    line: int = 0
    node_id: int = field(default_factory=lambda: next(_node_ids))


@dataclass
class Num(Node):
    value: float = 0.0


@dataclass
class Str(Node):
    value: str = ""


@dataclass
class ColorLit(Node):
    value: str = ""


@dataclass
class Bool(Node):
    value: bool = False


@dataclass
class NaLit(Node):
    pass


@dataclass
class Ident(Node):
    name: str = ""


@dataclass
class Call(Node):
    func: str = ""  # dotted name, e.g. "ta.sma"
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    type_arg: str | None = None  # array.new<float>


@dataclass
class Index(Node):
    target: Node = None
    offset: Node = None


@dataclass
class BinOp(Node):
    op: str = ""
    left: Node = None
    right: Node = None


@dataclass
class UnaryOp(Node):
    op: str = ""
    operand: Node = None


@dataclass
class Ternary(Node):
    cond: Node = None
    if_true: Node = None
    if_false: Node = None


@dataclass
class TupleExpr(Node):
    items: list = field(default_factory=list)


@dataclass
class Assign(Node):
    name: str = ""
    expr: Node = None
    declared_var: bool = False   # `var` / `varip`
    is_reassign: bool = False    # `:=`


@dataclass
class TupleAssign(Node):
    names: list[str] = field(default_factory=list)
    expr: Node = None


@dataclass
class If(Node):
    cond: Node = None
    body: list = field(default_factory=list)
    orelse: list = field(default_factory=list)


@dataclass
class FuncDef(Node):
    name: str = ""
    params: list[str] = field(default_factory=list)
    body: list = field(default_factory=list)  # statements; last one is the value


@dataclass
class ExprStmt(Node):
    expr: Node = None


@dataclass
class Script:
    statements: list = field(default_factory=list)
    source_name: str = ""


def parse(source: str, source_name: str = "<pine>") -> Script:
    lines = tokenize(source)
    statements, pos = _parse_block(lines, 0, depth=0)
    if pos != len(lines):
        raise PineSyntaxError("unexpected trailing content", lines[pos].line)
    return Script(statements=statements, source_name=source_name)


def _parse_block(lines: list[LogicalLine], pos: int, depth: int) -> tuple[list, int]:
    statements: list = []
    while pos < len(lines) and lines[pos].depth >= depth:
        if lines[pos].depth > depth:
            raise PineSyntaxError("unexpected indent", lines[pos].line)
        stmt, pos = _parse_statement(lines, pos)
        if stmt is not None:
            statements.append(stmt)
    return statements, pos


def _parse_statement(lines: list[LogicalLine], pos: int) -> tuple[Node | None, int]:
    line = lines[pos]
    tokens = line.tokens
    p = _P(tokens, line.line)

    first = tokens[0]
    if first.kind == "KW":
        if first.value in ("for", "while", "switch", "type", "import", "export", "enum"):
            raise PineUnsupportedError(f"'{first.value}' is not supported", line.line)
        if first.value == "if":
            return _parse_if(lines, pos)
        if first.value == "else":
            raise PineSyntaxError("'else' without matching 'if'", line.line)
        if first.value in ("var", "varip"):
            p.take()  # var
            _skip_type(p)
            name = p.expect_id()
            p.expect_op("=")
            expr = _parse_expression(p)
            p.expect_end()
            return Assign(line=line.line, name=name, expr=expr, declared_var=True), pos + 1

    # tuple destructuring: [a, b, c] = expr
    if first.kind == "OP" and first.value == "[":
        p.take()
        names = [p.expect_id()]
        while p.peek_op(","):
            p.take()
            names.append(p.expect_id())
        p.expect_op("]")
        p.expect_op("=")
        expr = _parse_expression(p)
        p.expect_end()
        return TupleAssign(line=line.line, names=names, expr=expr), pos + 1

    # typed declaration: `int x = 0` / `float y = ...`
    if (
        first.kind == "ID"
        and first.value in TYPE_NAMES
        and len(tokens) > 1
        and tokens[1].kind == "ID"
    ):
        p.take()
        name = p.expect_id()
        p.expect_op("=")
        expr = _parse_expression(p)
        p.expect_end()
        return Assign(line=line.line, name=name, expr=expr), pos + 1

    # function definition: name(params) => ...
    fdef = _try_parse_funcdef(lines, pos)
    if fdef is not None:
        return fdef

    if first.kind == "ID" and len(tokens) > 1 and tokens[1].kind == "OP":
        if tokens[1].value == "=":
            p.take()
            name = first.value
            p.take()
            expr = _parse_expression(p)
            p.expect_end()
            return Assign(line=line.line, name=name, expr=expr), pos + 1
        if tokens[1].value == ":=":
            p.take()
            name = first.value
            p.take()
            expr = _parse_expression(p)
            p.expect_end()
            return Assign(line=line.line, name=name, expr=expr, is_reassign=True), pos + 1

    expr = _parse_expression(p)
    p.expect_end()
    return ExprStmt(line=line.line, expr=expr), pos + 1


def _try_parse_funcdef(lines: list[LogicalLine], pos: int):
    tokens = lines[pos].tokens
    if not (tokens[0].kind == "ID" and len(tokens) > 1 and tokens[1].value == "("):
        return None
    depth = 0
    arrow_at = None
    for i, tok in enumerate(tokens[1:], start=1):
        if tok.kind == "OP":
            if tok.value == "(":
                depth += 1
            elif tok.value == ")":
                depth -= 1
                if depth == 0 and i + 1 < len(tokens) and tokens[i + 1].value == "=>":
                    arrow_at = i + 1
                    break
                if depth == 0:
                    return None
    if arrow_at is None:
        return None

    line = lines[pos]
    p = _P(tokens, line.line)
    name = p.expect_id()
    p.expect_op("(")
    params: list[str] = []
    while not p.peek_op(")"):
        _skip_type(p)
        params.append(p.expect_id())
        if p.peek_op(","):
            p.take()
    p.expect_op(")")
    p.expect_op("=>")

    if not p.at_end():  # single-line body
        expr = _parse_expression(p)
        p.expect_end()
        return FuncDef(line=line.line, name=name, params=params,
                       body=[ExprStmt(line=line.line, expr=expr)]), pos + 1
    body, next_pos = _parse_block(lines, pos + 1, depth=line.depth + 1)
    if not body:
        raise PineSyntaxError(f"empty body for function {name}", line.line)
    return FuncDef(line=line.line, name=name, params=params, body=body), next_pos


def _parse_if(lines: list[LogicalLine], pos: int) -> tuple[Node, int]:
    line = lines[pos]
    p = _P(line.tokens, line.line)
    p.take()  # if
    cond = _parse_expression(p)
    p.expect_end()
    body, pos = _parse_block(lines, pos + 1, depth=line.depth + 1)
    orelse: list = []
    if (
        pos < len(lines)
        and lines[pos].depth == line.depth
        and lines[pos].tokens[0].kind == "KW"
        and lines[pos].tokens[0].value == "else"
    ):
        else_tokens = lines[pos].tokens
        if len(else_tokens) > 1 and else_tokens[1].kind == "KW" and else_tokens[1].value == "if":
            trimmed = LogicalLine(
                depth=lines[pos].depth, tokens=else_tokens[1:], line=lines[pos].line
            )
            nested, pos = _parse_if(lines[:pos] + [trimmed] + lines[pos + 1 :], pos)
            orelse = [nested]
        else:
            orelse, pos = _parse_block(lines, pos + 1, depth=line.depth + 1)
    return If(line=line.line, cond=cond, body=body, orelse=orelse), pos


def _skip_type(p: _P) -> None:
    """Skip an optional type annotation before an identifier."""
    if p.at_end() or p.current().kind != "ID":
        return
    if p.current().value not in TYPE_NAMES:
        return
    nxt = p.lookahead(1)
    if nxt is None:
        return
    if nxt.kind == "ID":
        p.take()
        return
    if nxt.kind == "OP" and nxt.value == "<":  # array<float> etc.
        p.take()
        p.expect_op("<")
        p.take()  # inner type
        p.expect_op(">")
        return


# ── Expression parsing (precedence climbing) ─────────────────────────────────


class _P:
    def __init__(self, tokens: list[Token], line: int):
        self.tokens = tokens
        self.i = 0
        self.line = line

    def at_end(self) -> bool:
        return self.i >= len(self.tokens)

    def current(self) -> Token:
        if self.at_end():
            raise PineSyntaxError("unexpected end of line", self.line)
        return self.tokens[self.i]

    def lookahead(self, k: int) -> Token | None:
        return self.tokens[self.i + k] if self.i + k < len(self.tokens) else None

    def take(self) -> Token:
        tok = self.current()
        self.i += 1
        return tok

    def peek_op(self, op: str) -> bool:
        return not self.at_end() and self.current().kind == "OP" and self.current().value == op

    def expect_op(self, op: str) -> None:
        if not self.peek_op(op):
            got = "end of line" if self.at_end() else repr(self.current().value)
            raise PineSyntaxError(f"expected {op!r}, got {got}", self.line)
        self.take()

    def expect_id(self) -> str:
        tok = self.current()
        if tok.kind != "ID":
            raise PineSyntaxError(f"expected identifier, got {tok.value!r}", self.line)
        self.take()
        return str(tok.value)

    def expect_end(self) -> None:
        if not self.at_end():
            raise PineSyntaxError(
                f"unexpected token {self.current().value!r}", self.line
            )


def _parse_expression(p: _P) -> Node:
    return _parse_ternary(p)


def _parse_ternary(p: _P) -> Node:
    cond = _parse_or(p)
    if p.peek_op("?"):
        p.take()
        if_true = _parse_ternary(p)
        p.expect_op(":")
        if_false = _parse_ternary(p)
        return Ternary(line=p.line, cond=cond, if_true=if_true, if_false=if_false)
    return cond


def _parse_or(p: _P) -> Node:
    node = _parse_and(p)
    while not p.at_end() and p.current().kind == "KW" and p.current().value == "or":
        p.take()
        node = BinOp(line=p.line, op="or", left=node, right=_parse_and(p))
    return node


def _parse_and(p: _P) -> Node:
    node = _parse_not(p)
    while not p.at_end() and p.current().kind == "KW" and p.current().value == "and":
        p.take()
        node = BinOp(line=p.line, op="and", left=node, right=_parse_not(p))
    return node


def _parse_not(p: _P) -> Node:
    if not p.at_end() and p.current().kind == "KW" and p.current().value == "not":
        p.take()
        return UnaryOp(line=p.line, op="not", operand=_parse_not(p))
    return _parse_comparison(p)


_COMPARISONS = ("==", "!=", "<=", ">=", "<", ">")


def _parse_comparison(p: _P) -> Node:
    node = _parse_additive(p)
    while not p.at_end() and p.current().kind == "OP" and p.current().value in _COMPARISONS:
        op = str(p.take().value)
        node = BinOp(line=p.line, op=op, left=node, right=_parse_additive(p))
    return node


def _parse_additive(p: _P) -> Node:
    node = _parse_multiplicative(p)
    while not p.at_end() and p.current().kind == "OP" and p.current().value in ("+", "-"):
        op = str(p.take().value)
        node = BinOp(line=p.line, op=op, left=node, right=_parse_multiplicative(p))
    return node


def _parse_multiplicative(p: _P) -> Node:
    node = _parse_unary(p)
    while not p.at_end() and p.current().kind == "OP" and p.current().value in ("*", "/", "%"):
        op = str(p.take().value)
        node = BinOp(line=p.line, op=op, left=node, right=_parse_unary(p))
    return node


def _parse_unary(p: _P) -> Node:
    if p.peek_op("-"):
        p.take()
        return UnaryOp(line=p.line, op="-", operand=_parse_unary(p))
    if p.peek_op("+"):
        p.take()
        return _parse_unary(p)
    return _parse_postfix(p)


def _parse_postfix(p: _P) -> Node:
    node = _parse_primary(p)
    while not p.at_end() and p.peek_op("["):
        p.take()
        offset = _parse_expression(p)
        p.expect_op("]")
        node = Index(line=p.line, target=node, offset=offset)
    return node


def _parse_primary(p: _P) -> Node:
    tok = p.current()
    if tok.kind == "NUM":
        p.take()
        return Num(line=p.line, value=float(tok.value))
    if tok.kind == "STR":
        p.take()
        return Str(line=p.line, value=str(tok.value))
    if tok.kind == "COLOR":
        p.take()
        return ColorLit(line=p.line, value=str(tok.value))
    if tok.kind == "KW":
        if tok.value in ("true", "false"):
            p.take()
            return Bool(line=p.line, value=tok.value == "true")
        if tok.value == "na":
            p.take()
            # `na(x)` is a call; bare `na` is the literal
            if p.peek_op("("):
                return _parse_call(p, "na")
            return NaLit(line=p.line)
        raise PineSyntaxError(f"unexpected keyword {tok.value!r} in expression", p.line)
    if tok.kind == "OP" and tok.value == "(":
        p.take()
        node = _parse_expression(p)
        p.expect_op(")")
        return node
    if tok.kind == "OP" and tok.value == "[":
        p.take()
        items = [_parse_expression(p)]
        while p.peek_op(","):
            p.take()
            items.append(_parse_expression(p))
        p.expect_op("]")
        return TupleExpr(line=p.line, items=items)
    if tok.kind == "ID":
        name = p.expect_id()
        while p.peek_op("."):
            nxt = p.lookahead(1)
            if nxt is None or nxt.kind != "ID":
                break
            p.take()
            name += "." + p.expect_id()
        # generic call: array.new<float>(...)
        if p.peek_op("<"):
            la1, la2, la3 = p.lookahead(1), p.lookahead(2), p.lookahead(3)
            if (
                la1 is not None and la1.kind == "ID"
                and la2 is not None and la2.kind == "OP" and la2.value == ">"
                and la3 is not None and la3.kind == "OP" and la3.value == "("
            ):
                p.take()
                type_arg = p.expect_id()
                p.take()  # >
                call = _parse_call(p, name)
                call.type_arg = type_arg
                return call
        if p.peek_op("("):
            return _parse_call(p, name)
        return Ident(line=p.line, name=name)
    raise PineSyntaxError(f"unexpected token {tok.value!r}", p.line)


def _parse_call(p: _P, func: str) -> Call:
    p.expect_op("(")
    args: list = []
    kwargs: dict = {}
    while not p.peek_op(")"):
        tok = p.current()
        nxt = p.lookahead(1)
        if (
            tok.kind == "ID"
            and nxt is not None
            and nxt.kind == "OP"
            and nxt.value == "="
        ):
            key = p.expect_id()
            p.take()  # =
            kwargs[key] = _parse_expression(p)
        else:
            if kwargs:
                raise PineSyntaxError("positional argument after keyword argument", p.line)
            args.append(_parse_expression(p))
        if p.peek_op(","):
            p.take()
    p.expect_op(")")
    return Call(line=p.line, func=func, args=args, kwargs=kwargs)
