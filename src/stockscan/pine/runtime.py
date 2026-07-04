"""Pine bar-loop runtime.

Executes a parsed script once per bar, oldest to newest, faithfully to Pine's
execution model: `var` declarations persist and are carried forward at bar
start, every assigned variable keeps full history (``x[n]``), and each
stateful built-in call site owns its own rolling state (keyed by AST node and
user-function call path).

The scanner feeds completed bars only, so ``barstate.isconfirmed`` is always
true and ``barstate.islast`` marks the final bar.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..sessions import EASTERN
from .errors import PineRuntimeError, PineUnsupportedError
from .parser import (
    Assign,
    BinOp,
    Bool,
    Call,
    ColorLit,
    ExprStmt,
    FuncDef,
    Ident,
    If,
    Index,
    NaLit,
    Node,
    Num,
    Script,
    Str,
    Ternary,
    TupleAssign,
    TupleExpr,
    UnaryOp,
)

NA = float("nan")


def is_na(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def truthy(v: Any) -> bool:
    if is_na(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float)):
        return v != 0
    raise PineRuntimeError(f"value {v!r} cannot be used as a condition")


class PineArray:
    __slots__ = ("items",)

    def __init__(self) -> None:
        self.items: list[Any] = []


class Opaque:
    """Placeholder for colors, plot ids, tables, ... — carried, never inspected."""

    __slots__ = ("kind", "info")

    def __init__(self, kind: str, info: Any = None):
        self.kind = kind
        self.info = info

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.kind}>"


class Slot:
    """Per-variable history: one value per bar."""

    __slots__ = ("values", "is_var", "initialized")

    def __init__(self, n: int, is_var: bool = False):
        self.values: list[Any] = [None] * n
        self.is_var = is_var
        self.initialized = False


class RunResult:
    def __init__(self, n: int):
        self.n = n
        self.columns: dict[str, list[Any]] = {}
        self.title: str = ""
        self.shorttitle: str = ""
        self.inputs: dict[str, Any] = {}

    def column(self, name: str) -> list[Any]:
        """Allocate a fresh column, deduplicating names (one column per plot
        call site — callers cache the returned list)."""
        base, k = name, 2
        while name in self.columns:
            name = f"{base}_{k}"
            k += 1
        self.columns[name] = [None] * self.n
        return self.columns[name]

    def to_frame(self, index: pd.Index) -> pd.DataFrame:
        data = {}
        for name, values in self.columns.items():
            cleaned = [NA if v is None else v for v in values]
            data[name] = cleaned
        return pd.DataFrame(data, index=index)


def slugify(title: str) -> str:
    out = []
    prev_us = True
    for ch in title.lower():
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    slug = "".join(out).strip("_")
    return slug or "col"


class Interpreter:
    def __init__(
        self,
        script: Script,
        input_overrides: dict[str, Any] | None = None,
        *,
        subrun: bool = False,
        symbol: str = "",
    ):
        self.script = script
        self.input_overrides = dict(input_overrides or {})
        self.subrun = subrun
        self.symbol = symbol

        self.bars: pd.DataFrame | None = None
        self.n = 0
        self.i = 0  # current bar
        self.globals: dict[str, Slot] = {}
        self.functions: dict[str, FuncDef] = {}
        self.call_stack: tuple[int, ...] = ()
        self.local_frames: list[dict[str, Any]] = []
        self.ta_state: dict[tuple, Any] = {}
        self.input_cache: dict[int, Any] = {}
        self.security_cache: dict[int, Any] = {}
        self.matched_overrides: set[str] = set()
        self.result: RunResult | None = None
        self._plot_counter = 0

        # context series filled per bar
        self._ctx: dict[str, Any] = {}
        self._ctx_hist: dict[str, list[float]] = {}

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self, bars: pd.DataFrame) -> RunResult:
        from . import builtins as bi  # late import (module cross-references)

        self.builtins = bi
        self.bars = bars
        self.n = len(bars)
        self.result = RunResult(self.n)

        opens = bars["open"].to_numpy(dtype=float)
        highs = bars["high"].to_numpy(dtype=float)
        lows = bars["low"].to_numpy(dtype=float)
        closes = bars["close"].to_numpy(dtype=float)
        volumes = bars["volume"].to_numpy(dtype=float)
        hlc3 = (highs + lows + closes) / 3.0
        hl2 = (highs + lows) / 2.0
        ohlc4 = (opens + highs + lows + closes) / 4.0
        is_rth = (bars["session"] == "rth").to_numpy()
        self._ctx_hist = {
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "hlc3": hlc3, "hl2": hl2, "ohlc4": ohlc4,
        }

        for i in range(self.n):
            self.i = i
            self._ctx = {
                "open": opens[i], "high": highs[i], "low": lows[i],
                "close": closes[i], "volume": volumes[i],
                "hlc3": hlc3[i], "hl2": hl2[i], "ohlc4": ohlc4[i],
                "bar_index": float(i),
                "session.ismarket": bool(is_rth[i]),
                "barstate.isconfirmed": True,
                "barstate.islast": i == self.n - 1,
                "barstate.isfirst": i == 0,
                "syminfo.tickerid": self.symbol,
                "syminfo.ticker": self.symbol,
            }
            for slot in self.globals.values():
                if slot.is_var and slot.initialized and i > 0:
                    slot.values[i] = slot.values[i - 1]
            for stmt in self.script.statements:
                self.exec_stmt(stmt)

        unmatched = set(self.input_overrides) - self.matched_overrides
        if unmatched:
            raise PineRuntimeError(
                f"input override(s) {sorted(unmatched)} matched no input; "
                f"available titles: {sorted(self.result.inputs)}"
            )
        return self.result

    # ── statements ───────────────────────────────────────────────────────────

    def exec_stmt(self, stmt: Node) -> Any:
        if isinstance(stmt, Assign):
            return self._exec_assign(stmt)
        if isinstance(stmt, TupleAssign):
            return self._exec_tuple_assign(stmt)
        if isinstance(stmt, If):
            return self._exec_if(stmt)
        if isinstance(stmt, FuncDef):
            self.functions[stmt.name] = stmt
            return None
        if isinstance(stmt, ExprStmt):
            return self.eval(stmt.expr)
        raise PineUnsupportedError(f"unsupported statement {type(stmt).__name__}", stmt.line)

    def _exec_assign(self, stmt: Assign) -> None:
        if self.local_frames:
            frame = self.local_frames[-1]
            if stmt.is_reassign and stmt.name not in frame:
                # Pine forbids assigning globals from functions; if-blocks at
                # global depth are not local frames, so this is function scope.
                raise PineUnsupportedError(
                    f"function cannot reassign outer variable {stmt.name!r}", stmt.line
                )
            frame[stmt.name] = self.eval(stmt.expr)
            return

        slot = self.globals.get(stmt.name)
        if stmt.is_reassign:
            if slot is None:
                raise PineRuntimeError(f"cannot ':=' undeclared variable {stmt.name!r}", stmt.line)
            slot.values[self.i] = self.eval(stmt.expr)
            return
        if slot is None:
            slot = Slot(self.n, is_var=stmt.declared_var)
            self.globals[stmt.name] = slot
        if slot.is_var:
            if not slot.initialized:
                slot.values[self.i] = self.eval(stmt.expr)
                slot.initialized = True
            return
        slot.values[self.i] = self.eval(stmt.expr)

    def _exec_tuple_assign(self, stmt: TupleAssign) -> None:
        value = self.eval(stmt.expr)
        if not isinstance(value, tuple) or len(value) != len(stmt.names):
            raise PineRuntimeError(
                f"expected a {len(stmt.names)}-tuple on the right of destructuring", stmt.line
            )
        if self.local_frames:
            self.local_frames[-1].update(zip(stmt.names, value, strict=True))
            return
        for name, item in zip(stmt.names, value, strict=True):
            slot = self.globals.get(name)
            if slot is None:
                slot = Slot(self.n)
                self.globals[name] = slot
            slot.values[self.i] = item

    def _exec_if(self, stmt: If) -> Any:
        branch = stmt.body if truthy(self.eval(stmt.cond)) else stmt.orelse
        value: Any = None
        for sub in branch:
            value = self.exec_stmt(sub)
        return value

    # ── expressions ──────────────────────────────────────────────────────────

    def eval(self, node: Node) -> Any:
        method = getattr(self, f"_eval_{type(node).__name__}", None)
        if method is None:
            raise PineUnsupportedError(f"unsupported expression {type(node).__name__}", node.line)
        return method(node)

    def _eval_Num(self, node: Num) -> float:
        return node.value

    def _eval_Str(self, node: Str) -> str:
        return node.value

    def _eval_Bool(self, node: Bool) -> bool:
        return node.value

    def _eval_NaLit(self, node: NaLit) -> float:
        return NA

    def _eval_ColorLit(self, node: ColorLit) -> Opaque:
        return Opaque("color", node.value)

    def _eval_TupleExpr(self, node: TupleExpr) -> tuple:
        return tuple(self.eval(item) for item in node.items)

    def _eval_Ident(self, node: Ident) -> Any:
        name = node.name
        if self.local_frames and name in self.local_frames[-1]:
            return self.local_frames[-1][name]
        if name in self.globals:
            return self.globals[name].values[self.i]
        if name in self._ctx:
            return self._ctx[name]
        const = self.builtins.constant(name)
        if const is not NotImplemented:
            return const
        raise PineRuntimeError(f"undefined name {name!r}", node.line)

    def _eval_Index(self, node: Index) -> Any:
        offset = self.eval(node.offset)
        if is_na(offset):
            return NA
        k = int(offset)
        target = node.target
        if not isinstance(target, Ident):
            raise PineUnsupportedError(
                "history access [] is supported on variables and built-in series only",
                node.line,
            )
        j = self.i - k
        if j < 0:
            return NA
        name = target.name
        if self.local_frames and name in self.local_frames[-1]:
            raise PineUnsupportedError(
                f"history access on function-local {name!r} is not supported", node.line
            )
        if name in self.globals:
            v = self.globals[name].values[j]
            return NA if v is None else v
        if name in self._ctx_hist:
            return float(self._ctx_hist[name][j])
        raise PineRuntimeError(f"undefined series {name!r}", node.line)

    def _eval_UnaryOp(self, node: UnaryOp) -> Any:
        v = self.eval(node.operand)
        if node.op == "-":
            return NA if is_na(v) else -v
        if node.op == "not":
            return NA if is_na(v) else not truthy(v)
        raise PineUnsupportedError(f"unary {node.op}", node.line)

    def _eval_BinOp(self, node: BinOp) -> Any:
        op = node.op
        left = self.eval(node.left)
        right = self.eval(node.right)  # Pine does not short-circuit series logic
        if op == "and":
            return truthy(left) and truthy(right)
        if op == "or":
            return truthy(left) or truthy(right)
        if op == "+":
            if isinstance(left, str) or isinstance(right, str):
                if is_na(left) or is_na(right):
                    return NA
                return str(left) + str(right)
        if is_na(left) or is_na(right):
            return NA
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return NA if right == 0 else left / right
        if op == "%":
            return NA if right == 0 else math.fmod(left, right)
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        raise PineUnsupportedError(f"operator {op}", node.line)

    def _eval_Ternary(self, node: Ternary) -> Any:
        # Evaluate both branches like Pine does for series expressions, then
        # select (an na condition selects the false branch).
        cond = self.eval(node.cond)
        if_true = self.eval(node.if_true)
        if_false = self.eval(node.if_false)
        return if_true if truthy(cond) else if_false

    def _eval_Call(self, node: Call) -> Any:
        func = node.func
        if func in self.functions:
            return self._call_user_function(node, self.functions[func])
        return self.builtins.call(self, node)

    def _call_user_function(self, node: Call, fdef: FuncDef) -> Any:
        if node.kwargs:
            raise PineUnsupportedError(
                "keyword arguments on user functions are not supported", node.line
            )
        if len(node.args) != len(fdef.params):
            raise PineRuntimeError(
                f"{fdef.name}() expects {len(fdef.params)} argument(s), got {len(node.args)}",
                node.line,
            )
        frame = {
            param: self.eval(arg) for param, arg in zip(fdef.params, node.args, strict=True)
        }
        self.call_stack = (*self.call_stack, node.node_id)
        self.local_frames.append(frame)
        try:
            value: Any = None
            for stmt in fdef.body:
                value = self.exec_stmt(stmt)
            return value
        finally:
            self.local_frames.pop()
            self.call_stack = self.call_stack[:-1]

    # ── helpers for builtins ─────────────────────────────────────────────────

    def state_for(self, node: Node, factory) -> Any:
        key = (self.call_stack, node.node_id)
        state = self.ta_state.get(key)
        if state is None:
            state = factory()
            self.ta_state[key] = state
        return state

    def resample_to(self, pine_tf: str) -> pd.DataFrame:
        """Aggregate the chart bars to a higher Pine timeframe string."""
        assert self.bars is not None
        freq = _pine_tf_to_freq(pine_tf)
        et = self.bars.copy()
        et.index = et.index.tz_convert(EASTERN)
        kwargs: dict[str, Any] = {}
        if freq.endswith("min") or freq.endswith("h"):
            kwargs = {"origin": "start_day", "offset": "9h30min"}
        agg = et.resample(freq, **kwargs).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "session": lambda s: "rth" if (s == "rth").any() else (s.iloc[0] if len(s) else "rth"),
            }
        )
        agg = agg.dropna(subset=["close"])
        agg.index = agg.index.tz_convert("UTC")
        return agg


def _pine_tf_to_freq(tf: str) -> str:
    tf = tf.strip()
    if tf.isdigit():
        minutes = int(tf)
        if minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}min"
    upper = tf.upper()
    if upper in ("D", "1D"):
        return "1D"
    if upper in ("W", "1W"):
        return "1W"
    if upper in ("M", "1M"):
        return "1MS"
    if upper.endswith("D") and upper[:-1].isdigit():
        return f"{int(upper[:-1])}D"
    raise PineUnsupportedError(f"timeframe {tf!r} is not supported")
