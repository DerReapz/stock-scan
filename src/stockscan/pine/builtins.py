"""Pine built-in namespace: ta.*, math.*, input.*, array.*, plotting exports,
request.security, and the constant namespaces.

Stateful built-ins (everything under ``ta.`` plus ``math.sum``) keep per-call-
site rolling state via ``interp.state_for``. Rolling windows are summed with
``np.sum`` over the window array — the same pairwise algorithm the vectorized
native engines use — so exact-tie plateaus rank identically in both.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any

import numpy as np

from .errors import PineRuntimeError, PineUnsupportedError
from .parser import Call, Ident, Node, TupleExpr
from .runtime import NA, Interpreter, Opaque, PineArray, is_na, slugify, truthy

# ── constant namespaces ───────────────────────────────────────────────────────

_CONST_PREFIXES = (
    "display.", "location.", "size.", "shape.", "position.", "format.",
    "plot.", "hline.", "line.", "label.", "extend.", "scale.", "xloc.", "yloc.",
    "barmerge.", "text.", "font.", "order.", "currency.", "dayofweek.",
)

_COLOR_NAMES = {
    "aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon",
    "navy", "olive", "orange", "purple", "red", "silver", "teal", "white", "yellow",
}


def constant(name: str) -> Any:
    if name.startswith(_CONST_PREFIXES):
        return Opaque("const", name)
    if name.startswith("color."):
        tail = name.split(".", 1)[1]
        if tail in _COLOR_NAMES:
            return Opaque("color", tail)
    if name.startswith("chart."):
        return Opaque("chart", name)
    if name == "timeframe.period":
        return ""  # chart timeframe marker
    return NotImplemented


# ── stateful helpers ──────────────────────────────────────────────────────────


class _Window:
    """Fixed-length rolling window; result is na until full or if any na inside."""

    __slots__ = ("buf", "length")

    def __init__(self, length: int):
        self.length = max(1, int(length))
        self.buf: deque = deque(maxlen=self.length)

    def push(self, value: Any) -> None:
        self.buf.append(NA if is_na(value) else float(value))

    def full(self) -> bool:
        return len(self.buf) == self.length

    def array(self) -> np.ndarray:
        return np.asarray(self.buf, dtype=float)

    def clean(self) -> np.ndarray | None:
        if not self.full():
            return None
        arr = self.array()
        if np.isnan(arr).any():
            return None
        return arr


class _SeededEma:
    """Pine EMA/RMA: na until the first full SMA window of non-na values,
    seeded with that SMA, then recursive. An na input resets the seed."""

    __slots__ = ("alpha", "length", "window", "prev")

    def __init__(self, length: int, alpha: float):
        self.length = max(1, int(length))
        self.alpha = alpha
        self.window: deque = deque(maxlen=self.length)
        self.prev: float | None = None

    def update(self, value: Any) -> float:
        if is_na(value):
            self.prev = None
            self.window.clear()
            return NA
        x = float(value)
        if self.prev is None:
            self.window.append(x)
            if len(self.window) == self.length:
                self.prev = float(np.sum(np.asarray(self.window, dtype=float)) / self.length)
                return self.prev
            return NA
        self.prev = self.alpha * x + (1.0 - self.alpha) * self.prev
        return self.prev


class _Rsi:
    __slots__ = ("up", "down", "prev")

    def __init__(self, length: int):
        self.up = _SeededEma(length, 1.0 / max(1, int(length)))
        self.down = _SeededEma(length, 1.0 / max(1, int(length)))
        self.prev: float | None = None

    def update(self, value: Any) -> float:
        if is_na(value):
            self.prev = None
            return NA
        x = float(value)
        if self.prev is None:
            self.prev = x
            self.up.update(NA)  # keep streams aligned: first change is unknown
            self.down.update(NA)
            return NA
        change = x - self.prev
        self.prev = x
        up = self.up.update(max(change, 0.0))
        down = self.down.update(max(-change, 0.0))
        if is_na(up) or is_na(down):
            return NA
        if down == 0.0:
            return 100.0
        if up == 0.0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + up / down)


class _Cross:
    __slots__ = ("prev_a", "prev_b")

    def __init__(self):
        self.prev_a: Any = NA
        self.prev_b: Any = NA

    def update(self, a: Any, b: Any, over: bool) -> bool:
        pa, pb = self.prev_a, self.prev_b
        self.prev_a, self.prev_b = a, b
        if is_na(a) or is_na(b) or is_na(pa) or is_na(pb):
            return False
        if over:
            return a > b and pa <= pb
        return a < b and pa >= pb


class _BarsSince:
    __slots__ = ("count",)

    def __init__(self):
        self.count: float = NA

    def update(self, cond: Any) -> float:
        if truthy(cond):
            self.count = 0.0
        elif not is_na(self.count):
            self.count += 1.0
        return self.count


class _PrevClose:
    __slots__ = ("prev",)

    def __init__(self):
        self.prev: float = NA


# ── dispatch ──────────────────────────────────────────────────────────────────


def call(interp: Interpreter, node: Call) -> Any:
    handler = _HANDLERS.get(node.func)
    if handler is not None:
        return handler(interp, node)
    if node.func.startswith(("ta.", "math.", "str.", "array.", "request.", "input")):
        raise PineUnsupportedError(f"built-in {node.func}() is not supported", node.line)
    if node.func.split(".", 1)[0] in ("table", "label", "line", "box", "color"):
        _eval_all(interp, node)  # visual namespace: evaluate args, no-op
        return Opaque(node.func)
    raise PineUnsupportedError(f"unknown function {node.func}()", node.line)


def _eval_all(interp: Interpreter, node: Call) -> tuple[list, dict]:
    args = [interp.eval(a) for a in node.args]
    kwargs = {k: interp.eval(v) for k, v in node.kwargs.items()}
    return args, kwargs


def _arg(args: list, kwargs: dict, pos: int, name: str, default: Any = None) -> Any:
    if name in kwargs:
        return kwargs[name]
    if pos < len(args):
        return args[pos]
    return default


# ── ta.* ──────────────────────────────────────────────────────────────────────


def _ta_window(interp: Interpreter, node: Call, reducer) -> float:
    args, kwargs = _eval_all(interp, node)
    src = _arg(args, kwargs, 0, "source")
    length = _arg(args, kwargs, 1, "length")
    win: _Window = interp.state_for(node, lambda: _Window(int(length)))
    win.push(src)
    arr = win.clean()
    return NA if arr is None else reducer(arr)


def _ta_sma(interp, node):
    return _ta_window(interp, node, lambda a: float(np.sum(a) / len(a)))


def _ta_stdev(interp, node):
    def pop_std(a: np.ndarray) -> float:
        mean = np.sum(a) / len(a)
        return float(math.sqrt(np.sum((a - mean) ** 2) / len(a)))

    return _ta_window(interp, node, pop_std)


def _ta_median(interp, node):
    return _ta_window(interp, node, lambda a: float(statistics.median(a.tolist())))


def _ta_highest(interp, node):
    return _ta_window(interp, node, lambda a: float(np.max(a)))


def _ta_lowest(interp, node):
    return _ta_window(interp, node, lambda a: float(np.min(a)))


def _math_sum(interp, node):
    return _ta_window(interp, node, lambda a: float(np.sum(a)))


def _ta_percentrank(interp, node):
    args, kwargs = _eval_all(interp, node)
    src = _arg(args, kwargs, 0, "source")
    length = int(_arg(args, kwargs, 1, "length"))
    win: _Window = interp.state_for(node, lambda: _Window(length + 1))
    win.push(src)
    arr = win.clean()
    if arr is None:
        return NA
    current = arr[-1]
    return float(np.sum(arr[:-1] <= current)) * 100.0 / length


def _ta_ema(interp, node):
    args, kwargs = _eval_all(interp, node)
    src = _arg(args, kwargs, 0, "source")
    length = int(_arg(args, kwargs, 1, "length"))
    if length == 1:
        return NA if is_na(src) else float(src)
    state: _SeededEma = interp.state_for(node, lambda: _SeededEma(length, 2.0 / (length + 1.0)))
    return state.update(src)


def _ta_rma(interp, node):
    args, kwargs = _eval_all(interp, node)
    src = _arg(args, kwargs, 0, "source")
    length = int(_arg(args, kwargs, 1, "length"))
    if length == 1:
        return NA if is_na(src) else float(src)
    state: _SeededEma = interp.state_for(node, lambda: _SeededEma(length, 1.0 / length))
    return state.update(src)


def _ta_rsi(interp, node):
    args, kwargs = _eval_all(interp, node)
    src = _arg(args, kwargs, 0, "source")
    length = int(_arg(args, kwargs, 1, "length"))
    state: _Rsi = interp.state_for(node, lambda: _Rsi(length))
    return state.update(src)


def _ta_atr(interp, node):
    args, kwargs = _eval_all(interp, node)
    length = int(_arg(args, kwargs, 0, "length"))

    def factory():
        return {"rma": _SeededEma(length, 1.0 / length), "prev": _PrevClose()}

    state = interp.state_for(node, factory)
    high, low, close = interp._ctx["high"], interp._ctx["low"], interp._ctx["close"]
    prev = state["prev"].prev
    if is_na(prev):
        tr = high - low
    else:
        tr = max(high - low, abs(high - prev), abs(low - prev))
    state["prev"].prev = close
    return state["rma"].update(tr)


def _ta_change(interp, node):
    args, kwargs = _eval_all(interp, node)
    src = _arg(args, kwargs, 0, "source")
    length = int(_arg(args, kwargs, 1, "length", 1))
    win: _Window = interp.state_for(node, lambda: _Window(length + 1))
    win.push(src)
    if not win.full():
        return NA
    arr = win.array()
    if math.isnan(arr[0]) or math.isnan(arr[-1]):
        return NA
    return float(arr[-1] - arr[0])


def _ta_crossover(interp, node):
    args, kwargs = _eval_all(interp, node)
    state: _Cross = interp.state_for(node, _Cross)
    return state.update(_arg(args, kwargs, 0, "source1"), _arg(args, kwargs, 1, "source2"), True)


def _ta_crossunder(interp, node):
    args, kwargs = _eval_all(interp, node)
    state: _Cross = interp.state_for(node, _Cross)
    return state.update(_arg(args, kwargs, 0, "source1"), _arg(args, kwargs, 1, "source2"), False)


def _ta_barssince(interp, node):
    args, kwargs = _eval_all(interp, node)
    state: _BarsSince = interp.state_for(node, _BarsSince)
    return state.update(_arg(args, kwargs, 0, "condition"))


# ── math.* / misc scalar ──────────────────────────────────────────────────────


def _scalar(fn):
    def handler(interp, node):
        args, kwargs = _eval_all(interp, node)
        if any(is_na(a) for a in args):
            return NA
        return fn(*args, **kwargs)

    return handler


def _nz(interp, node):
    args, kwargs = _eval_all(interp, node)
    value = _arg(args, kwargs, 0, "source")
    replacement = _arg(args, kwargs, 1, "replacement", 0.0)
    return replacement if is_na(value) else value


def _na_fn(interp, node):
    args, _ = _eval_all(interp, node)
    return is_na(args[0]) if args else True


def _str_tostring(interp, node):
    args, kwargs = _eval_all(interp, node)
    value = _arg(args, kwargs, 0, "value")
    fmt = _arg(args, kwargs, 1, "format", "")
    if is_na(value):
        return "NaN"
    if isinstance(fmt, str) and fmt:
        decimals = len(fmt.split(".", 1)[1]) if "." in fmt else 0
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ── array.* ───────────────────────────────────────────────────────────────────


def _array_new(interp, node):
    return PineArray()


def _array_method(fn):
    def handler(interp, node):
        args, kwargs = _eval_all(interp, node)
        arr = _arg(args, kwargs, 0, "id")
        if not isinstance(arr, PineArray):
            raise PineRuntimeError(f"{node.func}: first argument must be an array", node.line)
        return fn(arr, args[1:], kwargs)

    return handler


_ARRAY_HANDLERS = {
    "array.push": _array_method(lambda a, rest, kw: a.items.append(rest[0])),
    "array.shift": _array_method(lambda a, rest, kw: a.items.pop(0) if a.items else NA),
    "array.pop": _array_method(lambda a, rest, kw: a.items.pop() if a.items else NA),
    "array.size": _array_method(lambda a, rest, kw: float(len(a.items))),
    "array.avg": _array_method(
        lambda a, rest, kw: float(np.sum(np.asarray(a.items, dtype=float)) / len(a.items))
        if a.items
        else NA
    ),
    "array.sum": _array_method(
        lambda a, rest, kw: float(np.sum(np.asarray(a.items, dtype=float))) if a.items else NA
    ),
    "array.get": _array_method(lambda a, rest, kw: a.items[int(rest[0])]),
    "array.clear": _array_method(lambda a, rest, kw: a.items.clear()),
}


# ── input.* ───────────────────────────────────────────────────────────────────

_INPUT_CASTS = {
    "input": lambda v: v,
    "input.int": lambda v: int(v),
    "input.float": lambda v: float(v),
    "input.bool": lambda v: bool(v),
    "input.string": lambda v: str(v),
    "input.timeframe": lambda v: str(v),
    "input.color": lambda v: v,
    "input.session": lambda v: str(v),
}


def _input(interp: Interpreter, node: Call):
    if node.func == "input.source":
        # live series: re-evaluate the source argument every bar
        title = _input_title(interp, node)
        if title is not None and title in interp.input_overrides:
            interp.matched_overrides.add(title)
            name = str(interp.input_overrides[title])
            if name not in interp._ctx_hist:
                raise PineRuntimeError(
                    f"input.source override must be one of {sorted(interp._ctx_hist)}", node.line
                )
            interp.result.inputs[title] = name
            return interp._ctx[name]
        if node.args:
            value = interp.eval(node.args[0])
            if title is not None:
                interp.result.inputs.setdefault(title, "close")
            return value
        return interp._ctx["close"]

    if node.node_id in interp.input_cache:
        return interp.input_cache[node.node_id]

    args, kwargs = _eval_all(interp, node)
    default = _arg(args, kwargs, 0, "defval")
    title = _arg(args, kwargs, 1, "title")
    cast = _INPUT_CASTS.get(node.func)
    if cast is None:
        raise PineUnsupportedError(f"{node.func}() is not supported", node.line)
    value = default
    if isinstance(title, str) and title in interp.input_overrides:
        interp.matched_overrides.add(title)
        value = interp.input_overrides[title]
    if not isinstance(value, Opaque) and value is not None and not is_na(value):
        value = cast(value)
    if isinstance(title, str):
        interp.result.inputs[title] = value if not isinstance(value, Opaque) else str(value)
    interp.input_cache[node.node_id] = value
    return value


def _input_title(interp: Interpreter, node: Call):
    title_node = node.kwargs.get("title")
    if title_node is None and len(node.args) > 1:
        title_node = node.args[1]
    return interp.eval(title_node) if title_node is not None else None


# ── declaration / plots / alerts ─────────────────────────────────────────────


def _indicator(interp: Interpreter, node: Call):
    args, kwargs = _eval_all(interp, node)
    interp.result.title = str(_arg(args, kwargs, 0, "title", "") or "")
    interp.result.shorttitle = str(_arg(args, kwargs, 1, "shorttitle", "") or "")
    return None


def _record(interp: Interpreter, node: Call, name: str, value: Any) -> None:
    # one column per plot/alert call site, allocated once (dedup happens there)
    key = ("column", node.node_id)
    column = interp.ta_state.get(key)
    if column is None:
        column = interp.result.column(name)
        interp.ta_state[key] = column
    column[interp.i] = value


def _coerce_plot_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if is_na(value):
        return NA
    if isinstance(value, (int, float)):
        return float(value)
    return NA  # colors / strings are not plottable series


def _plot(interp: Interpreter, node: Call):
    series_node = node.kwargs.get("series", node.args[0] if node.args else None)
    if series_node is None:
        raise PineRuntimeError("plot() needs a series", node.line)
    value = interp.eval(series_node)
    title = _plot_title(interp, node)
    _eval_rest(interp, node, skip={"series", "title"})
    _record(interp, node, title, _coerce_plot_value(value))
    return Opaque("plot", title)


def _plot_title(interp: Interpreter, node: Call) -> str:
    title_node = node.kwargs.get("title")
    if title_node is None and len(node.args) > 1:
        title_node = node.args[1]
    if title_node is not None:
        raw = interp.eval(title_node)
        if isinstance(raw, str) and raw.strip():
            return slugify(raw)
    interp._plot_counter += 1
    return f"plot_{interp._plot_counter}"


def _eval_rest(interp: Interpreter, node: Call, skip: set[str]) -> None:
    for pos, arg in enumerate(node.args):
        if pos == 0 or (pos == 1 and "title" in skip):
            continue
        interp.eval(arg)
    for key, value in node.kwargs.items():
        if key not in skip:
            interp.eval(value)


def _plotmark(interp: Interpreter, node: Call):
    # plotchar / plotshape: first arg is a bool condition or a value series;
    # title is kwarg or positional arg 1, same slot as plot()
    value = interp.eval(node.args[0]) if node.args else NA
    title = _plot_title(interp, node)
    _eval_rest(interp, node, skip={"title"})
    if isinstance(value, bool):
        _record(interp, node, title, value)
    else:
        _record(interp, node, title, _coerce_plot_value(value))
    return None


def _alertcondition(interp: Interpreter, node: Call):
    args, kwargs = _eval_all(interp, node)
    cond = _arg(args, kwargs, 0, "condition")
    title = _arg(args, kwargs, 1, "title", "")
    name = slugify(str(title)) if title else None
    if name is None:
        interp._plot_counter += 1
        name = f"alert_{interp._plot_counter}"
    _record(interp, node, name, bool(truthy(cond)))
    return None


def _noop(interp: Interpreter, node: Call):
    _eval_all(interp, node)
    return Opaque(node.func)


# ── request.security ─────────────────────────────────────────────────────────


def _security(interp: Interpreter, node: Call):
    if len(node.args) < 3:
        raise PineRuntimeError("request.security(symbol, timeframe, expression)", node.line)
    tf = interp.eval(node.args[1])
    if not isinstance(tf, str):
        raise PineUnsupportedError("request.security timeframe must be a string", node.line)
    expr_node = node.args[2]
    exprs = expr_node.items if isinstance(expr_node, TupleExpr) else [expr_node]
    is_tuple = isinstance(expr_node, TupleExpr)

    lookahead_on = False
    la = node.kwargs.get("lookahead")
    if la is not None and isinstance(la, Ident):
        lookahead_on = la.name == "barmerge.lookahead_on"

    if interp.subrun or tf.strip() == "":
        values = tuple(interp.eval(e) for e in exprs)
        return values if is_tuple else values[0]

    cached = interp.security_cache.get(node.node_id)
    if cached is None:
        cached = _run_security_subrun(interp, tf, exprs)
        interp.security_cache[node.node_id] = cached
    starts, captured = cached

    bar_time = interp.bars.index[interp.i]
    bucket = int(np.searchsorted(starts, bar_time.value, side="right")) - 1
    idx = bucket if lookahead_on else bucket - 1
    if idx < 0 or bucket < 0:
        values = tuple(NA for _ in exprs)
    else:
        values = tuple(captured[k][idx] for k in range(len(exprs)))
    return values if is_tuple else values[0]


def _run_security_subrun(interp: Interpreter, tf: str, exprs: list[Node]):
    htf_bars = interp.resample_to(tf)
    sub = Interpreter(
        interp.script, interp.input_overrides, subrun=True, symbol=interp.symbol
    )
    from . import builtins as bi

    sub.builtins = bi
    captured: list[list[Any]] = [[] for _ in exprs]

    # run the sub-interpreter manually so we can capture after each bar
    sub.bars = htf_bars
    sub.n = len(htf_bars)
    from .runtime import RunResult

    sub.result = RunResult(sub.n)
    opens = htf_bars["open"].to_numpy(dtype=float)
    highs = htf_bars["high"].to_numpy(dtype=float)
    lows = htf_bars["low"].to_numpy(dtype=float)
    closes = htf_bars["close"].to_numpy(dtype=float)
    volumes = htf_bars["volume"].to_numpy(dtype=float)
    hlc3 = (highs + lows + closes) / 3.0
    is_rth = (htf_bars["session"] == "rth").to_numpy()
    sub._ctx_hist = {
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "hlc3": hlc3,
        "hl2": (highs + lows) / 2.0,
        "ohlc4": (opens + highs + lows + closes) / 4.0,
    }
    for i in range(sub.n):
        sub.i = i
        sub._ctx = {
            "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i],
            "volume": volumes[i], "hlc3": hlc3[i],
            "hl2": sub._ctx_hist["hl2"][i], "ohlc4": sub._ctx_hist["ohlc4"][i],
            "bar_index": float(i),
            "session.ismarket": bool(is_rth[i]),
            "barstate.isconfirmed": True,
            "barstate.islast": i == sub.n - 1,
            "barstate.isfirst": i == 0,
            "syminfo.tickerid": sub.symbol,
            "syminfo.ticker": sub.symbol,
        }
        for slot in sub.globals.values():
            if slot.is_var and slot.initialized and i > 0:
                slot.values[i] = slot.values[i - 1]
        for stmt in sub.script.statements:
            sub.exec_stmt(stmt)
        for k, expr in enumerate(exprs):
            captured[k].append(sub.eval(expr))

    # bucket starts as ns epoch — index resolution varies (us vs ns), while
    # Timestamp.value below is always ns, so normalize explicitly
    starts = htf_bars.index.as_unit("ns").asi8
    return starts, captured


# ── handler table ─────────────────────────────────────────────────────────────

_HANDLERS = {
    "indicator": _indicator,
    "strategy": _indicator,
    "plot": _plot,
    "plotchar": _plotmark,
    "plotshape": _plotmark,
    "plotarrow": _plotmark,
    "alertcondition": _alertcondition,
    "fill": _noop,
    "hline": _noop,
    "bgcolor": _noop,
    "barcolor": _noop,
    "alert": _noop,
    "ta.sma": _ta_sma,
    "ta.ema": _ta_ema,
    "ta.rma": _ta_rma,
    "ta.stdev": _ta_stdev,
    "ta.median": _ta_median,
    "ta.highest": _ta_highest,
    "ta.lowest": _ta_lowest,
    "ta.rsi": _ta_rsi,
    "ta.atr": _ta_atr,
    "ta.change": _ta_change,
    "ta.mom": _ta_change,
    "ta.percentrank": _ta_percentrank,
    "ta.crossover": _ta_crossover,
    "ta.crossunder": _ta_crossunder,
    "ta.barssince": _ta_barssince,
    "math.sum": _math_sum,
    "math.abs": _scalar(lambda x: abs(x)),
    "math.max": _scalar(lambda *xs: max(xs)),
    "math.min": _scalar(lambda *xs: min(xs)),
    "math.exp": _scalar(lambda x: math.exp(min(x, 700.0))),
    "math.log": _scalar(lambda x: math.log(x) if x > 0 else NA),
    "math.sqrt": _scalar(lambda x: math.sqrt(x) if x >= 0 else NA),
    "math.pow": _scalar(lambda x, y: x**y),
    "math.floor": _scalar(lambda x: float(math.floor(x))),
    "math.ceil": _scalar(lambda x: float(math.ceil(x))),
    "math.round": _scalar(lambda x, precision=0: float(round(x, int(precision)))),
    "math.sign": _scalar(lambda x: float((x > 0) - (x < 0))),
    "math.avg": _scalar(lambda *xs: sum(xs) / len(xs)),
    "math.tanh": _scalar(math.tanh),
    "nz": _nz,
    "na": _na_fn,
    "str.tostring": _str_tostring,
    "str.format": _scalar(lambda fmt, *xs: fmt),
    "array.new": _array_new,
    "request.security": _security,
    "input": _input,
    "input.int": _input,
    "input.float": _input,
    "input.bool": _input,
    "input.string": _input,
    "input.timeframe": _input,
    "input.color": _input,
    "input.source": _input,
    "input.session": _input,
    **_ARRAY_HANDLERS,
}
