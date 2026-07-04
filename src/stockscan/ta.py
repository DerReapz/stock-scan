"""Pine-parity technical-analysis primitives.

Every function mirrors the semantics of the corresponding Pine v6 `ta.*`
built-in, including warmup NaNs and seeding rules. The indicator engines are
straight ports of Pine scripts, so parity here is what makes their outputs
trustworthy.

Conventions shared with Pine:
- Rolling windows return NaN until a full window is available.
- `ta.stdev` is population standard deviation (ddof=0).
- `ta.ema` / RMA seed with the SMA of the first `n` values, then recurse.
- `ta.percentrank` counts how many of the previous `n` values are less than
  or equal to the current value, as a percentage 0-100.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def stdev_pop(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).std(ddof=0)


def rolling_sum(s: pd.Series, n: int) -> pd.Series:
    """Pine math.sum: sliding-window sum, NaN until the window fills.

    Computed as an exact per-window sum (not pandas' incremental algorithm) so
    that equal windows produce bitwise-equal sums — percentrank tie-counting
    depends on it, and the Pine interpreter sums windows the same way.
    """
    values = s.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if n >= 1 and len(values) >= n:
        out[n - 1 :] = np.lib.stride_tricks.sliding_window_view(values, n).sum(axis=1)
    return pd.Series(out, index=s.index)


def rolling_median(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).median()


def _seeded_recursive(s: pd.Series, n: int, alpha: float) -> pd.Series:
    """Pine's EMA/RMA recursion: NaN until the first full SMA window over the
    valid (non-NaN) part of the series, seeded with that SMA, then
    ``alpha * x + (1 - alpha) * prev``.
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if len(valid) < n or n < 1:
        return out
    values = valid.to_numpy(dtype=float)
    result = np.empty(len(values))
    result[: n - 1] = np.nan
    prev = values[:n].mean()
    result[n - 1] = prev
    for i in range(n, len(values)):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        result[i] = prev
    out.loc[valid.index] = result
    return out


def pine_ema(s: pd.Series, n: int) -> pd.Series:
    if n == 1:
        return s.astype(float)
    return _seeded_recursive(s, n, 2.0 / (n + 1.0))


def rma(s: pd.Series, n: int) -> pd.Series:
    if n == 1:
        return s.astype(float)
    return _seeded_recursive(s, n, 1.0 / n)


def rsi(s: pd.Series, n: int) -> pd.Series:
    change = s.diff()
    gain = change.clip(lower=0.0)
    loss = (-change).clip(lower=0.0)
    # diff() puts NaN on the first bar; Pine treats the first change as absent
    # too, so drop it from the seed window rather than counting it as zero.
    avg_gain = rma(gain[1:], n).reindex(s.index)
    avg_loss = rma(loss[1:], n).reindex(s.index)
    # Pine reference: down == 0 ? 100 : up == 0 ? 0 : 100 - 100 / (1 + up/down)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    out = out.where(avg_loss != 0.0, 100.0)
    return out.where(~(avg_gain.isna() | avg_loss.isna()))


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    hi = df[["high"]].copy()["high"]
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = hi.iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return rma(true_range(df), n)


def hlc3(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def percentrank(s: pd.Series, n: int) -> pd.Series:
    """Percentage of the previous ``n`` values less than or equal to the
    current value (Pine reference-manual convention)."""
    values = s.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) > n:
        windows = np.lib.stride_tricks.sliding_window_view(values, n)[:-1]
        current = values[n:]
        counts = (windows <= current[:, None]).sum(axis=1)
        out[n:] = counts * 100.0 / n
        # NaN in the window or current value → NaN result (Pine na propagation)
        bad = np.isnan(windows).any(axis=1) | np.isnan(current)
        out[n:][bad] = np.nan
    return pd.Series(out, index=s.index)


def crossover(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    now = a > b
    before = a.shift() <= b.shift()
    valid = a.notna() & b.notna() & a.shift().notna() & b.shift().notna()
    return (now & before & valid).fillna(False)


def crossunder(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    now = a < b
    before = a.shift() >= b.shift()
    valid = a.notna() & b.notna() & a.shift().notna() & b.shift().notna()
    return (now & before & valid).fillna(False)


def barssince(cond: pd.Series) -> pd.Series:
    """Bars since ``cond`` was last True; NaN before the first occurrence."""
    truthy = cond.fillna(False).to_numpy(dtype=bool)
    idx = np.arange(len(truthy), dtype=float)
    last_true = np.where(truthy, idx, np.nan)
    last_true = pd.Series(last_true).ffill().to_numpy()
    out = idx - last_true
    return pd.Series(out, index=cond.index)
