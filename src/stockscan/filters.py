"""Safe filter expressions over the scan results frame.

Grammar is whatever ``DataFrame.query`` accepts, restricted to identifiers
that are actual result columns plus boolean keywords — no attribute access,
no calls, no dunders. Example: ``mb_state >= 2 and ggr_state >= 1``.
"""

from __future__ import annotations

import re

import pandas as pd

_ALLOWED_KEYWORDS = {"and", "or", "not", "in", "True", "False", "abs"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FORBIDDEN = re.compile(r"(__|@|`|\.[A-Za-z_])")


class FilterError(ValueError):
    pass


def apply_filter(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    expr = expr.strip()
    if not expr:
        return df
    if _FORBIDDEN.search(expr):
        raise FilterError("filter may only reference result columns and comparison operators")
    unknown = [
        name
        for name in set(_IDENT.findall(expr))
        if name not in _ALLOWED_KEYWORDS and name not in df.columns
    ]
    if unknown:
        raise FilterError(
            f"unknown column(s) in filter: {sorted(unknown)}. "
            f"Available: {', '.join(sorted(df.columns))}"
        )
    try:
        return df.query(expr, engine="python", local_dict={}, global_dict={})
    except Exception as exc:
        raise FilterError(f"bad filter expression: {exc}") from exc
