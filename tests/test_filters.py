import pandas as pd
import pytest

from stockscan.filters import FilterError, apply_filter


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mb_state": [2, 1, 0, -1, -2],
            "ggr_state": [3, 1, 0, -2, -3],
            "mb_rvol": [2.5, 1.0, 0.5, 3.0, 1.1],
        },
        index=pd.Index(["A", "B", "C", "D", "E"], name="symbol"),
    )


def test_combined_expression(frame):
    out = apply_filter(frame, "mb_state >= 1 and ggr_state >= 1")
    assert list(out.index) == ["A", "B"]


def test_empty_expression_is_identity(frame):
    assert apply_filter(frame, "  ") is frame


def test_unknown_column_lists_available(frame):
    with pytest.raises(FilterError, match="unknown column"):
        apply_filter(frame, "bogus > 1")


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "mb_state.__class__",
        "@pd",
        "mb_state.abs()",
        "`weird col` > 0",
    ],
)
def test_injection_attempts_rejected(frame, expr):
    with pytest.raises(FilterError):
        apply_filter(frame, expr)


def test_bear_side(frame):
    out = apply_filter(frame, "mb_state <= -1 and mb_rvol >= 2")
    assert list(out.index) == ["D"]
