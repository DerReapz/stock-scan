"""Pine interpreter: unit tests + conformance against the native engines.

The three bundled .pine files are interpreted over synthetic fixtures and
must reproduce the native vectorized engines' series to 1e-9 (after warmup,
where both are defined).
"""

import numpy as np
import pandas as pd
import pytest

from stockscan.engines.gold_regime import GoldRegimeConfig
from stockscan.engines.gold_regime import compute_series as ggr_series
from stockscan.engines.iron_momentum import IronMomentumConfig
from stockscan.engines.iron_momentum import compute_series as mb_series
from stockscan.engines.silver_flow import SilverFlowConfig
from stockscan.engines.silver_flow import compute_series as sf_series
from stockscan.pine import PineEngine, PineUnsupportedError, parse
from stockscan.pine.runtime import Interpreter

from .conftest import make_bars


def run_pine(source: str, bars, overrides=None) -> pd.DataFrame:
    interp = Interpreter(parse(source), overrides or {})
    return interp.run(bars).to_frame(bars.index)


class TestBasics:
    def test_series_and_history(self, bars_rth):
        df = run_pine(
            """//@version=6
indicator("t", "t")
x = close - close[1]
plot(x, "diff")
""",
            bars_rth,
        )
        expected = bars_rth["close"].diff()
        np.testing.assert_allclose(df["diff"][1:], expected[1:], rtol=1e-12)
        assert np.isnan(df["diff"].iloc[0])

    def test_var_persistence_and_reassign(self, bars_rth):
        df = run_pine(
            """//@version=6
indicator("t", "t")
var float acc = 0.0
acc := acc + 1.0
plot(acc, "count")
""",
            bars_rth,
        )
        assert df["count"].iloc[-1] == len(bars_rth)

    def test_if_block_and_ladder(self, bars_rth):
        df = run_pine(
            """//@version=6
indicator("t", "t")
int state = 0
if close > open
    state := 1
else
    state := -1
plot(state, "state")
""",
            bars_rth,
        )
        expected = np.where(bars_rth["close"] > bars_rth["open"], 1, -1)
        np.testing.assert_array_equal(df["state"].to_numpy(), expected)

    def test_user_function(self, bars_rth):
        df = run_pine(
            """//@version=6
indicator("t", "t")
f_double(float x) =>
    y = x * 2.0
    y
plot(f_double(close), "doubled")
""",
            bars_rth,
        )
        np.testing.assert_allclose(df["doubled"], bars_rth["close"] * 2)

    def test_input_override(self, bars_rth):
        src = """//@version=6
indicator("t", "t")
len = input.int(5, "Length")
plot(ta.sma(close, len), "ma")
"""
        base = run_pine(src, bars_rth)
        overridden = run_pine(src, bars_rth, {"Length": 20})
        assert np.isnan(overridden["ma"].iloc[10])
        assert not np.isnan(base["ma"].iloc[10])

    def test_unknown_override_rejected(self, bars_rth):
        with pytest.raises(Exception, match="matched no input"):
            run_pine(
                """//@version=6
indicator("t", "t")
len = input.int(5, "Length")
plot(ta.sma(close, len), "ma")
""",
                bars_rth,
                {"Lenght": 20},
            )

    def test_unsupported_construct_reports_line(self):
        with pytest.raises(PineUnsupportedError, match="line 3.*for"):
            parse(
                """//@version=6
indicator("t", "t")
for i = 0 to 10
    x = i
"""
            )

    def test_ternary_na_condition_takes_false_branch(self, bars_rth):
        df = run_pine(
            """//@version=6
indicator("t", "t")
x = na
y = x > 0 ? 1.0 : 2.0
plot(y, "y")
""",
            bars_rth,
        )
        assert (df["y"] == 2.0).all()


@pytest.fixture(scope="module")
def fixture_bars():
    return make_bars(320, include_extended=True, trend=0.05, seed=11)


def assert_series_match(pine: pd.Series, native: pd.Series, start: int, name: str):
    p = pine.to_numpy(dtype=float)[start:]
    n = native.to_numpy(dtype=float)[start:]
    both_nan = np.isnan(p) & np.isnan(n)
    np.testing.assert_allclose(
        np.where(both_nan, 0.0, p),
        np.where(both_nan, 0.0, n),
        rtol=1e-9,
        atol=1e-9,
        err_msg=name,
    )


class TestIronMomentumConformance:
    def test_matches_native(self, fixture_bars):
        engine = PineEngine.from_file("indicators/iron_momentum.pine")
        pine = engine.run_series(fixture_bars, symbol="TEST")
        native = mb_series(fixture_bars, IronMomentumConfig())
        start = 80
        assert_series_match(pine["mb_score"], native["score"], start, "score")
        assert_series_match(pine["mb_state"], native["state"], start, "state")
        assert_series_match(pine["mb_strength"], native["strength_pct"], start, "strength")
        assert_series_match(pine["mb_raw_ratio"], native["raw"], start, "raw")
        assert_series_match(pine["mb_rvol"], native["rvol"], start, "rvol")
        assert_series_match(pine["mb_star"], native["star"], start, "star")

    def test_position_mode_via_override(self, fixture_bars):
        engine = PineEngine.from_file(
            "indicators/iron_momentum.pine",
            input_overrides={"Benchmark mode": "Position vs 50 SMA"},
        )
        pine = engine.run_series(fixture_bars, symbol="TEST")
        native = mb_series(fixture_bars, IronMomentumConfig(mode="position"))
        assert_series_match(pine["mb_score"], native["score"], 80, "score")


class TestSilverFlowConformance:
    def test_matches_native(self, fixture_bars):
        engine = PineEngine.from_file("indicators/silver_flow.pine")
        pine = engine.run_series(fixture_bars, symbol="TEST")
        native = sf_series(fixture_bars, SilverFlowConfig())
        start = 130
        assert_series_match(pine["retail_share_white"], native["retail"], start, "retail")
        assert_series_match(
            pine["institutional_share_gold"], native["inst"], start, "inst"
        )
        assert_series_match(pine["dbg_regime"], native["regime"], start, "regime")
        assert_series_match(pine["dbg_instbias"], native["inst_bias"], start, "inst_bias")
        assert_series_match(
            pine["dbg_retailbias"], native["retail_bias"], start, "retail_bias"
        )
        assert_series_match(pine["dbg_medvolpct"], native["tape_pct"], start, "tape_pct")
        # events (edge-only defaults on both sides)
        np.testing.assert_array_equal(
            pine["institutional_accumulation"].notna().to_numpy()[start:],
            native["accum"].to_numpy()[start:],
            err_msg="accum",
        )
        np.testing.assert_array_equal(
            pine["institutional_distribution"].notna().to_numpy()[start:],
            native["dist"].to_numpy()[start:],
            err_msg="dist",
        )
        conf = np.where(
            pine["bullish_confluence"].notna(), 1, np.where(pine["bearish_confluence"].notna(), -1, 0)
        )
        np.testing.assert_array_equal(
            conf[start:], native["confluence"].to_numpy()[start:], err_msg="confluence"
        )

    def test_signed_mode_via_override(self, fixture_bars):
        engine = PineEngine.from_file(
            "indicators/silver_flow.pine",
            input_overrides={"Scale mode": "Signed (-100 to +100)"},
        )
        pine = engine.run_series(fixture_bars, symbol="TEST")
        native = sf_series(fixture_bars, SilverFlowConfig(scale_mode="signed"))
        assert_series_match(pine["retail_share_white"], native["retail"], 130, "retail signed")


class TestGoldRegimeConformance:
    def test_matches_native(self, fixture_bars):
        engine = PineEngine.from_file("indicators/gold_regime.pine")
        pine = engine.run_series(fixture_bars, symbol="TEST")
        native = ggr_series(fixture_bars, GoldRegimeConfig())
        start = 60
        assert_series_match(pine["ribbon_state_3_3"], native["state"], start, "state")
        assert_series_match(
            pine["ribbon_width_atr_mult_signed"], native["width_atr"], start, "width"
        )
        assert_series_match(pine["rsi_raw"], native["rsi"], start, "rsi")
        assert_series_match(
            pine["rsi_band_width"], native["rsi_band_width"], start, "band width"
        )
        assert_series_match(
            pine["ma_rsi_agreement"], native["ma_rsi_agree"], start, "agreement"
        )
        # bars-since-flip agrees from the first post-warmup reset onward
        native_bsf = native["bars_since_flip"].to_numpy()
        resets = np.where(native_bsf[start:] == 0)[0]
        assert len(resets) > 0
        s = start + int(resets[0])
        assert_series_match(
            pine["bars_since_flip_confirmed"], native["bars_since_flip"], s, "bars since flip"
        )
        np.testing.assert_array_equal(
            pine["ribbon_bull_cross"].to_numpy()[start:].astype(bool),
            native["cross_up"].to_numpy()[start:],
            err_msg="bull cross",
        )

    def test_htf_rsi_matches_native(self):
        from stockscan.models import Timeframe

        bars = make_bars(400, seed=9)
        engine = PineEngine.from_file(
            "indicators/gold_regime.pine", input_overrides={"RSI Timeframe": "60"}
        )
        pine = engine.run_series(bars, symbol="TEST")
        native = ggr_series(bars, GoldRegimeConfig(rsi_htf=Timeframe.H1))
        # hourly RSI band needs ~35 hourly bars ≈ 274 5m RTH bars of warmup
        start = 300
        assert not np.isnan(native["rsi"].to_numpy()[start:]).all()
        assert_series_match(pine["rsi_raw"], native["rsi"], start, "htf rsi")
        assert_series_match(pine["ribbon_state_3_3"], native["state"], start, "htf state")
