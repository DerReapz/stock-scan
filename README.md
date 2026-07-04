# stockscan

Intraday stock scanner that runs three TradingView indicators — **Iron
Momentum**, **Silver Flow**, and **Gold Regime** — over a pluggable market-data
feed (live/paid or free/delayed), from the terminal or a local web dashboard.
You can also feed **your own PineScript files** into the scanner: a built-in
Pine v6 interpreter turns any supported `.pine` indicator into scan columns.

```
uv run stockscan scan watchlists/default.txt --timeframe 5m \
    --filter "mb_state >= 2 and ggr_state >= 1" --sort mb_score
```

## Quick start

```bash
# install (Python >= 3.11; uv recommended, plain pip works too)
uv venv && uv pip install -e ".[dev]"

# scan with the zero-key delayed feed (yfinance)
uv run stockscan scan AAPL,MSFT,NVDA --timeframe 15m

# launch the auto-refreshing dashboard on http://127.0.0.1:8501
uv run stockscan serve watchlists/default.txt --timeframe 5m --interval 60

# see feed options and whether API keys are configured
uv run stockscan providers
```

A watchlist is a text file with one symbol per line (`#` comments), or a
comma-separated list passed directly on the command line.

## Data feeds: free (delayed) vs live (paid)

The provider layer is pluggable; all three ship enabled and return identical,
canonical bars (UTC timestamps, RTH/pre/post session tags, **completed bars
only** — every signal is computed on closed bars and never repaints).

| provider | class | notes |
|---|---|---|
| `yfinance` (default) | free, ~15 min delayed | no API key; intraday history capped (1m ≈ 7d, 5m/15m ≈ 60d) |
| `alpaca` | free **IEX** real-time / paid **SIP** | one API for both: set `feed = "iex"` (free keys) or `feed = "sip"` (paid subscription); batched + paginated |
| `polygon` | free 15-min delayed / paid real-time | free tier is 5 req/min (rate-limited automatically, fresh bars trimmed); set `tier = "paid"` to lift both |

Keys go in `.env` (copy `.env.example`): `ALPACA_KEY_ID` / `ALPACA_SECRET`,
`POLYGON_API_KEY`. Select the feed per scan with `--provider alpaca` or in
`stockscan.toml`.

## The indicator engines

All engine parameters mirror the original Pine inputs and are configurable in
`stockscan.toml` (see `stockscan.example.toml` for every knob).

**Iron Momentum (`mb_*`)** — momentum score hard-bounded to ±50 by a tanh
squash (velocity or position engine vs a 50-bar benchmark), EMA-smoothed.
Exports the score, a 5-tier state ladder (`mb_state`: +2 strong bull … −2
strong bear), strength %, relative volume (`mb_rvol`, optionally computed from
regular-session bars only), volume-surge star events (`mb_star`: ±1), and
zero/strong-level crosses.

**Silver Flow (`sf_*`)** — partitions each bar's dollar volume into
institutional and retail buckets by percentile (or z-score) size gating, with
a soft-gate ramp between thresholds. Exports the smoothed share lines
(`sf_inst`, `sf_retail`), dominance regime and flips, signed flow biases,
quiet-tape accumulation/distribution events (`sf_accum`, `sf_dist`), and
active-tape confluence (`sf_confluence`: ±1).

**Gold Regime (`ggr_*`)** — 8/20 SMA ribbon + RSI band state machine. Exports
the −3…+3 state ladder (direction, +RSI agreement, +RSI thrust; zeroed by an
ATR-normalized width gate), ribbon width, bars since the last regime flip, and
cross/strong-entry/thrust-entry events. The RSI leg can run on a higher
timeframe (`rsi_htf = "1h"`), using closed HTF bars only.

## Filtering, sorting, exporting

Every exported metric is a column of the results table and usable in filters:

```bash
stockscan scan watchlists/default.txt \
    --filter "mb_state >= 1 and mb_rvol >= 2 and sf_regime == -1" \
    --sort ggr_state --limit 20 --json out.json --csv out.csv
```

Filters accept comparison operators plus `and` / `or` / `not`, and only column
names — no arbitrary code. The dashboard has the same filter box, applied
server-side against the cached scan (no extra API calls).

## Bring your own PineScript

```bash
stockscan scan AAPL,MSFT --pine my_indicator.pine \
    --pine-input "Sensitivity=1.5"
```

The interpreter executes a substantial Pine v6 subset with TradingView
semantics: per-bar execution, `var` persistence, series history (`x[n]`),
per-call-site `ta.*` state, `if`/user functions/arrays/tuple destructuring,
and non-repainting `request.security` higher-timeframe requests. Every titled
`plot()` becomes a scan column (title slugified, prefixed by the script's
short title); `alertcondition`, `plotchar`, and `plotshape` become signal
columns. Purely visual calls (`fill`, `hline`, `table.*`, colors) parse and
no-op. Anything outside the subset fails loudly with
`PineUnsupportedError: line N: ...` — never a silently wrong number.

Inputs use their script defaults, overridable by input title via
`--pine-input "Title=value"` or per script in TOML:

```toml
[pine."indicators/iron_momentum.pine"]
"Sensitivity" = 1.5
"RVOL trigger (x average)" = 2.5
```

The three source indicators ship in `indicators/` and double as the
interpreter's conformance suite: interpreted output must match the native
engines to 1e-9.

## Configuration

Precedence: CLI flags > environment variables > `stockscan.toml` > defaults.
Copy `stockscan.example.toml` to `stockscan.toml` to change engine parameters,
default timeframe/provider/filter, or Alpaca/Polygon feed class.

## Standalone executable (.exe)

The scanner compiles to a single self-contained executable — no Python
required on the target machine:

```bash
pip install -e ".[build]"        # or: uv pip install -e ".[build]"
pyinstaller stockscan.spec       # → dist/stockscan  (dist/stockscan.exe on Windows)
```

PyInstaller builds for the platform it runs on: run the build on Windows to
get `stockscan.exe` (Linux/macOS produce native binaries the same way). The
executable embeds the dashboard front end, so `stockscan serve` works from
the single file; `.env`, `stockscan.toml`, watchlists, and `--pine` scripts
are read from the directory you run it in, exactly like the pip install.

**Windows note:** `stockscan.exe` is a command-line tool — open PowerShell or
cmd in the folder containing it and run `stockscan scan ...` / `stockscan
serve ...`. Double-clicking it in Explorer shows a quickstart and keeps the
window open (it used to flash and close), but real use happens from a
terminal or a shortcut with arguments, e.g.
`stockscan.exe serve watchlist.txt`.

CI does this automatically: the `build` workflow
(`.github/workflows/build.yml`) runs tests on every push/PR, and on version
tags (`v*`) or manual dispatch it builds Windows/Linux/macOS executables,
uploads them as artifacts, and attaches them to the GitHub release.

## Development

```bash
uv run pytest        # 100+ tests: golden TA values, engine-vs-Pine-reference
                     # loops, interpreter conformance, mocked providers
uv run ruff check src tests
```

Layout: `src/stockscan/ta.py` (Pine-parity primitives — population stdev,
SMA-seeded EMA/RMA, Pine percentrank, …), `engines/` (native ports),
`pine/` (lexer → parser → bar-loop runtime), `providers/`, `scan.py`,
`web/` (FastAPI + single-file dashboard), `cli.py`.

### Known limitations (by design, for now)

- Market-calendar precision: half-days are tagged by wall clock only.
- yfinance is unofficial and rate-limits aggressively on big watchlists.
- Pine subset: `for`/`while`/`switch`/`type`, labels/lines/boxes, and
  `request.*` other than same-symbol `security` are unsupported (they raise).
- Websocket streaming (true tick-level live) is planned; REST polling of
  completed bars is the current model for both feed classes.
