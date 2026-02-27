# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --all-extras

# Run a pipeline
uv run python -m src.run_pipeline --pipeline=data_engineering
uv run python -m src.run_pipeline --pipeline=feature_engineering

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_data_pipeline.py -v              # single file
uv run pytest tests/test_data_pipeline.py::TestCleanOhlcv -v  # single class

# Lint
uv run ruff check src/ tests/
```

## Architecture

**DRL crypto trading bot** using a 5-stage Parquet pipeline → Gymnasium environment → TD3 agent.

**Entry point:** `src/run_pipeline.py` — argparse CLI that dynamically imports `src.pipelines.<name>.pipeline` and calls `run()`.

**Data flow:**
```
Binance (ccxt) → 01_raw/ → 02_intermediate/ → 03_features/ → 04_normalized/ → 05_splits/
                 fetch      clean+fill         tech indicators  rolling z-score   temporal 80/20
```

**Pipeline pattern:** Each pipeline exposes `run() -> None`. Pipelines call `get_config()` once at the top, then pass config into node functions. Node functions are pure — they receive config/data as args and return data; they never call `get_config()` themselves. The pipeline (caller) handles all I/O.

**Config system:** `conf/parameters.yml` → loaded by `src/utils/config.py` into nested frozen dataclasses (`AppConfig`). All config is immutable at runtime. Never hardcode values in source.

**Logging:** `src/utils/logger.py` — idempotent `setup_logging()`, use `get_logger(__name__)` everywhere. Log level via `LOG_LEVEL` env var.

## Current State

Implemented: `data_engineering`, `feature_engineering` (nodes + pipeline + tests).
Stubs only: `training`, `backtesting`, `deployment`, `src/environments/trading_env.py`.

## Key Conventions

- **Package manager:** uv. **Formatter:** Black. **Linter:** Ruff (line-length 120, py312).
- **Never mock financial calculations** — mock only I/O (e.g., ccxt exchange). Hand-compute expected values in tests, assert with `pytest.approx`.
- **Temporal splits only** — never random. 80/20 train/test with walk-forward CV.
- **Environment and live bot must share identical execution logic.** If they diverge, backtest results are invalid.
- Google-style docstrings. Functions ≤ 40 lines.
- Test helpers (`_make_ohlcv_df()`, `_make_config()`) defined at module level in each test file. Use `tmp_path` for file-writing tests.
- Symbol filenames: `BTC/USDT` → `BTCUSDT_*.parquet`.

## Domain Details

**Missing candle policy:** ≤3 gaps → fill; 4–20 → fill + mark stale; >20 → drop + new segment.

**Technical indicators (12 columns):** `rsi_14`, `sma_20`, `ema_20`, `stoch_k`, `stoch_d`, `macd`, `macd_signal`, `ad`, `obv`, `roc_10`, `williams_r_14`, `disparity_20`. All via `pandas-ta`.

**Normalization:** Rolling z-score on TI columns only. Zero-std → 0.0. First 500 rows dropped (warmup).



# Coding Principles — DRL Crypto Trading Bot

Paste this at the start of every phase prompt. These are non-negotiable.


1. TEST EVERYTHING THAT TOUCHES MONEY

Every financial calculation (portfolio value, rewards, costs, metrics) gets a hand-verified test with manually computed expected results.
Property-based tests for invariants: balance ≥ 0, shares ≥ 0, portfolio_value = balance + sum(shares × prices), round-trip cost = exactly 2 × (tc + slippage) × price × shares.
No mocking financial logic. Mock I/O and APIs, never the math.
Aim for 90%+ coverage on the Gymnasium environment — it's the foundation everything else depends on.

2. ENGINEERED ENOUGH

Type hints on every function. Dataclasses for structured data. Enums for fixed categories.
Extract shared logic at 3+ usages. No abstract base classes without 2+ concrete implementations.
No plugin systems, event buses, or "future extensibility" scaffolding.
Litmus test: can a Python-intermediate dev understand this function in 60 seconds?

3. HANDLE EDGE CASES, DON'T SKIP THEM
Every edge case must be: detected → handled → logged → alerted (if live) → tested.
Must handle: insufficient cash (partial fill), selling more than held (sell all remaining), NaN/Inf in state vector (catch at boundary, never propagate), exchange timeout mid-order (query status before retry, never assume failure), stale/missing candles (follow the policy in config), model output outside [-1,1] (clip + warn).
4. DRY — ONE TRUTH PER FORMULA
Portfolio value, TI computation, order execution logic, cost calculation, and Sharpe ratio each live in ONE utility function. If the same formula appears in two files, one of them will become wrong after the next edit.
Critical: The Gymnasium environment and live bot MUST use identical execution logic (sell-first priority, partial fills, cost calc). Extract to shared utils. If they diverge, backtests become fiction.
5. DEFENSIVE & LOUD
python# Assert after every state transition:
assert self.balance >= 0, f"Negative balance: {self.balance}"
assert all(s >= 0 for s in self.shares.values())
assert np.all(np.isfinite(observation))

In development: crash hard on bad state. A crashed backtest beats a silently wrong one.
In production: never crash the main loop. Catch → log → alert via Telegram → skip cycle. On unrecoverable error: liquidate all, halt, alert. Default to inaction when uncertain.
Never silently swallow exceptions. No bare except: pass. Ever.

6. LOGGING & REPRODUCIBILITY

Log every trade decision, every skipped action, every edge case, every API call with latency.
Pin all random seeds explicitly (numpy, torch, gymnasium, SB3). Never rely on global state.
Save parameters.yml + git commit hash alongside every trained model.
Use deterministic=True for all evaluation and live inference.

7. CODE STYLE

Python 3.11+. Google-style docstrings. Functions ≤ 40 lines.
Full names over abbreviations: portfolio_value not pv, transaction_cost not tc.
Constants in UPPER_SNAKE_CASE. No magic numbers in function bodies.
Format with Black, lint with Ruff. No exceptions.


The Three Rules That Matter Most

Environment and live bot use identical execution logic. If they diverge, your backtest is a lie.
Test every calculation that touches money. Hand-compute the expected result, then assert it.
When in doubt, do nothing and alert. The market will be there in 5 minutes. Your capital might not.