# Log 002 — Data Engineering & Feature Engineering Pipelines

**Date:** 2026-02-25
**Phase:** 1 — Data & Feature Pipelines

## Summary

Implemented the data engineering and feature engineering pipelines end-to-end, including config loading, logging, 10 technical indicators, rolling z-score normalization, and 55 passing tests.

## Changes

### Dependencies (`pyproject.toml`)
- Replaced `ta>=0.11` with `pandas-ta>=0.3.14`
- Added `pytest-mock>=3.14` to dev dependencies
- Bumped `requires-python` from `>=3.11` to `>=3.12` (pandas-ta requirement)
- Updated ruff `target-version` to `py312`
- `.python-version` updated from `3.11` to `3.12`

### Config (`conf/parameters.yml`) — full rewrite
- **Symbols:** BTC/USDT, ETH/USDT, SOL/USDT, AVAX/USDT, LINK/USDT, MATIC/USDT, UNI/USDT, AAVE/USDT, LTC/USDT, NEAR/USDT
- **Timeframe:** 5m (was 1h)
- **Lookback:** 120 days (was 730)
- **Missing candle policy:** ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20
- **10 TIs:** RSI(14), SMA(20), EMA(20), Stochastic(14), MACD(12,26,9), A/D, OBV, ROC(10), Williams %R(14), Disparity Index(SMA20)
- **Normalization:** rolling z-score, window=500
- **Data paths:** 5-layer Parquet structure (01_raw → 05_splits)
- **Split:** 80/20 temporal, 5 CV folds

### Utils
- **`src/utils/logger.py`** — Idempotent `setup_logging()` with LOG_LEVEL env var, `get_logger()` auto-initializes
- **`src/utils/config.py`** — 8 frozen dataclasses (`AppConfig`, `DataConfig`, `DataPaths`, `MissingCandlePolicy`, `MacdConfig`, `IndicatorsConfig`, `NormalizationConfig`, `SplitConfig`), `load_yaml()`, `parse_config()`, `get_config()`

### Data Engineering (`src/pipelines/data_engineering/`)
- **`nodes.py`** (new):
  - `create_exchange()` — Binance with rate limiting
  - `timeframe_to_milliseconds()` — converts '5m', '1h', '1d' to ms
  - `fetch_ohlcv_single_symbol()` — paginated OHLCV fetch with 100ms sleep
  - `fetch_ohlcv()` — iterates symbols, saves to `data/01_raw/`
  - `detect_candle_gaps()` — finds and reports gaps in candle index
  - `clean_single_symbol()` — reindex, ffill <=3, stale flag 4-20, segment split >20, OHLC validation
  - `clean_ohlcv()` — applies cleaning to all symbols, saves to `data/02_intermediate/`
- **`pipeline.py`** — orchestrator: config → fetch → clean

### Feature Engineering (`src/pipelines/feature_engineering/`)
- **`nodes.py`** (new):
  - 10 individual TI functions wrapping pandas-ta: `compute_rsi`, `compute_sma`, `compute_ema`, `compute_stochastic`, `compute_macd`, `compute_accumulation_distribution`, `compute_on_balance_volume`, `compute_rate_of_change`, `compute_williams_r`, `compute_disparity_index`
  - `compute_technical_indicators()` — applies all TIs, drops warmup NaN rows
  - `save_features()` — saves to `data/03_features/`
  - `normalize_features()` — rolling z-score on TI columns only, creates `{col}_norm` columns, drops first 500 rows
  - `save_normalized()` — saves to `data/04_normalized/`
  - `create_train_test_split()` — temporal 80/20 split with walk-forward CV folds
  - `save_splits()` — saves to `data/05_splits/`
- **`pipeline.py`** — orchestrator: read intermediate → TIs → normalize → split

### Tests (55 tests, all passing)
- **`tests/__init__.py`** — created for pytest discovery
- **`tests/test_config.py`** (9 tests) — YAML loading (valid, missing, malformed), parse_config types/missing keys, frozen dataclass immutability, real config loading
- **`tests/test_data_pipeline.py`** (19 tests) — timeframe conversion (valid + invalid), mocked ccxt fetch (single batch, pagination, empty), gap detection, cleaning (ffill, stale flag, segment split, idempotent, empty), OHLC validation (negative prices, bad low/high), parquet save
- **`tests/test_feature_engineering.py`** (27 tests) — hand-verified TIs (SMA known values, EMA formula, RSI bounds, stochastic columns/range, MACD columns/zero, OBV step-by-step, ROC known value, Williams %R range, disparity index zero, A/D naming), orchestration (all indicators present, no NaNs, empty), normalization (columns created, mean/std, zero-std=0, empty), train/test split (temporal order, sizes, CV folds, empty)

## Verification
- `uv sync` installs cleanly
- `uv run ruff check src/ tests/` — all checks passed
- `uv run pytest tests/ -v` — 55 passed in ~2s
