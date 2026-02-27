# Trading v3 — System Overview

## What is this?

A Deep Reinforcement Learning (DRL) crypto trading bot that uses the TD3 (Twin Delayed DDPG) algorithm to learn trading strategies from historical market data. The system is built as a pipeline architecture where each stage transforms data for the next.

## Architecture

```
conf/parameters.yml          <-- All configuration lives here
        │
        v
┌─────────────────────┐
│  Data Engineering    │  Fetches raw OHLCV candles from Binance,
│  (Pipeline 1)        │  handles missing candles, validates data
│                     │
│  data/01_raw/       │  → Raw Parquet files
│  data/02_intermediate│ → Cleaned Parquet files
└────────┬────────────┘
         │
         v
┌─────────────────────┐
│  Feature Engineering │  Computes 10 technical indicators,
│  (Pipeline 2)        │  normalizes with rolling z-score,
│                     │  creates temporal train/test splits
│  data/03_features/  │  → OHLCV + TI columns
│  data/04_normalized/│  → Z-score normalized TI columns
│  data/05_splits/    │  → Train/test Parquet files
└────────┬────────────┘
         │
         v
┌─────────────────────┐
│  Training            │  TD3 agent trained in Gymnasium env,
│  (Pipeline 3)        │  5-seed runs, rolling CV, MLflow logging
└────────┬────────────┘
         │
         v
┌─────────────────────┐
│  Backtesting         │  Historical evaluation against benchmark
│  (Pipeline 4)        │  (not yet implemented)
└────────┬────────────┘
         │
         v
┌─────────────────────┐
│  Deployment          │  Live trading bot
│  (Pipeline 5)        │  (not yet implemented)
└─────────────────────┘
```

## How to Run

```bash
# Install dependencies
uv sync --all-extras

# Run a pipeline
uv run python -m src.run_pipeline --pipeline=data_engineering
uv run python -m src.run_pipeline --pipeline=feature_engineering

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

## Project Structure

```
Trading_v3/
├── conf/
│   └── parameters.yml        # All configuration (symbols, TIs, hyperparams, etc.)
├── src/
│   ├── run_pipeline.py        # CLI entry point (--pipeline=<name>)
│   ├── utils/
│   │   ├── config.py          # Frozen dataclass config loader
│   │   └── logger.py          # Structured logging setup
│   ├── pipelines/
│   │   ├── data_engineering/
│   │   │   ├── nodes.py       # Fetch + clean OHLCV functions
│   │   │   └── pipeline.py    # Orchestrator
│   │   ├── feature_engineering/
│   │   │   ├── nodes.py       # TI computation + normalization + splitting
│   │   │   └── pipeline.py    # Orchestrator
│   │   ├── training/          # (placeholder)
│   │   ├── backtesting/       # (placeholder)
│   │   └── deployment/        # (placeholder)
│   └── environments/
│       └── trading_env.py     # Gymnasium trading env (placeholder)
├── tests/
│   ├── test_config.py         # Config loading tests
│   ├── test_data_pipeline.py  # Data engineering tests
│   └── test_feature_engineering.py  # Feature engineering tests
├── data/                      # Gitignored, created at runtime
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_features/
│   ├── 04_normalized/
│   └── 05_splits/
├── logs/                      # Changelogs
└── pyproject.toml             # uv + hatchling, Python >=3.12
```

## Configuration (`conf/parameters.yml`)

All settings are centralized in one YAML file, loaded into frozen (immutable) dataclasses:

| Section | Key settings |
|---------|-------------|
| **Trading universe** | 10 symbols (BTC, ETH, SOL, AVAX, LINK, MATIC, UNI, AAVE, LTC, NEAR), 5m candles |
| **Portfolio** | 10k USDT initial, 0.1% fee, 0.05% slippage |
| **Data** | 120-day lookback, missing candle policy (ffill ≤3, stale flag 4–20, split >20) |
| **Indicators** | RSI(14), SMA(20), EMA(20), Stochastic(14), MACD(12,26,9), A/D, OBV, ROC(10), Williams %R(14), Disparity(20) |
| **Normalization** | Rolling z-score, window=500 |
| **Split** | 80/20 temporal, 5 walk-forward CV folds |
| **TD3** | LR 0.0001/0.001, gamma 0.99, tau 0.005, batch 256, 1M timesteps |

## Pipeline 1: Data Engineering

**Purpose:** Fetch raw market data and clean it for downstream processing.

### Flow
1. **Fetch** — Connects to Binance via ccxt, paginates through OHLCV candles (1000 per request, 100ms sleep between). Saves raw Parquet per symbol to `data/01_raw/`.
2. **Gap detection** — Identifies missing candles by comparing actual vs expected timestamps.
3. **Clean** — Reindexes to full expected time range, then applies missing candle policy:
   - **≤3 consecutive missing:** Forward-fill, mark `is_filled=True`
   - **4–20 consecutive missing:** Forward-fill, mark `is_filled=True` + `is_stale=True`
   - **>20 consecutive missing:** Don't fill; rows are dropped and a new `segment_id` is assigned
4. **Validate** — Flags negative prices (set to NaN) and warns on invalid OHLC relationships (low > high).
5. Saves cleaned data to `data/02_intermediate/`.

### Key design decisions
- Exchange instance uses ccxt's built-in rate limiter
- Parquet format for efficient columnar storage
- `segment_id` column allows downstream code to avoid training across large data gaps

## Pipeline 2: Feature Engineering

**Purpose:** Enrich price data with technical indicators, normalize them, and split for training.

### Flow
1. **Compute TIs** — Reads cleaned Parquet files, computes 10 indicators per symbol using pandas-ta. Each indicator is a thin wrapper function with explicit column naming. Warmup NaN rows are dropped. Saves to `data/03_features/`.
2. **Normalize** — Applies rolling z-score (window=500) to TI columns only (not price/volume). Creates `{col}_norm` columns alongside originals. Handles zero-std edge case by replacing inf/NaN with 0.0. Drops first 500 rows. Saves to `data/04_normalized/`.
3. **Split** — Temporal (not random) 80/20 train/test split. Generates walk-forward CV folds for hyperparameter tuning. Saves to `data/05_splits/`.

### Technical Indicators

| Indicator | Function | Output columns |
|-----------|----------|---------------|
| RSI | `compute_rsi(close, 14)` | `rsi_14` |
| SMA | `compute_sma(close, 20)` | `sma_20` |
| EMA | `compute_ema(close, 20)` | `ema_20` |
| Stochastic | `compute_stochastic(high, low, close, 14)` | `stoch_k`, `stoch_d` |
| MACD | `compute_macd(close, 12, 26, 9)` | `macd`, `macd_signal` |
| A/D | `compute_accumulation_distribution(h, l, c, v)` | `ad` |
| OBV | `compute_on_balance_volume(close, volume)` | `obv` |
| ROC | `compute_rate_of_change(close, 10)` | `roc_10` |
| Williams %R | `compute_williams_r(high, low, close, 14)` | `williams_r_14` |
| Disparity | `compute_disparity_index(close, 20)` | `disparity_20` |

### Rolling Z-Score Normalization

```
rolling_mean = col.rolling(500, min_periods=1).mean()
rolling_std  = col.rolling(500, min_periods=1).std()
z = (col - rolling_mean) / rolling_std
# inf/NaN from zero std → replaced with 0.0
```

Only TI columns are normalized. Price and volume columns are kept as-is. Each normalized column is named `{original}_norm` (e.g., `rsi_14_norm`).

## Utils

### Config (`src/utils/config.py`)
- Loads `conf/parameters.yml` via `pyyaml.safe_load`
- Parses into nested frozen dataclasses (immutable after creation)
- `get_config()` is the main entry point — returns a fully typed `AppConfig`

### Logger (`src/utils/logger.py`)
- `setup_logging()` — idempotent, configures root logger with structured format
- Reads `LOG_LEVEL` env var (default: INFO)
- `get_logger(name)` — returns a named logger, auto-initializes if needed

## Testing

55 tests across 3 files:

- **test_config.py** — YAML loading edge cases, parse validation, dataclass immutability
- **test_data_pipeline.py** — Timeframe conversion, mocked ccxt fetching (pagination, empty), gap detection, all cleaning policies, OHLC validation
- **test_feature_engineering.py** — Hand-verified TI values (SMA, EMA, RSI bounds, OBV steps, ROC, MACD zero, Williams %R range, disparity zero), normalization (mean/std, zero-std), temporal split ordering

Tests mock I/O (ccxt exchange) but never mock financial calculations, per project coding principles.
