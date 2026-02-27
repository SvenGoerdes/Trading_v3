# Trading v3 — DRL Crypto Trading Bot

A Deep Reinforcement Learning crypto trading bot using the **TD3** (Twin Delayed DDPG) algorithm to learn multi-asset portfolio allocation strategies from historical market data. Built as a 5-stage pipeline architecture where each stage transforms data for the next.

## Architecture

```
conf/parameters.yml          <-- All configuration lives here
        |
        v
+---------------------+
|  Data Engineering    |  Fetches raw OHLCV candles from Binance,
|  (Pipeline 1)        |  handles missing candles, validates data
|                     |
|  data/01_raw/       |  -> Raw Parquet files
|  data/02_intermediate| -> Cleaned Parquet files
+--------+------------+
         |
         v
+---------------------+
|  Feature Engineering |  Computes 12 technical indicators,
|  (Pipeline 2)        |  normalizes with rolling z-score,
|                     |  creates temporal train/test splits
|  data/03_features/  |  -> OHLCV + TI columns
|  data/04_normalized/|  -> Z-score normalized TI columns
|  data/05_splits/    |  -> Train/test Parquet files
+--------+------------+
         |
         v
+---------------------+
|  Training            |  TD3 agent trained in Gymnasium env,
|  (Pipeline 3)        |  5-seed runs, rolling CV, MLflow logging
|                     |
|  models/            |  -> Best model checkpoints per seed
|  mlruns/            |  -> MLflow experiment tracking
+--------+------------+
         |
         v
+---------------------+
|  Backtesting         |  Historical evaluation against benchmark
|  (Pipeline 4)        |  (not yet implemented)
+--------+------------+
         |
         v
+---------------------+
|  Deployment          |  Live trading bot
|  (Pipeline 5)        |  (not yet implemented)
+---------------------+
```

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# 1. Fetch and clean market data
uv run python -m src.run_pipeline --pipeline=data_engineering

# 2. Compute indicators, normalize, and split
uv run python -m src.run_pipeline --pipeline=feature_engineering

# 3. Train TD3 agent (5 seeds x rolling CV folds)
uv run python -m src.run_pipeline --pipeline=training

# View training results in MLflow
uv run mlflow ui --backend-store-uri mlruns
# Then open http://localhost:5000

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

## Project Structure

```
Trading_v3/
├── conf/
│   └── parameters.yml            # All configuration (symbols, TIs, hyperparams, etc.)
├── src/
│   ├── run_pipeline.py            # CLI entry point (--pipeline=<name>)
│   ├── utils/
│   │   ├── config.py              # Frozen dataclass config loader
│   │   ├── logger.py              # Structured logging setup
│   │   ├── portfolio.py           # Shared portfolio execution logic (env + live bot)
│   │   └── metrics.py             # Sharpe ratio, CPR, max drawdown
│   ├── pipelines/
│   │   ├── data_engineering/
│   │   │   ├── nodes.py           # Fetch + clean OHLCV functions
│   │   │   └── pipeline.py        # Orchestrator
│   │   ├── feature_engineering/
│   │   │   ├── nodes.py           # TI computation + normalization + splitting
│   │   │   └── pipeline.py        # Orchestrator
│   │   ├── training/
│   │   │   ├── nodes.py           # Seed pinning, CV folds, env/agent creation, evaluation
│   │   │   └── pipeline.py        # Multi-seed rolling CV orchestrator
│   │   ├── backtesting/           # (placeholder)
│   │   └── deployment/            # (placeholder)
│   └── environments/
│       └── trading_env.py         # Gymnasium multi-asset trading environment
├── tests/
│   ├── test_config.py             # Config loading tests
│   ├── test_data_pipeline.py      # Data engineering tests
│   ├── test_feature_engineering.py # Feature engineering tests
│   ├── test_portfolio.py          # Portfolio execution logic tests
│   ├── test_metrics.py            # Performance metrics tests
│   ├── test_trading_env.py        # Gymnasium environment tests
│   └── test_training_nodes.py     # Training node function tests
├── data/                          # Gitignored, created at runtime
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_features/
│   ├── 04_normalized/
│   └── 05_splits/
├── models/                        # Saved model checkpoints
├── mlruns/                        # MLflow experiment tracking
├── logs/                          # Development changelogs
├── docs/
│   └── system_overview.md         # Detailed system documentation
└── pyproject.toml                 # uv + hatchling, Python >=3.12
```

## Configuration

All settings live in `conf/parameters.yml`, loaded into frozen (immutable) dataclasses at runtime:

| Section | Key Settings |
|---|---|
| **Trading universe** | 10 symbols (BTC, ETH, SOL, AVAX, LINK, MATIC, UNI, AAVE, LTC, NEAR), 5m candles |
| **Portfolio** | 10k USDT initial balance, 0.1% taker fee, 0.05% slippage |
| **Data** | 120-day lookback, missing candle policy (ffill <=3, stale 4-20, split >20) |
| **Indicators** | RSI(14), SMA(20), EMA(20), Stochastic(14), MACD(12,26,9), A/D, OBV, ROC(10), Williams %R(14), Disparity(20) |
| **Normalization** | Rolling z-score, window=500 |
| **Split** | 80/20 temporal, 5 walk-forward CV folds |
| **TD3** | LR 0.0001, gamma 0.99, tau 0.005, batch 256, buffer 100K, 1M timesteps |
| **Environment** | Window size 50, reward scaling 1.0, max position 1.0 |
| **Training** | 5 seeds [42, 123, 456, 789, 1024], 3mo train / 1mo val rolling CV |

## Pipelines

### Pipeline 1: Data Engineering

Fetches raw OHLCV data from Binance via ccxt and cleans it:

1. **Fetch** -- Paginated candle downloads, saved as raw Parquet per symbol
2. **Gap detection** -- Identifies missing candles by comparing actual vs expected timestamps
3. **Clean** -- Applies missing candle policy: forward-fill small gaps, flag stale for medium gaps, segment-split for large gaps
4. **Validate** -- Flags negative prices and invalid OHLC relationships

### Pipeline 2: Feature Engineering

Enriches price data with technical indicators and prepares for training:

1. **Compute TIs** -- 12 indicator columns per symbol via pandas-ta
2. **Normalize** -- Rolling z-score (window=500) on TI columns only, zero-std handled as 0.0
3. **Split** -- Temporal 80/20 train/test split with walk-forward CV folds

### Pipeline 3: Training

Trains a TD3 agent across multiple seeds with rolling cross-validation:

1. **Load** -- Reads `{SYMBOL}_train.parquet` from `data/05_splits/`
2. **Generate folds** -- Calendar-based sliding window (3mo train / 1mo val)
3. **Per seed** -- Pin random seeds (numpy, torch, random) for reproducibility
4. **Per fold** -- Slice data, create train/val `TradingEnv` instances, create TD3 agent, train, evaluate
5. **Track best** -- Save best model checkpoint per seed (by validation Sharpe ratio)
6. **Log to MLflow** -- Params, per-fold metrics, mean metrics, model artifacts
7. **Aggregate** -- Cross-seed mean/std summary

### Trading Environment

`TradingEnv(gym.Env)` -- multi-asset portfolio trading:

- **Observation**: Flat vector = `[windowed_features | portfolio_weights | cash_ratio]`
- **Action**: `Box(-1, 1)` per asset, mapped to allocation weights via `(action + 1) / 2`
- **Reward**: `log(new_portfolio_value / old_portfolio_value) * reward_scaling`
- **Execution**: Sell-first rebalancing with partial fills when cash is insufficient
- **Termination**: End of data (truncated) or portfolio value <= 0 (terminated)
- **Invariants**: Balance >= 0, holdings >= 0, observation always finite -- asserted after every step

### Shared Portfolio Logic

`src/utils/portfolio.py` is the **single source of truth** for all portfolio math. Both the Gymnasium environment and the future live bot use these identical functions:

- `compute_portfolio_value` -- cash + sum(holdings * prices)
- `compute_transaction_cost` -- trade_value * (fee + slippage)
- `compute_target_holdings` -- weight-to-shares with max_position clipping
- `execute_rebalance` -- sell-first priority, partial fills, cost tracking

### Performance Metrics

- **Sharpe ratio** -- Log returns, annualized for 5-minute candle frequency
- **Cumulative profit ratio (CPR)** -- final / initial portfolio value
- **Max drawdown** -- Peak-to-trough decline as a positive fraction

## Testing

110 tests across 7 files:

```bash
uv run pytest tests/ -v                           # all tests
uv run pytest tests/test_trading_env.py -v         # environment (15 tests)
uv run pytest tests/test_portfolio.py -v           # portfolio math (10 tests)
uv run pytest tests/test_training_nodes.py -v      # training nodes (11 tests)
uv run pytest tests/test_metrics.py -v             # metrics (8 tests)
uv run pytest tests/test_config.py -v              # config (9 tests)
uv run pytest tests/test_data_pipeline.py -v       # data pipeline (17 tests)
uv run pytest tests/test_feature_engineering.py -v # features (26 tests)
```

Key testing principles:
- **Never mock financial calculations** -- mock only I/O (ccxt exchange, MLflow)
- **Hand-compute expected values** in tests, assert with `pytest.approx`
- **Stress tests** on the environment: random actions over full episodes to verify balance/holdings invariants
- **Round-trip cost invariant**: buy then sell same shares costs exactly `2 * (fee + slippage) * value`

## Tech Stack

- **Python** >= 3.12
- **Package manager**: uv
- **RL**: Stable-Baselines3 (TD3), Gymnasium
- **Data**: pandas, numpy, pandas-ta, ccxt
- **ML tracking**: MLflow
- **Deep learning**: PyTorch
- **Formatting**: Black
- **Linting**: Ruff (line-length 120, py312)
- **Testing**: pytest, pytest-mock
