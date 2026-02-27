# Log 003 — TD3 Training Pipeline

**Date:** 2026-02-27
**Phase:** 2 — Gymnasium Environment & TD3 Training

## Summary

Implemented the full TD3 training pipeline: shared portfolio execution logic, performance metrics, a multi-asset Gymnasium trading environment, rolling cross-validation training with 5-seed runs, and MLflow experiment tracking. All 110 tests pass, lint is clean.

## Changes

### Config (`conf/parameters.yml`)
- Replaced `learning_rate_actor` / `learning_rate_critic` with single `learning_rate: 0.0001`
- Reduced `buffer_size` from `1_000_000` to `100_000`
- Added new `training:` section:
  - `seeds: [42, 123, 456, 789, 1024]` — 5-seed training for statistical robustness
  - `cv_train_months: 3`, `cv_validation_months: 1` — rolling CV windows
  - `model_dir: "models"` — checkpoint save location
  - `mlflow_experiment_name`, `mlflow_tracking_uri` — MLflow settings

### Config Dataclasses (`src/utils/config.py`)
- `NetArchConfig(pi, qf)` — network architecture for actor/critic
- `TD3Config` — all TD3 hyperparameters (learning_rate, gamma, tau, batch_size, buffer_size, etc.)
- `EnvironmentConfig(window_size, reward_scaling, max_position)`
- `TrainingConfig(seeds, cv_train_months, cv_validation_months, model_dir, mlflow_*)`
- All wired into `AppConfig` and `parse_config()`

### New: Shared Portfolio Logic (`src/utils/portfolio.py`)
Single source of truth for portfolio math — used by both the Gymnasium env and (future) live bot:
- `compute_portfolio_value(balance, holdings, prices)` — cash + holdings value
- `compute_transaction_cost(trade_value, fee_pct, slippage_pct)` — per-trade cost
- `compute_target_holdings(weights, portfolio_value, prices, max_position)` — weight-to-shares conversion with max_position clipping and renormalization
- `execute_rebalance(balance, holdings, target, prices, fee, slip)` — **sell-first priority**, partial fills when cash insufficient, assertions on every output

### New: Performance Metrics (`src/utils/metrics.py`)
- `compute_sharpe_ratio(portfolio_values)` — log returns, annualized for 5-min candles (√(365×24×12))
- `compute_cumulative_profit_ratio(portfolio_values)` — final / initial
- `compute_max_drawdown(portfolio_values)` — peak-to-trough as positive fraction

### New: Gymnasium Trading Environment (`src/environments/trading_env.py`)
`TradingEnv(gym.Env)` — multi-asset portfolio trading:
- **Constructor**: Takes `data: dict[str, DataFrame]`, inner-joins all symbols on datetime index, extracts `*_norm` columns as 3D feature array and `close` as 2D price array
- **Observation**: flat vector = `[window_features | portfolio_weights | cash_ratio]`
- **Action**: `Box(-1, 1, shape=(n_assets,))` mapped to weights via `(action + 1) / 2`
- **Step**: calls `compute_target_holdings` → `execute_rebalance` from `portfolio.py`
- **Reward**: `log(new_pv / old_pv) * reward_scaling`
- **Termination**: step >= max_steps (truncated) or portfolio_value <= 0 (terminated)
- **Invariants**: `_assert_state_valid()` after every step — balance ≥ 0, holdings ≥ 0, observation finite

### New: Training Nodes (`src/pipelines/training/nodes.py`)
Pure functions (no I/O, no `get_config()` calls):
- `pin_random_seeds(seed)` — numpy, torch, random, CUDA
- `generate_rolling_cv_folds(data, train_months, val_months)` — calendar-based sliding window
- `slice_data_by_dates(data, start, end)` — slice all symbol DataFrames
- `create_trading_env(data, symbols, balance, fees, env_config)` — factory
- `create_td3_agent(env, td3_config, seed)` — TD3 with NormalActionNoise, net_arch dict
- `evaluate_agent(agent, env)` — deterministic full episode → Sharpe, CPR, max_drawdown
- `log_experiment_to_mlflow(seed, fold_metrics, model_path, config_params)` — one run per seed
- `aggregate_fold_metrics(fold_metrics)` / `aggregate_seed_results(seed_results)` — mean/std

### New: Training Pipeline (`src/pipelines/training/pipeline.py`)
Orchestration flow:
1. Load all `{SYMBOL}_train.parquet` from `data/05_splits/`
2. Generate rolling CV folds (3mo train / 1mo val)
3. Setup MLflow (tracking URI + experiment)
4. For each seed → pin seeds → for each fold → slice data → create train/val envs → create TD3 → train → evaluate → track best by Sharpe → save checkpoint → log to MLflow
5. Aggregate and log cross-seed summary (mean/std of all metrics)

### Tests (4 new files, 1 modified)
- `tests/test_portfolio.py` — 10 tests: portfolio value, transaction cost, target holdings, rebalance (buy, sell-then-buy, partial fill, no-trade, round-trip cost invariant)
- `tests/test_metrics.py` — 8 tests: Sharpe (constant, known, edge cases), CPR (gain, loss), max drawdown (none, 25%, full loss)
- `tests/test_trading_env.py` — 15 tests: space shapes/bounds, reset state, zero-action holds cash, reward finite/scaling, full episode termination, balance/holdings never negative (random action stress tests), observation always finite, portfolio weights + cash sum to 1
- `tests/test_training_nodes.py` — 11 tests: seed reproducibility, CV fold count/sliding/no-overlap, data slicing, TD3 agent creation, fold/seed aggregation math, MLflow logging (mocked)
- `tests/test_config.py` — updated fixture and assertions for new config types
- `tests/test_data_pipeline.py` — updated `_make_config` helper for new `AppConfig` fields

**Total: 110 tests passing, 0 failures, lint clean.**

## How to Use

### Prerequisites
Make sure data pipelines have been run first:
```bash
uv run python -m src.run_pipeline --pipeline=data_engineering
uv run python -m src.run_pipeline --pipeline=feature_engineering
```

This populates `data/05_splits/{SYMBOL}_train.parquet` files.

### Run Training
```bash
uv run python -m src.run_pipeline --pipeline=training
```

This will:
- Load training data for all 10 symbols
- Run 5-seed training with rolling 3mo/1mo cross-validation
- Save best model checkpoints per seed to `models/td3_seed{N}_best.zip`
- Log all metrics and artifacts to MLflow under `mlruns/`

### View MLflow Results
```bash
uv run mlflow ui --backend-store-uri mlruns
```
Then open http://localhost:5000 to compare seed runs, fold metrics, and download model checkpoints.

### Run Tests
```bash
uv run pytest tests/ -v                          # all 110 tests
uv run pytest tests/test_trading_env.py -v        # environment only
uv run pytest tests/test_portfolio.py -v          # portfolio math only
uv run pytest tests/test_training_nodes.py -v     # training nodes only
uv run pytest tests/test_metrics.py -v            # metrics only
```

### Key Configuration (conf/parameters.yml)
| Parameter | Default | Description |
|---|---|---|
| `td3.learning_rate` | 0.0001 | Shared actor/critic LR |
| `td3.buffer_size` | 100000 | Replay buffer size |
| `td3.total_timesteps` | 1000000 | Training steps per fold |
| `environment.window_size` | 50 | Lookback window for observations |
| `environment.max_position` | 1.0 | Max weight per asset |
| `training.seeds` | [42,123,456,789,1024] | Seeds for multi-seed training |
| `training.cv_train_months` | 3 | Rolling CV training window |
| `training.cv_validation_months` | 1 | Rolling CV validation window |

## Architecture Notes

- **Portfolio logic is shared**: `src/utils/portfolio.py` is the single source of truth. The Gymnasium env uses it, and the future live bot must also use it. If they diverge, backtest results are invalid.
- **Sell-first rebalance**: Sells execute before buys to free up cash. When cash is insufficient for all buys, proportional partial fills are applied.
- **Rolling CV**: Calendar-based sliding window (not row-based). Each fold slides forward by `cv_validation_months`.
- **MLflow**: One run per seed, fold-level and mean metrics logged. Best model artifact attached.
