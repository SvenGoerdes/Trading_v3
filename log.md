# Development Log

## 2026-02-25 — Project Scaffold

Created the full project scaffold for a DRL-based crypto trading bot using TD3.

### What was set up
- **pyproject.toml** — Project metadata, all deps (ccxt, gymnasium, stable-baselines3, pandas, torch, mlflow, etc.), dev tools (ruff, pytest), hatchling build config
- **Makefile** — Targets: `setup`, `test`, `lint`, `train`, `backtest`
- **conf/parameters.yml** — 10 trading pairs, TI config (SMA/EMA/RSI/MACD/Bollinger/ATR/OBV/VWAP), TD3 hyperparams, env settings, backtest/deployment config
- **src/run_pipeline.py** — CLI runner dispatching to pipeline modules via argparse
- **src/pipelines/** — 5 placeholder pipelines: data_engineering, feature_engineering, training, backtesting, deployment
- **src/environments/trading_env.py** — Gymnasium env placeholder
- **src/utils/** — config.py (YAML loader) and logger.py placeholders
- **Root files** — .gitignore, .env.example, .python-version (3.11)

### Verified
- `uv sync` installs 135 packages
- `python -m src.run_pipeline --pipeline=data_engineering` runs without import errors
- `ruff check` passes
- `pytest` discovers tests/ dir
- YAML loads correctly
