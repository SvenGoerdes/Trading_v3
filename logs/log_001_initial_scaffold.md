# Log 001 — Initial Scaffold

**Date:** 2026-02-25
**Commit:** `ba6fff5` — Initial scaffold for DRL crypto trading bot

## What was set up

- Project structure with `src/`, `conf/`, `tests/` directories
- `pyproject.toml` with uv + hatchling, Python 3.11, dependencies (ccxt, gymnasium, stable-baselines3, pandas, numpy, ta, pyyaml, mlflow, torch, matplotlib)
- `conf/parameters.yml` with 10 symbols (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK), 1h timeframe, TD3 hyperparams, environment config
- `src/run_pipeline.py` — CLI runner dispatching to 5 pipelines
- 5 pipeline placeholders: data_engineering, feature_engineering, training, backtesting, deployment (all stubs)
- `src/utils/config.py` and `src/utils/logger.py` — empty placeholder files
- `src/environments/trading_env.py` — placeholder Gymnasium env
- `.gitignore` with Python, venv, IDE, ML artifacts, data, logs entries
