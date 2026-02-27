"""Training pipeline node functions.

Pure functions that receive config/data as arguments and return results.
Never call get_config() — the pipeline (caller) handles all I/O.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from pandas import DataFrame, Timestamp
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.environments.trading_env import TradingEnv
from src.utils.config import EnvironmentConfig, TD3Config
from src.utils.logger import get_logger
from src.utils.metrics import (
    compute_cumulative_profit_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
)

logger = get_logger(__name__)


def pin_random_seeds(seed: int) -> None:
    """Pin all random seeds for reproducibility.

    Args:
        seed: The seed value to set for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("Pinned random seeds to %d", seed)


def generate_rolling_cv_folds(
    data: dict[str, DataFrame],
    train_months: int,
    validation_months: int,
) -> list[dict[str, Any]]:
    """Generate calendar-based rolling cross-validation folds.

    Args:
        data: Dict mapping symbol to DataFrame with datetime index.
        train_months: Number of months for each training window.
        validation_months: Number of months for each validation window.

    Returns:
        List of fold dicts with keys: fold_index, train_start, train_end,
        val_start, val_end.
    """
    # Use the first symbol to determine date range
    first_symbol = next(iter(data))
    index = data[first_symbol].index
    data_start = index.min()
    data_end = index.max()

    folds = []
    fold_index = 0
    train_start = data_start

    while True:
        train_end = train_start + _months_offset(train_months)
        val_start = train_end
        val_end = val_start + _months_offset(validation_months)

        if val_end > data_end:
            break

        folds.append({
            "fold_index": fold_index,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
        })

        fold_index += 1
        # Slide forward by validation_months
        train_start = train_start + _months_offset(validation_months)

    logger.info("Generated %d rolling CV folds", len(folds))
    return folds


def _months_offset(months: int) -> Any:
    """Return a pandas DateOffset for the given number of months."""
    from pandas.tseries.offsets import DateOffset

    return DateOffset(months=months)


def slice_data_by_dates(
    data: dict[str, DataFrame],
    start: Timestamp,
    end: Timestamp,
) -> dict[str, DataFrame]:
    """Slice all symbol DataFrames to a date range.

    Args:
        data: Dict mapping symbol to DataFrame with datetime index.
        start: Start timestamp (inclusive).
        end: End timestamp (exclusive).

    Returns:
        Dict with sliced DataFrames.
    """
    sliced = {}
    for symbol, df in data.items():
        mask = (df.index >= start) & (df.index < end)
        sliced[symbol] = df.loc[mask].copy()
    return sliced


def create_trading_env(
    data: dict[str, DataFrame],
    symbols: list[str],
    initial_balance: float,
    trading_fee_pct: float,
    slippage_pct: float,
    env_config: EnvironmentConfig,
) -> TradingEnv:
    """Create a TradingEnv instance.

    Args:
        data: Dict mapping symbol to DataFrame.
        symbols: List of symbol names.
        initial_balance: Starting cash balance.
        trading_fee_pct: Trading fee fraction.
        slippage_pct: Slippage fraction.
        env_config: Environment configuration.

    Returns:
        Configured TradingEnv.
    """
    return TradingEnv(
        data=data,
        symbols=symbols,
        initial_balance=initial_balance,
        trading_fee_pct=trading_fee_pct,
        slippage_pct=slippage_pct,
        window_size=env_config.window_size,
        reward_scaling=env_config.reward_scaling,
        max_position=env_config.max_position,
    )


def create_td3_agent(
    env: TradingEnv,
    td3_config: TD3Config,
    seed: int,
) -> TD3:
    """Create a TD3 agent with specified configuration.

    Args:
        env: The Gymnasium trading environment.
        td3_config: TD3 hyperparameter configuration.
        seed: Random seed for the agent.

    Returns:
        Configured TD3 agent.
    """
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=np.full(n_actions, td3_config.action_noise_std),
    )

    return TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=td3_config.learning_rate,
        gamma=td3_config.gamma,
        tau=td3_config.tau,
        batch_size=td3_config.batch_size,
        buffer_size=td3_config.buffer_size,
        learning_starts=td3_config.learning_starts,
        train_freq=td3_config.train_freq,
        policy_delay=td3_config.policy_delay,
        target_noise_clip=td3_config.target_noise_clip,
        target_policy_noise=td3_config.target_policy_noise,
        action_noise=action_noise,
        policy_kwargs={
            "net_arch": {
                "pi": td3_config.net_arch.pi,
                "qf": td3_config.net_arch.qf,
            }
        },
        seed=seed,
        verbose=0,
    )


def evaluate_agent(
    agent: TD3,
    env: TradingEnv,
) -> dict[str, float]:
    """Run a full deterministic episode and compute metrics.

    Args:
        agent: Trained TD3 agent.
        env: Evaluation environment.

    Returns:
        Dict with sharpe_ratio, cpr, max_drawdown keys.
    """
    obs, _ = env.reset()
    portfolio_values = [env.initial_balance]
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

    pv_array = np.array(portfolio_values, dtype=np.float64)

    return {
        "sharpe_ratio": compute_sharpe_ratio(pv_array),
        "cpr": compute_cumulative_profit_ratio(pv_array),
        "max_drawdown": compute_max_drawdown(pv_array),
    }


def log_experiment_to_mlflow(
    seed: int,
    fold_metrics: list[dict[str, float]],
    best_model_path: str | None,
    config_params: dict[str, Any],
) -> None:
    """Log a single seed's results to MLflow.

    Args:
        seed: The random seed used.
        fold_metrics: List of metric dicts per fold.
        best_model_path: Path to the best model checkpoint.
        config_params: Flattened config dict to log as params.
    """
    with mlflow.start_run(run_name=f"seed_{seed}"):
        mlflow.log_params(config_params)
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_folds", len(fold_metrics))

        for i, metrics in enumerate(fold_metrics):
            for key, value in metrics.items():
                mlflow.log_metric(f"fold_{i}_{key}", value)

        # Log mean metrics across folds
        if fold_metrics:
            mean_metrics = aggregate_fold_metrics(fold_metrics)
            for key, value in mean_metrics.items():
                mlflow.log_metric(key, value)

        if best_model_path and Path(best_model_path).exists():
            mlflow.log_artifact(best_model_path)

    logger.info("Logged MLflow run for seed=%d", seed)


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Compute mean metrics across folds.

    Args:
        fold_metrics: List of metric dicts per fold.

    Returns:
        Dict with mean_{key} for each metric.
    """
    if not fold_metrics:
        return {}

    keys = fold_metrics[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in fold_metrics]
        result[f"mean_{key}"] = float(np.mean(values))
    return result


def aggregate_seed_results(
    seed_results: list[dict[str, float]],
) -> dict[str, float]:
    """Compute mean and std across seed results.

    Args:
        seed_results: List of aggregated metric dicts per seed.

    Returns:
        Dict with mean_{key} and std_{key} for each metric.
    """
    if not seed_results:
        return {}

    keys = seed_results[0].keys()
    result = {}
    for key in keys:
        values = [r[key] for r in seed_results]
        result[f"mean_{key}"] = float(np.mean(values))
        result[f"std_{key}"] = float(np.std(values))
    return result
