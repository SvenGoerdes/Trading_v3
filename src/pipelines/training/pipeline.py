"""Training Pipeline.

Trains a TD3 agent using Stable-Baselines3 on the custom
Gymnasium trading environment. Logs metrics to MLflow.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

from .nodes import (
    aggregate_seed_results,
    create_td3_agent,
    create_trading_env,
    evaluate_agent,
    generate_rolling_cv_folds,
    log_experiment_to_mlflow,
    pin_random_seeds,
    slice_data_by_dates,
)

logger = get_logger(__name__)


def _load_train_data(splits_dir: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load train parquet files for all symbols.

    Args:
        splits_dir: Path to the splits directory.
        symbols: List of symbol names (e.g. BTC/USDT).

    Returns:
        Dict mapping symbol to DataFrame.
    """
    data = {}
    for symbol in symbols:
        filename = symbol.replace("/", "") + "_train.parquet"
        path = Path(splits_dir) / filename
        df = pd.read_parquet(path)
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        data[symbol] = df
        logger.info("Loaded %s: %d rows", symbol, len(df))
    return data


def _flatten_config_params(config) -> dict[str, str]:
    """Flatten config into a flat dict for MLflow param logging."""
    return {
        "learning_rate": str(config.td3.learning_rate),
        "gamma": str(config.td3.gamma),
        "tau": str(config.td3.tau),
        "batch_size": str(config.td3.batch_size),
        "buffer_size": str(config.td3.buffer_size),
        "total_timesteps": str(config.td3.total_timesteps),
        "window_size": str(config.environment.window_size),
        "reward_scaling": str(config.environment.reward_scaling),
        "max_position": str(config.environment.max_position),
        "initial_balance": str(config.initial_balance),
        "trading_fee_pct": str(config.trading_fee_pct),
        "slippage_pct": str(config.slippage_pct),
    }


def run() -> None:
    """Run the TD3 training pipeline.

    1. Load all train parquet files.
    2. Generate rolling CV folds.
    3. For each seed: pin seeds, train on each fold, evaluate, log to MLflow.
    4. Aggregate and log cross-seed summary.
    """
    config = get_config()
    splits_dir = config.data.paths.splits
    symbols = config.symbols
    training_config = config.training

    # 1. Load data
    logger.info("Loading training data from %s", splits_dir)
    data = _load_train_data(splits_dir, symbols)

    # 2. Generate CV folds
    folds = generate_rolling_cv_folds(
        data,
        train_months=training_config.cv_train_months,
        validation_months=training_config.cv_validation_months,
    )
    logger.info("Generated %d CV folds", len(folds))

    # 3. Setup MLflow
    mlflow.set_tracking_uri(training_config.mlflow_tracking_uri)
    mlflow.set_experiment(training_config.mlflow_experiment_name)

    # 4. Create model directory
    model_dir = Path(training_config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    config_params = _flatten_config_params(config)
    all_seed_results = []

    # 5. Train across seeds
    for seed in training_config.seeds:
        logger.info("=== Training with seed=%d ===", seed)
        pin_random_seeds(seed)

        fold_metrics = []
        best_sharpe = -float("inf")
        best_model_path = None

        for fold in folds:
            fold_idx = fold["fold_index"]
            logger.info(
                "Fold %d: train [%s, %s) val [%s, %s)",
                fold_idx,
                fold["train_start"],
                fold["train_end"],
                fold["val_start"],
                fold["val_end"],
            )

            # Slice data for this fold
            train_data = slice_data_by_dates(
                data, fold["train_start"], fold["train_end"]
            )
            val_data = slice_data_by_dates(
                data, fold["val_start"], fold["val_end"]
            )

            # Create environments
            train_env = create_trading_env(
                train_data,
                symbols,
                config.initial_balance,
                config.trading_fee_pct,
                config.slippage_pct,
                config.environment,
            )
            val_env = create_trading_env(
                val_data,
                symbols,
                config.initial_balance,
                config.trading_fee_pct,
                config.slippage_pct,
                config.environment,
            )

            # Create and train agent
            agent = create_td3_agent(train_env, config.td3, seed)
            agent.learn(total_timesteps=config.td3.total_timesteps)

            # Evaluate on validation set
            metrics = evaluate_agent(agent, val_env)
            fold_metrics.append(metrics)
            logger.info("Fold %d metrics: %s", fold_idx, metrics)

            # Track best model by Sharpe ratio
            if metrics["sharpe_ratio"] > best_sharpe:
                best_sharpe = metrics["sharpe_ratio"]
                save_path = str(model_dir / f"td3_seed{seed}_best")
                agent.save(save_path)
                best_model_path = save_path + ".zip"
                logger.info(
                    "New best model (Sharpe=%.4f) saved to %s",
                    best_sharpe,
                    best_model_path,
                )

        # Log seed results to MLflow
        log_experiment_to_mlflow(seed, fold_metrics, best_model_path, config_params)

        # Aggregate fold metrics for this seed
        if fold_metrics:
            from .nodes import aggregate_fold_metrics

            seed_summary = aggregate_fold_metrics(fold_metrics)
            all_seed_results.append(seed_summary)

    # 6. Cross-seed summary
    if all_seed_results:
        final_summary = aggregate_seed_results(all_seed_results)
        logger.info("=== Cross-seed summary ===")
        for key, value in final_summary.items():
            logger.info("  %s: %.6f", key, value)
