"""Training Pipeline.

Trains a TD3 agent using Stable-Baselines3 on the custom
Gymnasium trading environment. Logs metrics to MLflow.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from stable_baselines3.common.callbacks import CallbackList

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.mlflow_metrics import TradingMetricsLogger

from .callbacks import MLflowDiagnosticsCallback
from .evaluation import run_full_evaluation
from .nodes import (
    aggregate_seed_results,
    create_td3_agent,
    create_trading_env,
    create_training_callback,
    evaluate_agent,
    generate_rolling_cv_folds,
    get_torch_device,
    pin_random_seeds,
    slice_data_by_dates,
)

logger = get_logger(__name__)

TIMING_LOG_PATH = Path("logs/training_timing.jsonl")


def _format_elapsed(seconds: float) -> str:
    """Format seconds as Xh YYm ZZs."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _append_timing_record(record: dict) -> None:
    """Append a JSON record to the timing log file."""
    TIMING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMING_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


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
    device = get_torch_device()
    total_folds_all_seeds = len(training_config.seeds) * len(folds)
    completed_folds = 0
    cumulative_fold_time = 0.0

    # 5. Train across seeds
    for seed_idx, seed in enumerate(training_config.seeds):
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

            fold_start_time = time.monotonic()

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

            # Create agent and callbacks
            agent = create_td3_agent(train_env, config.td3, seed)
            progress_callback = create_training_callback(
                total_timesteps=config.td3.total_timesteps,
                seed=seed,
                fold_index=fold_idx,
                log_every_steps=training_config.progress_log_every_steps,
            )

            metrics_logger = TradingMetricsLogger(
                symbols=symbols,
                initial_balance=config.initial_balance,
            )
            diagnostics_callback = MLflowDiagnosticsCallback(
                metrics_logger=metrics_logger,
            )

            callback_list = CallbackList([progress_callback, diagnostics_callback])
            agent.learn(
                total_timesteps=config.td3.total_timesteps,
                callback=callback_list,
            )

            # Evaluate on validation set (basic metrics for fold comparison)
            metrics = evaluate_agent(agent, val_env)
            fold_metrics.append(metrics)
            logger.info("Fold %d metrics: %s", fold_idx, metrics)

            # Fold timing
            fold_elapsed = time.monotonic() - fold_start_time
            completed_folds += 1
            cumulative_fold_time += fold_elapsed

            avg_fold_time = cumulative_fold_time / completed_folds
            remaining_folds = total_folds_all_seeds - completed_folds
            total_eta = avg_fold_time * remaining_folds

            logger.info(
                "Seed %d/%d | Fold %d/%d | Elapsed: %s | Fold ETA: %s | Total ETA: %s",
                seed_idx + 1,
                len(training_config.seeds),
                fold_idx + 1,
                len(folds),
                _format_elapsed(fold_elapsed),
                _format_elapsed(avg_fold_time * (len(folds) - fold_idx - 1)),
                _format_elapsed(total_eta),
            )

            # Append timing record to JSONL
            steps_per_second = (
                config.td3.total_timesteps / fold_elapsed
                if fold_elapsed > 0
                else 0.0
            )
            _append_timing_record({
                "timestamp": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "seed": seed,
                "fold_index": fold_idx,
                "device": device,
                "total_timesteps": config.td3.total_timesteps,
                "train_freq": config.td3.train_freq,
                "elapsed_seconds": round(fold_elapsed, 1),
                "steps_per_second": round(steps_per_second, 1),
                "train_range": (
                    f"{fold['train_start'].date()}/{fold['train_end'].date()}"
                ),
                "val_range": (
                    f"{fold['val_start'].date()}/{fold['val_end'].date()}"
                ),
            })

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

        # Log seed results to MLflow (with full evaluation on last fold's val env)
        with mlflow.start_run(run_name=f"seed_{seed}"):
            mlflow.log_params(config_params)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_folds", len(fold_metrics))

            for i, fm in enumerate(fold_metrics):
                for key, value in fm.items():
                    mlflow.log_metric(f"fold_{i}_{key}", value)

            if fold_metrics:
                from .nodes import aggregate_fold_metrics as _agg

                mean_metrics = _agg(fold_metrics)
                for key, value in mean_metrics.items():
                    mlflow.log_metric(key, value)

            if best_model_path and Path(best_model_path).exists():
                mlflow.log_artifact(best_model_path)

            # Run full diagnostics evaluation on last fold's validation env
            if val_env is not None:
                eval_metrics_logger = TradingMetricsLogger(
                    symbols=symbols,
                    initial_balance=config.initial_balance,
                )
                # Build price_data dict for timing/regime analysis
                price_data = {
                    sym: val_env.prices[:, i]
                    for i, sym in enumerate(val_env.symbols)
                }
                run_full_evaluation(
                    agent, val_env, price_data, eval_metrics_logger
                )

        logger.info("Logged MLflow run for seed=%d", seed)

        # Aggregate fold metrics for this seed
        if fold_metrics:
            from .nodes import aggregate_fold_metrics as _agg_folds

            seed_summary = _agg_folds(fold_metrics)
            all_seed_results.append(seed_summary)

    # 6. Cross-seed summary
    if all_seed_results:
        final_summary = aggregate_seed_results(all_seed_results)
        logger.info("=== Cross-seed summary ===")
        for key, value in final_summary.items():
            logger.info("  %s: %.6f", key, value)

    logger.info(
        "Training complete. Total wall time: %s",
        _format_elapsed(cumulative_fold_time),
    )
