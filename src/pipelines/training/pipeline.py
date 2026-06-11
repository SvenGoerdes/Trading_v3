"""Training Pipeline.

Trains a TD3 agent using Stable-Baselines3 on the custom
Gymnasium trading environment. Logs metrics to MLflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from stable_baselines3.common.callbacks import CallbackList

from src.utils.config import DEFAULT_CONFIG_PATH, get_config
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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

TIMING_LOG_PATH = Path("logs/training_timing.jsonl")


def _get_git_commit() -> str:
    """Return the short HEAD commit hash of the current repository.

    Returns:
        Seven-character commit hash, or ``"unknown"`` on any failure.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _compute_data_fingerprint(splits_dir: str | Path) -> str:
    """Compute a content-based SHA-256 fingerprint of training parquet files.

    Streams all ``*_train.parquet`` files in sorted order into a single
    SHA-256 digest. Both the file name and file content are included so
    that renames and content changes each alter the fingerprint.

    Args:
        splits_dir: Directory containing the split parquet files.

    Returns:
        First 16 hex characters of the SHA-256 digest, or ``"empty"``
        when no matching files are found.
    """
    chunk_size = 1024 * 1024  # 1 MiB
    train_files = sorted(Path(splits_dir).glob("*_train.parquet"))
    if not train_files:
        return "empty"

    digest = hashlib.sha256()
    for file_path in train_files:
        digest.update(file_path.name.encode())
        with open(file_path, "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)

    return digest.hexdigest()[:16]


def _write_provenance(
    model_dir: Path,
    git_commit: str,
    config_source: str,
    experiment_id: str | None,
    created_utc: str,
    data_fingerprint: str = "",
) -> None:
    """Copy config snapshot and write provenance.json into model_dir.

    Args:
        model_dir: Destination directory for this experiment's models.
        git_commit: Short git commit hash at time of training.
        config_source: Resolved path to the config YAML that was used.
        experiment_id: EXPERIMENT_ID env value, or None if not set.
        created_utc: ISO-8601 UTC timestamp for when this run started.
        data_fingerprint: 16-hex-char SHA-256 of training parquet content.
    """
    # Config snapshot — copy2 preserves metadata
    config_dest = model_dir / "config_used.yml"
    try:
        shutil.copy2(config_source, config_dest)
    except (FileNotFoundError, OSError) as exc:
        logger.warning("Could not copy config to model_dir: %s", exc)

    provenance = {
        "git_commit": git_commit,
        "config_source": config_source,
        "experiment_id": experiment_id,
        "created_utc": created_utc,
        "data_fingerprint": data_fingerprint,
    }
    provenance_path = model_dir / "provenance.json"
    with open(provenance_path, "w") as fh:
        json.dump(provenance, fh, indent=2)

    logger.info("Provenance written to %s", provenance_path)


def _write_experiment_results(
    experiment_id: str,
    git_commit: str,
    config_path: str,
    n_seeds: int,
    n_folds: int,
    aggregate: dict[str, float],
    per_seed: dict[str, list[dict[str, float]]],
    model_dir: Path | None = None,
    seed_model_paths: dict[int, str | None] | None = None,
    data_fingerprint: str = "",
) -> Path:
    """Write cross-seed experiment results to a JSON file.

    Args:
        experiment_id: Unique identifier for this experiment run.
        git_commit: Short git commit hash at time of training.
        config_path: Resolved path to the config YAML used.
        n_seeds: Number of seeds trained.
        n_folds: Number of CV folds per seed.
        aggregate: Cross-seed aggregated metrics (mean/std per metric).
        per_seed: Dict mapping seed string to list of per-fold metric dicts.
        model_dir: Directory where models for this run were saved.
        seed_model_paths: Dict mapping seed int to best-model .zip path (or None).
        data_fingerprint: 16-hex-char SHA-256 of training parquet content.

    Returns:
        Path to the written JSON file.
    """
    output_dir = _PROJECT_ROOT / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_id}.json"

    models_map: dict[str, str | None] = {}
    if seed_model_paths:
        models_map = {str(seed): path for seed, path in seed_model_paths.items()}

    payload = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "config_path": config_path,
        "data_fingerprint": data_fingerprint,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "models": models_map,
        "n_seeds": n_seeds,
        "n_folds": n_folds,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }

    with open(output_path, "w") as file_handle:
        json.dump(payload, file_handle, indent=2)

    logger.info("Experiment results written to %s", output_path)
    return output_path


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
        # TD3 hyperparameters
        "lr_actor": str(config.td3.learning_rate.actor),
        "lr_critic": str(config.td3.learning_rate.critic),
        "lr_schedule": config.td3.lr_schedule,
        "gamma": str(config.td3.gamma),
        "tau": str(config.td3.tau),
        "batch_size": str(config.td3.batch_size),
        "buffer_size": str(config.td3.buffer_size),
        "learning_starts": str(config.td3.learning_starts),
        "train_freq": str(config.td3.train_freq),
        "policy_delay": str(config.td3.policy_delay),
        "target_noise_clip": str(config.td3.target_noise_clip),
        "target_policy_noise": str(config.td3.target_policy_noise),
        "action_noise_std": str(config.td3.action_noise_std),
        "total_timesteps": str(config.td3.total_timesteps),
        "net_arch_pi": str(config.td3.net_arch.pi),
        "net_arch_qf": str(config.td3.net_arch.qf),
        # Environment
        "window_size": str(config.environment.window_size),
        "reward_scaling": str(config.environment.reward_scaling),
        "max_position": str(config.environment.max_position),
        # Portfolio
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
    5. Optionally export results JSON when EXPERIMENT_ID env var is set.
    """
    experiment_id = os.environ.get("EXPERIMENT_ID")
    git_commit = _get_git_commit()
    run_started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    resolved_config_path = str(
        os.environ.get("TRADING_CONFIG") or DEFAULT_CONFIG_PATH
    )
    config = get_config()
    splits_dir = config.data.paths.splits
    symbols = config.symbols
    training_config = config.training

    logger.info("git_commit=%s config_path=%s", git_commit, resolved_config_path)

    # 1. Load data
    logger.info("Loading training data from %s", splits_dir)
    data = _load_train_data(splits_dir, symbols)
    data_fingerprint = _compute_data_fingerprint(splits_dir)
    logger.info("data_fingerprint=%s", data_fingerprint)

    # 2. Generate CV folds
    folds = generate_rolling_cv_folds(
        data,
        train_months=training_config.cv_train_months,
        validation_months=training_config.cv_validation_months,
    )
    logger.info("Generated %d CV folds", len(folds))

    if training_config.max_folds is not None:
        total_folds = len(folds)
        folds = folds[: training_config.max_folds]
        logger.info("Limiting to first %d of %d folds", len(folds), total_folds)

    # 3. Setup MLflow
    mlflow.set_tracking_uri(training_config.mlflow_tracking_uri)
    mlflow.set_experiment(training_config.mlflow_experiment_name)

    # 4. Create per-experiment model directory (never overwrites across runs)
    if experiment_id:
        model_dir = Path(training_config.model_dir) / experiment_id
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir = Path(training_config.model_dir) / f"run_{ts}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Write config snapshot + provenance into model_dir
    _write_provenance(
        model_dir=model_dir,
        git_commit=git_commit,
        config_source=resolved_config_path,
        experiment_id=experiment_id,
        created_utc=run_started_utc,
        data_fingerprint=data_fingerprint,
    )

    config_params = _flatten_config_params(config)
    all_seed_results = []
    per_seed_fold_metrics: dict[str, list[dict[str, float]]] = {}
    seed_model_paths: dict[int, str | None] = {}
    device = get_torch_device()
    total_folds_all_seeds = len(training_config.seeds) * len(folds)
    completed_folds = 0
    cumulative_fold_time = 0.0

    # 5. Train across seeds
    for seed_idx, seed in enumerate(training_config.seeds):
        logger.info("=== Training with seed=%d ===", seed)

        with mlflow.start_run(run_name=f"seed_{seed}"):
            # Log all hyperparameters up front
            mlflow.log_params(config_params)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_folds", len(folds))
            mlflow.log_param("git_commit", git_commit)
            mlflow.log_param("data_fingerprint", data_fingerprint)

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

            # Log fold metrics
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

        # Record best model path for this seed (None if never saved)
        seed_model_paths[seed] = best_model_path

        # Store per-seed fold metrics for experiment results export
        per_seed_fold_metrics[str(seed)] = fold_metrics

        # Aggregate fold metrics for this seed
        if fold_metrics:
            from .nodes import aggregate_fold_metrics as _agg_folds

            seed_summary = _agg_folds(fold_metrics)
            all_seed_results.append(seed_summary)

    # 6. Cross-seed summary
    final_summary: dict[str, float] = {}
    if all_seed_results:
        final_summary = aggregate_seed_results(all_seed_results)
        logger.info("=== Cross-seed summary ===")
        for key, value in final_summary.items():
            logger.info("  %s: %.6f", key, value)

    logger.info(
        "Training complete. Total wall time: %s",
        _format_elapsed(cumulative_fold_time),
    )

    # 7. Export experiment results JSON (always — model_dir is the unique key)
    effective_experiment_id = experiment_id or model_dir.name
    _write_experiment_results(
        experiment_id=effective_experiment_id,
        git_commit=git_commit,
        config_path=resolved_config_path,
        n_seeds=len(training_config.seeds),
        n_folds=len(folds),
        aggregate=final_summary,
        per_seed=per_seed_fold_metrics,
        model_dir=model_dir,
        seed_model_paths=seed_model_paths,
        data_fingerprint=data_fingerprint,
    )
