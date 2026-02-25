"""CLI runner for executing pipelines.

Usage:
    python -m src.run_pipeline --pipeline=<name>

Available pipelines:
    data_engineering    Fetch + clean OHLCV data from Binance
    feature_engineering Compute technical indicators + normalize
    training            TD3 training with Stable-Baselines3
    backtesting         Historical evaluation
    deployment          Live trading bot
"""

import argparse
import importlib
import sys

PIPELINES = [
    "data_engineering",
    "feature_engineering",
    "training",
    "backtesting",
    "deployment",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trading pipeline.")
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=PIPELINES,
        help="Name of the pipeline to run.",
    )
    args = parser.parse_args()

    module_path = f"src.pipelines.{args.pipeline}.pipeline"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        print(f"Error: Could not import pipeline '{module_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    module.run()


if __name__ == "__main__":
    main()
