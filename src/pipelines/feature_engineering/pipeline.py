"""Feature Engineering Pipeline.

Computes technical indicators, normalizes features via rolling z-score,
and creates temporal train/test splits.
"""

from pathlib import Path

import pandas as pd

from src.pipelines.feature_engineering.nodes import (
    compute_technical_indicators,
    create_train_test_split,
    normalize_features,
    save_features,
    save_normalized,
    save_splits,
)
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logging


def run() -> None:
    """Execute the feature engineering pipeline."""
    setup_logging()
    logger = get_logger(__name__)
    config = get_config()

    logger.info("Feature Engineering Pipeline")

    intermediate_dir = Path(config.data.paths.intermediate)
    clean_data: dict[str, pd.DataFrame] = {}
    for symbol in config.symbols:
        safe_name = symbol.replace("/", "")
        path = intermediate_dir / f"{safe_name}_clean.parquet"
        if not path.exists():
            logger.warning("Missing intermediate file for %s, skipping", symbol)
            continue
        clean_data[symbol] = pd.read_parquet(path)

    if not clean_data:
        logger.error("No intermediate data found. Run data_engineering first.")
        return

    features = compute_technical_indicators(clean_data, config.indicators)
    save_features(features, config.data.paths.features)

    normalized = normalize_features(features, config.normalization)
    save_normalized(normalized, config.data.paths.normalized)

    train, test, cv_folds = create_train_test_split(normalized, config.split)
    save_splits(train, test, config.data.paths.splits)

    for symbol in train:
        logger.info(
            "%s — train: %d, test: %d",
            symbol,
            len(train[symbol]),
            len(test[symbol]),
        )

    logger.info("Feature engineering complete.")
