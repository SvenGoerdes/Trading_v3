"""Feature Engineering nodes.

Computes technical indicators, normalises features via rolling z-score,
and creates temporal train/test splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils.config import IndicatorsConfig, NormalizationConfig, SplitConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Individual TI functions ──────────────────────────────────────────────


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """RSI via pandas-ta."""
    result = ta.rsi(close, length=window)
    result.name = f"rsi_{window}"
    return result


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    result = ta.sma(close, length=window)
    result.name = f"sma_{window}"
    return result


def compute_ema(close: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    result = ta.ema(close, length=window)
    result.name = f"ema_{window}"
    return result


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.DataFrame:
    """Stochastic oscillator (%K and %D)."""
    stoch = ta.stoch(high, low, close, k=window, d=3)
    result = pd.DataFrame(index=stoch.index)
    result["stoch_k"] = stoch.iloc[:, 0]
    result["stoch_d"] = stoch.iloc[:, 1]
    return result


def compute_macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """MACD line and signal line."""
    result = ta.macd(close, fast=fast, slow=slow, signal=signal)
    result.columns = ["macd", "macd_histogram", "macd_signal"]
    return result[["macd", "macd_signal"]]


def compute_accumulation_distribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Accumulation/Distribution line."""
    result = ta.ad(high, low, close, volume)
    result.name = "ad"
    return result


def compute_on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    result = ta.obv(close, volume)
    result.name = "obv"
    return result


def compute_rate_of_change(close: pd.Series, window: int) -> pd.Series:
    """Rate of Change."""
    result = ta.roc(close, length=window)
    result.name = f"roc_{window}"
    return result


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Williams %R."""
    result = ta.willr(high, low, close, length=window)
    result.name = f"williams_r_{window}"
    return result


def compute_disparity_index(close: pd.Series, sma_window: int) -> pd.Series:
    """Disparity Index: ((close - SMA) / SMA) * 100."""
    sma = ta.sma(close, length=sma_window)
    disparity = ((close - sma) / sma) * 100
    disparity.name = f"disparity_{sma_window}"
    return disparity


# ── Orchestration ────────────────────────────────────────────────────────


def compute_technical_indicators(
    clean_data: dict[str, pd.DataFrame],
    ti_config: IndicatorsConfig,
) -> dict[str, pd.DataFrame]:
    """Compute all TIs for every symbol.

    Drops warmup NaN rows and returns enriched DataFrames.
    Does NOT save to disk — the caller (pipeline) handles that.
    """
    result: dict[str, pd.DataFrame] = {}

    for symbol, df in clean_data.items():
        if df.empty:
            result[symbol] = df
            continue

        logger.info("Computing TIs for %s", symbol)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        indicators = [
            compute_rsi(close, ti_config.rsi_window),
            compute_sma(close, ti_config.sma_window),
            compute_ema(close, ti_config.ema_window),
            compute_stochastic(high, low, close, ti_config.stochastic_window),
            compute_macd(close, ti_config.macd.fast, ti_config.macd.slow, ti_config.macd.signal),
            compute_accumulation_distribution(high, low, close, volume),
            compute_on_balance_volume(close, volume),
            compute_rate_of_change(close, ti_config.roc_window),
            compute_williams_r(high, low, close, ti_config.williams_r_window),
            compute_disparity_index(close, ti_config.disparity_sma_window),
        ]

        for ind in indicators:
            if isinstance(ind, pd.DataFrame):
                for col in ind.columns:
                    df[col] = ind[col]
            else:
                df[ind.name] = ind

        df = df.dropna().copy()
        logger.info("  -> %d rows after warmup drop", len(df))
        result[symbol] = df

    return result


def save_features(features: dict[str, pd.DataFrame], features_dir: str) -> None:
    """Save feature DataFrames to Parquet."""
    out_dir = Path(features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol, df in features.items():
        safe_name = symbol.replace("/", "")
        df.to_parquet(out_dir / f"{safe_name}_features.parquet")


TI_COLUMNS = [
    "rsi_14",
    "sma_20",
    "ema_20",
    "stoch_k",
    "stoch_d",
    "macd",
    "macd_signal",
    "ad",
    "obv",
    "roc_10",
    "williams_r_14",
    "disparity_20",
]


def normalize_features(
    features: dict[str, pd.DataFrame],
    norm_config: NormalizationConfig,
) -> dict[str, pd.DataFrame]:
    """Apply rolling z-score normalization to TI columns.

    Creates `{col}_norm` columns while keeping originals.
    Drops the first `window` rows where z-score is unreliable.
    """
    window = norm_config.window
    result: dict[str, pd.DataFrame] = {}

    for symbol, df in features.items():
        if df.empty:
            result[symbol] = df
            continue

        logger.info("Normalizing %s", symbol)
        df = df.copy()

        for col in TI_COLUMNS:
            if col not in df.columns:
                continue
            series = df[col].astype(float)
            rolling_mean = series.rolling(window, min_periods=1).mean()
            rolling_std = series.rolling(window, min_periods=1).std()
            z = (series - rolling_mean) / rolling_std
            z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            df[f"{col}_norm"] = z

        df = df.iloc[window:]
        logger.info("  -> %d rows after normalization warmup drop", len(df))
        result[symbol] = df

    return result


def save_normalized(normalized: dict[str, pd.DataFrame], normalized_dir: str) -> None:
    """Save normalized DataFrames to Parquet."""
    out_dir = Path(normalized_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol, df in normalized.items():
        safe_name = symbol.replace("/", "")
        df.to_parquet(out_dir / f"{safe_name}_normalized.parquet")


def create_train_test_split(
    normalized: dict[str, pd.DataFrame],
    split_config: SplitConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[list[tuple[int, int]]]]:
    """Temporal train/test split with walk-forward CV folds.

    Args:
        normalized: Dict of symbol to normalized DataFrame.
        split_config: Contains test_ratio and n_cv_folds.

    Returns:
        (train_dict, test_dict, cv_folds) where cv_folds is a list of
        (train_end_idx, val_start_idx) tuples per fold.
    """
    train_dict: dict[str, pd.DataFrame] = {}
    test_dict: dict[str, pd.DataFrame] = {}
    cv_folds: list[list[tuple[int, int]]] = []

    for symbol, df in normalized.items():
        if df.empty:
            train_dict[symbol] = df
            test_dict[symbol] = df
            continue

        n = len(df)
        split_idx = int(n * (1 - split_config.test_ratio))
        train_dict[symbol] = df.iloc[:split_idx].copy()
        test_dict[symbol] = df.iloc[split_idx:].copy()

        folds = []
        train_n = split_idx
        fold_size = train_n // (split_config.n_cv_folds + 1)
        for i in range(split_config.n_cv_folds):
            train_end = fold_size * (i + 1)
            val_start = train_end
            val_end = min(train_end + fold_size, train_n)
            folds.append((train_end, val_start, val_end))
        cv_folds.append(folds)

    return train_dict, test_dict, cv_folds


def save_splits(
    train: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    splits_dir: str,
) -> None:
    """Save train/test DataFrames to Parquet."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol, df in train.items():
        safe_name = symbol.replace("/", "")
        df.to_parquet(out_dir / f"{safe_name}_train.parquet")
    for symbol, df in test.items():
        safe_name = symbol.replace("/", "")
        df.to_parquet(out_dir / f"{safe_name}_test.parquet")
