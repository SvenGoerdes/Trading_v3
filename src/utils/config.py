"""Configuration loader.

Reads conf/parameters.yml and provides typed access to settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "conf" / "parameters.yml"


@dataclass(frozen=True)
class MissingCandlePolicy:
    ffill_limit: int
    stale_flag_limit: int
    segment_split_beyond: int


@dataclass(frozen=True)
class DataPaths:
    raw: str
    intermediate: str
    features: str
    normalized: str
    splits: str


@dataclass(frozen=True)
class DataConfig:
    lookback_days: int
    missing_candle_policy: MissingCandlePolicy
    paths: DataPaths


@dataclass(frozen=True)
class MacdConfig:
    fast: int
    slow: int
    signal: int


@dataclass(frozen=True)
class IndicatorsConfig:
    rsi_window: int
    sma_window: int
    ema_window: int
    stochastic_window: int
    macd: MacdConfig
    roc_window: int
    williams_r_window: int
    disparity_sma_window: int
    ad: bool
    obv: bool


@dataclass(frozen=True)
class NormalizationConfig:
    method: str
    window: int


@dataclass(frozen=True)
class SplitConfig:
    test_ratio: float
    n_cv_folds: int


@dataclass(frozen=True)
class AppConfig:
    symbols: list[str]
    timeframe: str
    initial_balance: float
    trading_fee_pct: float
    slippage_pct: float
    data: DataConfig
    indicators: IndicatorsConfig
    normalization: NormalizationConfig
    split: SplitConfig


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file and return the raw dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def parse_config(raw: dict) -> AppConfig:
    """Construct an AppConfig from a raw dict."""
    data_raw = raw["data"]
    mcp = data_raw["missing_candle_policy"]
    paths = data_raw["paths"]

    ind_raw = raw["indicators"]
    macd_raw = ind_raw["macd"]

    return AppConfig(
        symbols=raw["symbols"],
        timeframe=raw["timeframe"],
        initial_balance=raw["initial_balance"],
        trading_fee_pct=raw["trading_fee_pct"],
        slippage_pct=raw["slippage_pct"],
        data=DataConfig(
            lookback_days=data_raw["lookback_days"],
            missing_candle_policy=MissingCandlePolicy(
                ffill_limit=mcp["ffill_limit"],
                stale_flag_limit=mcp["stale_flag_limit"],
                segment_split_beyond=mcp["segment_split_beyond"],
            ),
            paths=DataPaths(
                raw=paths["raw"],
                intermediate=paths["intermediate"],
                features=paths["features"],
                normalized=paths["normalized"],
                splits=paths["splits"],
            ),
        ),
        indicators=IndicatorsConfig(
            rsi_window=ind_raw["rsi_window"],
            sma_window=ind_raw["sma_window"],
            ema_window=ind_raw["ema_window"],
            stochastic_window=ind_raw["stochastic_window"],
            macd=MacdConfig(
                fast=macd_raw["fast"],
                slow=macd_raw["slow"],
                signal=macd_raw["signal"],
            ),
            roc_window=ind_raw["roc_window"],
            williams_r_window=ind_raw["williams_r_window"],
            disparity_sma_window=ind_raw["disparity_sma_window"],
            ad=ind_raw["ad"],
            obv=ind_raw["obv"],
        ),
        normalization=NormalizationConfig(
            method=raw["normalization"]["method"],
            window=raw["normalization"]["window"],
        ),
        split=SplitConfig(
            test_ratio=raw["split"]["test_ratio"],
            n_cv_folds=raw["split"]["n_cv_folds"],
        ),
    )


def get_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load YAML and return a typed AppConfig."""
    return parse_config(load_yaml(path))
