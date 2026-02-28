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
class NetArchConfig:
    pi: list[int]
    qf: list[int]


@dataclass(frozen=True)
class TD3Config:
    learning_rate: float
    gamma: float
    tau: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    train_freq: int
    policy_delay: int
    target_noise_clip: float
    target_policy_noise: float
    action_noise_std: float
    total_timesteps: int
    net_arch: NetArchConfig


@dataclass(frozen=True)
class EnvironmentConfig:
    window_size: int
    reward_scaling: float
    max_position: float


@dataclass(frozen=True)
class TrainingConfig:
    seeds: list[int]
    cv_train_months: int
    cv_validation_months: int
    model_dir: str
    mlflow_experiment_name: str
    mlflow_tracking_uri: str
    progress_log_every_steps: int


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
    td3: TD3Config
    environment: EnvironmentConfig
    training: TrainingConfig


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
        td3=TD3Config(
            learning_rate=raw["td3"]["learning_rate"],
            gamma=raw["td3"]["gamma"],
            tau=raw["td3"]["tau"],
            batch_size=raw["td3"]["batch_size"],
            buffer_size=raw["td3"]["buffer_size"],
            learning_starts=raw["td3"]["learning_starts"],
            train_freq=raw["td3"]["train_freq"],
            policy_delay=raw["td3"]["policy_delay"],
            target_noise_clip=raw["td3"]["target_noise_clip"],
            target_policy_noise=raw["td3"]["target_policy_noise"],
            action_noise_std=raw["td3"]["action_noise_std"],
            total_timesteps=raw["td3"]["total_timesteps"],
            net_arch=NetArchConfig(
                pi=raw["td3"]["net_arch"]["pi"],
                qf=raw["td3"]["net_arch"]["qf"],
            ),
        ),
        environment=EnvironmentConfig(
            window_size=raw["environment"]["window_size"],
            reward_scaling=raw["environment"]["reward_scaling"],
            max_position=raw["environment"]["max_position"],
        ),
        training=TrainingConfig(
            seeds=raw["training"]["seeds"],
            cv_train_months=raw["training"]["cv_train_months"],
            cv_validation_months=raw["training"]["cv_validation_months"],
            model_dir=raw["training"]["model_dir"],
            mlflow_experiment_name=raw["training"]["mlflow_experiment_name"],
            mlflow_tracking_uri=raw["training"]["mlflow_tracking_uri"],
            progress_log_every_steps=raw["training"]["progress_log_every_steps"],
        ),
    )


def get_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load YAML and return a typed AppConfig."""
    return parse_config(load_yaml(path))
