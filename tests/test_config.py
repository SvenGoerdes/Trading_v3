"""Tests for configuration loading."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    AppConfig,
    DataConfig,
    IndicatorsConfig,
    MissingCandlePolicy,
    NormalizationConfig,
    SplitConfig,
    get_config,
    load_yaml,
    parse_config,
)


@pytest.fixture()
def valid_yaml(tmp_path: Path) -> Path:
    """Create a minimal valid parameters.yml for testing."""
    data = {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "5m",
        "initial_balance": 10000.0,
        "trading_fee_pct": 0.001,
        "slippage_pct": 0.0005,
        "data": {
            "lookback_days": 120,
            "missing_candle_policy": {
                "ffill_limit": 3,
                "stale_flag_limit": 20,
                "segment_split_beyond": 20,
            },
            "paths": {
                "raw": "data/01_raw",
                "intermediate": "data/02_intermediate",
                "features": "data/03_features",
                "normalized": "data/04_normalized",
                "splits": "data/05_splits",
            },
        },
        "indicators": {
            "rsi_window": 14,
            "sma_window": 20,
            "ema_window": 20,
            "stochastic_window": 14,
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "roc_window": 10,
            "williams_r_window": 14,
            "disparity_sma_window": 20,
            "ad": True,
            "obv": True,
        },
        "normalization": {"method": "rolling_zscore", "window": 500},
        "split": {"test_ratio": 0.2, "n_cv_folds": 5},
    }
    path = tmp_path / "parameters.yml"
    path.write_text(yaml.dump(data))
    return path


class TestLoadYaml:
    def test_loads_valid_file(self, valid_yaml: Path) -> None:
        raw = load_yaml(valid_yaml)
        assert isinstance(raw, dict)
        assert "symbols" in raw

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yml")

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yml"
        path.write_text("{invalid: yaml: [}")
        with pytest.raises(yaml.YAMLError):
            load_yaml(path)


class TestParseConfig:
    def test_returns_app_config(self, valid_yaml: Path) -> None:
        raw = load_yaml(valid_yaml)
        config = parse_config(raw)
        assert isinstance(config, AppConfig)
        assert config.timeframe == "5m"
        assert len(config.symbols) == 2

    def test_nested_types(self, valid_yaml: Path) -> None:
        config = parse_config(load_yaml(valid_yaml))
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.data.missing_candle_policy, MissingCandlePolicy)
        assert isinstance(config.indicators, IndicatorsConfig)
        assert isinstance(config.normalization, NormalizationConfig)
        assert isinstance(config.split, SplitConfig)
        assert config.data.missing_candle_policy.ffill_limit == 3
        assert config.indicators.macd.fast == 12

    def test_missing_key_raises(self, valid_yaml: Path) -> None:
        raw = load_yaml(valid_yaml)
        del raw["symbols"]
        with pytest.raises(KeyError):
            parse_config(raw)


class TestFrozenDataclass:
    def test_config_is_immutable(self, valid_yaml: Path) -> None:
        config = parse_config(load_yaml(valid_yaml))
        with pytest.raises(FrozenInstanceError):
            config.timeframe = "1h"  # type: ignore[misc]

    def test_nested_is_immutable(self, valid_yaml: Path) -> None:
        config = parse_config(load_yaml(valid_yaml))
        with pytest.raises(FrozenInstanceError):
            config.data.lookback_days = 999  # type: ignore[misc]


class TestGetConfig:
    def test_loads_real_config(self) -> None:
        config = get_config()
        assert isinstance(config, AppConfig)
        assert "BTC/USDT" in config.symbols
