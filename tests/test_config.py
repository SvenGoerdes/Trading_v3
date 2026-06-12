"""Tests for configuration loading."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from src.pipelines.training.pipeline import (
    _compute_data_fingerprint,
    _write_experiment_results,
    _write_provenance,
)
from src.utils.config import (
    AppConfig,
    DataConfig,
    EnvironmentConfig,
    IndicatorsConfig,
    MissingCandlePolicy,
    NetArchConfig,
    NormalizationConfig,
    SplitConfig,
    TD3Config,
    TrainingConfig,
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
        "td3": {
            "learning_rate": {"actor": 0.0001, "critic": 0.0003},
            "lr_schedule": "linear",
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "buffer_size": 100000,
            "learning_starts": 10000,
            "train_freq": 1,
            "policy_delay": 2,
            "target_noise_clip": 0.5,
            "target_policy_noise": 0.2,
            "action_noise_std": 0.1,
            "total_timesteps": 1000000,
            "net_arch": {"pi": [400, 300], "qf": [400, 300]},
        },
        "environment": {
            "window_size": 50,
            "reward_scaling": 1.0,
            "max_position": 1.0,
        },
        "training": {
            "seeds": [42, 123],
            "cv_train_months": 3,
            "cv_validation_months": 1,
            "model_dir": "models",
            "mlflow_experiment_name": "test_experiment",
            "mlflow_tracking_uri": "mlruns",
            "progress_log_every_steps": 10000,
            "max_folds": None,
        },
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
        assert isinstance(config.td3, TD3Config)
        assert isinstance(config.td3.net_arch, NetArchConfig)
        assert isinstance(config.environment, EnvironmentConfig)
        assert isinstance(config.training, TrainingConfig)
        assert config.data.missing_candle_policy.ffill_limit == 3
        assert config.indicators.macd.fast == 12
        assert config.td3.learning_rate.actor == 0.0001
        assert config.td3.learning_rate.critic == 0.0003
        assert config.td3.net_arch.pi == [400, 300]
        assert config.environment.window_size == 50
        assert config.training.seeds == [42, 123]

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

    def test_trading_config_env_var_redirects_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid_yaml: Path
    ) -> None:
        """TRADING_CONFIG env var should override the default config path."""
        # Write a copy of valid_yaml with a modified initial_balance
        raw = load_yaml(valid_yaml)
        raw["initial_balance"] = 99999.0
        custom_path = tmp_path / "custom_parameters.yml"
        custom_path.write_text(yaml.dump(raw))

        monkeypatch.setenv("TRADING_CONFIG", str(custom_path))
        config = get_config()
        assert config.initial_balance == pytest.approx(99999.0)

    def test_explicit_path_takes_priority_over_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid_yaml: Path
    ) -> None:
        """An explicit path argument must win over TRADING_CONFIG."""
        raw = load_yaml(valid_yaml)
        raw["initial_balance"] = 11111.0
        env_path = tmp_path / "env_params.yml"
        env_path.write_text(yaml.dump(raw))

        monkeypatch.setenv("TRADING_CONFIG", str(env_path))
        # Pass valid_yaml explicitly — its balance is 10000.0
        config = get_config(path=valid_yaml)
        assert config.initial_balance == pytest.approx(10000.0)


class TestMaxFolds:
    def test_max_folds_parses_as_int_when_set(self, valid_yaml: Path) -> None:
        """max_folds should be an int when explicitly set in YAML."""
        raw = load_yaml(valid_yaml)
        raw["training"]["max_folds"] = 3
        config = parse_config(raw)
        assert config.training.max_folds == 3

    def test_max_folds_is_none_when_null(self, valid_yaml: Path) -> None:
        """max_folds should be None when set to null in YAML."""
        raw = load_yaml(valid_yaml)
        raw["training"]["max_folds"] = None
        config = parse_config(raw)
        assert config.training.max_folds is None

    def test_max_folds_is_none_when_missing(self, valid_yaml: Path) -> None:
        """max_folds should be None when the key is absent."""
        raw = load_yaml(valid_yaml)
        raw["training"].pop("max_folds", None)
        config = parse_config(raw)
        assert config.training.max_folds is None


class TestWriteExperimentResults:
    def test_writes_correct_json_structure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_write_experiment_results should produce a valid, complete JSON file."""
        import json

        # Redirect PROJECT_ROOT so the helper writes into tmp_path
        import src.pipelines.training.pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "_PROJECT_ROOT", tmp_path)

        per_seed: dict[str, list[dict[str, float]]] = {
            "42": [{"sharpe_ratio": 1.2, "total_return": 0.05}],
            "123": [{"sharpe_ratio": 0.8, "total_return": 0.02}],
        }
        aggregate = {"mean_sharpe_ratio": 1.0, "std_sharpe_ratio": 0.2}

        output_path = _write_experiment_results(
            experiment_id="test_exp_001",
            git_commit="abc1234",
            config_path="/conf/parameters.yml",
            n_seeds=2,
            n_folds=1,
            aggregate=aggregate,
            per_seed=per_seed,
            data_fingerprint="abcdef1234567890",
        )

        assert output_path.exists()
        with open(output_path) as file_handle:
            payload = json.load(file_handle)

        assert payload["experiment_id"] == "test_exp_001"
        assert payload["git_commit"] == "abc1234"
        assert payload["config_path"] == "/conf/parameters.yml"
        assert payload["n_seeds"] == 2
        assert payload["n_folds"] == 1
        assert payload["aggregate"] == aggregate
        assert payload["per_seed"] == per_seed
        assert "timestamp" in payload
        assert payload["data_fingerprint"] == "abcdef1234567890"

    def test_includes_model_dir_and_models_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """model_dir and models keys must be present in the output JSON."""
        import json

        import src.pipelines.training.pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "_PROJECT_ROOT", tmp_path)

        model_dir = tmp_path / "models" / "run_20240101_120000"
        model_dir.mkdir(parents=True)

        seed_model_paths = {
            42: str(model_dir / "td3_seed42_best.zip"),
            123: None,
        }
        per_seed: dict[str, list[dict[str, float]]] = {
            "42": [{"sharpe_ratio": 1.5}],
            "123": [{"sharpe_ratio": 0.5}],
        }

        output_path = _write_experiment_results(
            experiment_id="test_exp_002",
            git_commit="def5678",
            config_path="/conf/parameters.yml",
            n_seeds=2,
            n_folds=1,
            aggregate={"mean_sharpe_ratio": 1.0},
            per_seed=per_seed,
            model_dir=model_dir,
            seed_model_paths=seed_model_paths,
        )

        with open(output_path) as fh:
            payload = json.load(fh)

        assert payload["model_dir"] == str(model_dir)
        assert "models" in payload
        assert payload["models"]["42"] == str(model_dir / "td3_seed42_best.zip")
        assert payload["models"]["123"] is None

    def test_model_dir_none_when_not_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """model_dir and models must be null/empty when not supplied."""
        import json

        import src.pipelines.training.pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "_PROJECT_ROOT", tmp_path)

        output_path = _write_experiment_results(
            experiment_id="test_exp_003",
            git_commit="aaa0000",
            config_path="/conf/parameters.yml",
            n_seeds=1,
            n_folds=1,
            aggregate={},
            per_seed={},
        )

        with open(output_path) as fh:
            payload = json.load(fh)

        assert payload["model_dir"] is None
        assert payload["models"] == {}


class TestWriteProvenance:
    def test_creates_provenance_json(self, tmp_path: Path) -> None:
        """provenance.json must contain all required fields."""
        import json

        model_dir = tmp_path / "models" / "run_abc"
        model_dir.mkdir(parents=True)

        # Create a dummy config file to copy
        config_src = tmp_path / "parameters.yml"
        config_src.write_text("symbols: [BTC/USDT]\n")

        _write_provenance(
            model_dir=model_dir,
            git_commit="abc1234",
            config_source=str(config_src),
            experiment_id="exp_42",
            created_utc="2024-01-01T12:00:00+00:00",
            data_fingerprint="deadbeef01234567",
        )

        provenance_path = model_dir / "provenance.json"
        assert provenance_path.exists()

        with open(provenance_path) as fh:
            data = json.load(fh)

        assert data["git_commit"] == "abc1234"
        assert data["config_source"] == str(config_src)
        assert data["experiment_id"] == "exp_42"
        assert data["created_utc"] == "2024-01-01T12:00:00+00:00"
        assert data["data_fingerprint"] == "deadbeef01234567"

    def test_copies_config_yaml(self, tmp_path: Path) -> None:
        """config_used.yml must be a copy of the source config."""
        model_dir = tmp_path / "models" / "run_abc"
        model_dir.mkdir(parents=True)

        config_src = tmp_path / "parameters.yml"
        config_src.write_text("initial_balance: 99999\n")

        _write_provenance(
            model_dir=model_dir,
            git_commit="abc1234",
            config_source=str(config_src),
            experiment_id=None,
            created_utc="2024-01-01T12:00:00+00:00",
        )

        config_dest = model_dir / "config_used.yml"
        assert config_dest.exists()
        assert config_dest.read_text() == "initial_balance: 99999\n"

    def test_experiment_id_none_allowed(self, tmp_path: Path) -> None:
        """experiment_id=None must be serialised as null in provenance.json."""
        import json

        model_dir = tmp_path / "models" / "run_ts"
        model_dir.mkdir(parents=True)

        config_src = tmp_path / "parameters.yml"
        config_src.write_text("symbols: []\n")

        _write_provenance(
            model_dir=model_dir,
            git_commit="xyz9999",
            config_source=str(config_src),
            experiment_id=None,
            created_utc="2024-06-01T00:00:00+00:00",
        )

        with open(model_dir / "provenance.json") as fh:
            data = json.load(fh)

        assert data["experiment_id"] is None

    def test_missing_config_source_logs_warning_no_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing config source must not raise; a warning must be emitted."""
        import logging

        model_dir = tmp_path / "models" / "run_warn"
        model_dir.mkdir(parents=True)

        with caplog.at_level(logging.WARNING, logger="src.pipelines.training.pipeline"):
            _write_provenance(
                model_dir=model_dir,
                git_commit="aaa0000",
                config_source="/nonexistent/path/parameters.yml",
                experiment_id=None,
                created_utc="2024-01-01T00:00:00+00:00",
            )

        # provenance.json should still be written
        assert (model_dir / "provenance.json").exists()
        # config_used.yml should NOT exist (copy failed)
        assert not (model_dir / "config_used.yml").exists()
        # Warning must have been emitted
        assert any("Could not copy config" in r.message for r in caplog.records)


class TestRebalanceThresholdConfig:
    """Tests for the new rebalance_threshold field in EnvironmentConfig."""

    def test_default_zero_when_absent(self, valid_yaml: Path) -> None:
        """rebalance_threshold defaults to 0.0 when not in YAML."""
        raw = load_yaml(valid_yaml)
        # The fixture does not include rebalance_threshold — it must default to 0.0
        assert "rebalance_threshold" not in raw["environment"]
        config = parse_config(raw)
        assert config.environment.rebalance_threshold == pytest.approx(0.0)

    def test_parsed_when_present(self, valid_yaml: Path) -> None:
        """rebalance_threshold is correctly parsed when present in YAML."""
        raw = load_yaml(valid_yaml)
        raw["environment"]["rebalance_threshold"] = 0.05
        config = parse_config(raw)
        assert config.environment.rebalance_threshold == pytest.approx(0.05)

    def test_immutable(self, valid_yaml: Path) -> None:
        """rebalance_threshold field must be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        config = parse_config(load_yaml(valid_yaml))
        with pytest.raises(FrozenInstanceError):
            config.environment.rebalance_threshold = 0.1  # type: ignore[misc]


class TestTurnoverPenaltyCoefConfig:
    """Tests for the turnover_penalty_coef field in EnvironmentConfig."""

    def test_default_zero_when_absent(self, valid_yaml: Path) -> None:
        """turnover_penalty_coef defaults to 0.0 when not in YAML."""
        raw = load_yaml(valid_yaml)
        # The fixture does not include turnover_penalty_coef — must default to 0.0
        assert "turnover_penalty_coef" not in raw["environment"]
        config = parse_config(raw)
        assert config.environment.turnover_penalty_coef == pytest.approx(0.0)

    def test_parsed_when_present(self, valid_yaml: Path) -> None:
        """turnover_penalty_coef is correctly parsed when present in YAML."""
        raw = load_yaml(valid_yaml)
        raw["environment"]["turnover_penalty_coef"] = 0.05
        config = parse_config(raw)
        assert config.environment.turnover_penalty_coef == pytest.approx(0.05)

    def test_immutable(self, valid_yaml: Path) -> None:
        """turnover_penalty_coef field must be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        config = parse_config(load_yaml(valid_yaml))
        with pytest.raises(FrozenInstanceError):
            config.environment.turnover_penalty_coef = 0.1  # type: ignore[misc]


class TestCrossSectionalMomentumConfig:
    """Tests for cross_sectional_momentum and momentum_window in EnvironmentConfig."""

    def test_defaults_when_absent(self, valid_yaml: Path) -> None:
        """cross_sectional_momentum defaults to False and momentum_window to 12 when absent."""
        raw = load_yaml(valid_yaml)
        assert "cross_sectional_momentum" not in raw["environment"]
        assert "momentum_window" not in raw["environment"]
        config = parse_config(raw)
        assert config.environment.cross_sectional_momentum is False
        assert config.environment.momentum_window == 12

    def test_cross_sectional_momentum_parsed_when_present(self, valid_yaml: Path) -> None:
        """cross_sectional_momentum is correctly parsed when True in YAML."""
        raw = load_yaml(valid_yaml)
        raw["environment"]["cross_sectional_momentum"] = True
        raw["environment"]["momentum_window"] = 6
        config = parse_config(raw)
        assert config.environment.cross_sectional_momentum is True
        assert config.environment.momentum_window == 6

    def test_immutable(self, valid_yaml: Path) -> None:
        """cross_sectional_momentum and momentum_window fields are immutable."""
        from dataclasses import FrozenInstanceError

        config = parse_config(load_yaml(valid_yaml))
        with pytest.raises(FrozenInstanceError):
            config.environment.cross_sectional_momentum = True  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            config.environment.momentum_window = 99  # type: ignore[misc]


class TestComputeDataFingerprint:
    """Tests for _compute_data_fingerprint."""

    def _write_parquet(self, path: Path) -> None:
        """Write a minimal parquet file at *path*."""
        import pandas as pd

        pd.DataFrame({"a": [1, 2]}).to_parquet(path)

    def test_deterministic(self, tmp_path: Path) -> None:
        """Same directory state must yield the same fingerprint on two calls."""
        self._write_parquet(tmp_path / "AAA_train.parquet")
        self._write_parquet(tmp_path / "BBB_train.parquet")

        first = _compute_data_fingerprint(tmp_path)
        second = _compute_data_fingerprint(tmp_path)

        assert first == second

    def test_content_change_alters_fingerprint(self, tmp_path: Path) -> None:
        """Changing a file's content must produce a different fingerprint."""
        import pandas as pd

        train_file = tmp_path / "AAA_train.parquet"
        pd.DataFrame({"a": [1, 2]}).to_parquet(train_file)
        fp_before = _compute_data_fingerprint(tmp_path)

        # Overwrite with different content
        pd.DataFrame({"a": [99, 100]}).to_parquet(train_file)
        fp_after = _compute_data_fingerprint(tmp_path)

        assert fp_before != fp_after

    def test_test_parquet_not_included(self, tmp_path: Path) -> None:
        """A *_test.parquet file must not affect the fingerprint."""
        import pandas as pd

        self._write_parquet(tmp_path / "AAA_train.parquet")
        fp_without_test = _compute_data_fingerprint(tmp_path)

        # Adding a _test file must leave fingerprint unchanged
        pd.DataFrame({"a": [9, 8, 7]}).to_parquet(tmp_path / "AAA_test.parquet")
        fp_with_test = _compute_data_fingerprint(tmp_path)

        assert fp_without_test == fp_with_test

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        """Directory with no *_train.parquet files must return ``'empty'``."""
        result = _compute_data_fingerprint(tmp_path)
        assert result == "empty"

    def test_result_is_16_hex_chars(self, tmp_path: Path) -> None:
        """Fingerprint for non-empty dir must be exactly 16 lowercase hex chars."""
        self._write_parquet(tmp_path / "AAA_train.parquet")
        fp = _compute_data_fingerprint(tmp_path)

        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)


class TestAllocationModeConfig:
    """Tests for allocation_mode field in EnvironmentConfig."""

    def test_default_renormalize_when_absent(self, valid_yaml: Path) -> None:
        """allocation_mode defaults to 'renormalize' when not present in YAML."""
        raw = load_yaml(valid_yaml)
        assert "allocation_mode" not in raw["environment"]
        config = parse_config(raw)
        assert config.environment.allocation_mode == "renormalize"

    def test_parsed_when_present_renormalize(self, valid_yaml: Path) -> None:
        """allocation_mode='renormalize' is correctly parsed when present."""
        raw = load_yaml(valid_yaml)
        raw["environment"]["allocation_mode"] = "renormalize"
        config = parse_config(raw)
        assert config.environment.allocation_mode == "renormalize"

    def test_parsed_when_present_scaled(self, valid_yaml: Path) -> None:
        """allocation_mode='scaled' is correctly parsed when present."""
        raw = load_yaml(valid_yaml)
        raw["environment"]["allocation_mode"] = "scaled"
        config = parse_config(raw)
        assert config.environment.allocation_mode == "scaled"

    def test_immutable(self, valid_yaml: Path) -> None:
        """allocation_mode field must be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        config = parse_config(load_yaml(valid_yaml))
        with pytest.raises(FrozenInstanceError):
            config.environment.allocation_mode = "scaled"  # type: ignore[misc]
