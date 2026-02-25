.PHONY: setup test lint train backtest

setup:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

train:
	uv run python -m src.run_pipeline --pipeline=training

backtest:
	uv run python -m src.run_pipeline --pipeline=backtesting
