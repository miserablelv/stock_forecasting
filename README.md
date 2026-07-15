# Stock Predictor

This project compares Machine Learning methods for S&P 500 weekly price forecasting, evaluated through automated trading strategy backtesting.

## Context

This project was developed as my Bachelor's thesis in Computer Science and defended in January 2025.

## Overview

This project implements and compares multiple forecasting models (LSTM, GRU, RNN, CNN, SARIMA, XGBoost, ARNN) for predicting S&P 500 weekly open prices. Models are optimised via Bayesian hyperparameter search (Hyperopt) and evaluated through backtested trading strategies using the PyAlgoTrade library.

## Project Structure

```
stock_predictor/
├── src/                       # Core Python package
│   ├── config.py              # Device & loss function registry
│   ├── config/                # Model & strategy config definitions
│   │   ├── model_config.py
│   │   └── strategy_config.py
│   ├── data/                  # Data loading & feature engineering
│   ├── models/                # DL, ML, statistical, naive & hybrid models
│   ├── optimization/          # Bayesian hyperparameter search
│   ├── trading/               # Backtesting strategies & performance metrics
│   ├── evaluation/            # Aggregation & visualisation
│   └── persistence/           # Model save/load utilities
├── notebooks/                 # Walkthrough notebook
├── data/                      # Data files (git-ignored)
│   ├── raw/                   # Immutable downloaded data
│   ├── processed/             # Train/val/test splits
│   └── external/              # Third-party datasets (i.e. AAII sentiment index)
├── datasets/                  # Prepared dataset artefacts (git-ignored)
├── models/                    # Trained model artefacts (git-ignored)
├── best_models/               # Best checkpoints per model type (git-ignored)
├── best_strategies/           # Best backtested strategy configs (git-ignored)
├── best_aggregations/         # Best ensemble aggregation results (git-ignored)
├── images/                    # Exported figures & images (git-ignored)
└── tests/                     # Unit tests (not yet implemented)
```

## Requirements

- Python ≥ 3.10
- TA-Lib system library (see note below)

## Setup

```bash
# Clone and install in editable mode
git clone https://github.com/raldanondo/stock_price_forecasting.git
cd stock_price_forecasting

# With pip
pip install -e ".[dev]"

# Or with uv (recommended)
uv sync
```

> **Note:** TA-Lib cannot be installed just by using pip and requires a system-level installation. See [TA-Lib docs](https://ta-lib.github.io/ta-lib-python/install.html).

## Usage

The primary workflow is through the Jupyter notebook `notebooks/main.ipynb`.

## Models

| Type | Models |
|------|--------|
| Deep Learning | LSTM, GRU, RNN, CNN |
| Machine Learning | XGBoost, ARNN |
| Statistical | SARIMA |
| Naive baselines | Drift, LastKMedian |

## Data

The system uses weekly S&P 500 data enriched with:
- VIX (volatility index), DXY (dollar index)
- AAII sentiment survey (bullish/bearish/neutral)
- VWAP, WMA technical indicators

Preprocessing included:
- Transformation to log scale
- Cyclical month encoding (sin/cos) for month
- Feature scaling (StandardScaler, MinMaxScaler, or PowerTransformer — hyperparameter)

Data is split into five non-overlapping sequential segments using a 52 / 12 / 12 / 12 / 12 % ratio:
train → val 1 (model selection) → val 2 (strategy selection) → val 3 (ensemble selection) → test

## Evaluation

Evaluation is structured in three sequential stages, each with its own dedicated validation split.

### 1. Forecasting accuracy (val 1)

Models are first assessed on point-forecast quality using the loss function selected during hyperparameter search (RMSE, MAE, Smooth L1, or Huber). The best hyperparameter configuration per model is retained based on this validation loss.

### 2. Trading strategy backtesting (val 2)

Each model's predictions are fed into a [`PredictionBasedStrategy`](src/trading/strategies.py) implemented with PyAlgoTrade (initial capital: $100,000). The strategy decides to enter long, enter short, or wait based on:

- **Prediction-only mode**: enters if the predicted price change exceeds a tunable threshold (fixed or ATR-scaled)
- **Prediction + indicators mode**: combines predictions with RSI crossovers, MACD crossovers, and OBV trend confirmation

Strategy hyperparameters (threshold type, ATR multiplier, short selling, indicator periods for RSI/MACD/ATR/OBV) are also optimised via Bayesian search. Strategies are ranked by the following metrics computed in [`metrics.py`](src/trading/metrics.py):

| Metric | Description |
|---|---|
| Total profit | Final portfolio value minus initial capital |
| Win rate | Percentage of trades closed in profit |
| Profit factor | Gross profit / gross loss |
| Max drawdown | Largest peak-to-trough equity decline |
| Sharpe ratio | Mean return / std of returns |
| Sortino ratio | Mean excess return / downside deviation |

### 3. Ensemble aggregation (val 3)

Predictions from the best-performing models are combined and the aggregation method is optimised on a third held-out split. Available aggregation functions in [`aggregation.py`](src/evaluation/aggregation.py):

- Simple mean, median, max
- Geometric mean, harmonic mean
- Weighted blends (blend1/2/3) mixing median, geometric mean, harmonic mean, and arithmetic mean with fixed coefficients

The best model combination, aggregation function, and trading strategy are saved under `best_aggregations/` and `best_strategies/`.

## License

This project was developed as an academic thesis. All rights reserved unless otherwise stated.
