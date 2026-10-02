# Traffic Forecasting

A reproducible machine-learning baseline for forecasting road traffic volume from timestamped observations.

The repository starts with a leakage-aware time-series workflow rather than a random train/test split:

```text
CSV data
  ↓
schema validation
  ↓
timestamp sorting
  ↓
calendar + lag/rolling features
  ↓
chronological train/test split
  ↓
RandomForest baseline
  ↓
MAE / RMSE / R²
  ↓
saved model artifact
  ↓
batch prediction
```

## Data format

Prepare a CSV with at least:

| column | type | description |
|---|---|---|
| `timestamp` | datetime | observation time |
| `traffic_volume` | numeric | target traffic count/flow |

Optional numeric exogenous columns are retained automatically and can be used as predictors.

Example:

```csv
timestamp,traffic_volume,temperature,rain_mm
2026-01-01 00:00:00,320,18.2,0
2026-01-01 01:00:00,280,17.8,0
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick end-to-end demo

Generate a synthetic hourly dataset:

```bash
python -m traffic_ml.generate_sample --output data/traffic.csv
```

Then train and evaluate:

```bash
python -m traffic_ml.train --data data/traffic.csv
```

The model is written to `artifacts/traffic_model.joblib` and evaluation metrics to `artifacts/metrics.json`.

## Train

```bash
python -m traffic_ml.train \
  --data data/traffic.csv \
  --model artifacts/traffic_model.joblib \
  --metrics artifacts/metrics.json
```

## Predict

```bash
python -m traffic_ml.predict \
  --data data/traffic.csv \
  --model artifacts/traffic_model.joblib \
  --output artifacts/predictions.csv
```

## Tests

```bash
pytest
```

## Design choices

- **Chronological split:** prevents future observations from leaking into training.
- **Lag features:** capture short-term traffic persistence.
- **Rolling statistics:** represent recent traffic level and volatility.
- **Calendar features:** hour, weekday, month and weekend effects.
- **RandomForest baseline:** robust nonlinear benchmark with little tuning.

This is a baseline for research and prototyping. For stronger forecasting studies, compare against gradient boosting, XGBoost/LightGBM, LSTM/TCN, temporal transformers, and spatiotemporal graph neural networks where multi-location sensor topology is available.

## Repository layout

```text
traffic_ml/
  features.py
  train.py
  predict.py
tests/
data/
artifacts/
```

Raw datasets and trained artifacts are ignored by Git by default.
