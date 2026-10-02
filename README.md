# Traffic Forecasting

Comparative machine-learning, deep-learning, and spatiotemporal forecasting framework for multi-junction road traffic.

## Implemented models

The same chronological train/test windows and metrics are used for:

| family | model |
|---|---|
| Ensemble baseline | Random Forest |
| Gradient boosting | XGBoost |
| Gradient boosting | LightGBM |
| Recurrent DL | LSTM |
| Temporal convolution | TCN |
| Attention | Transformer encoder |
| Spatiotemporal graph DL | ST-GNN: normalized graph convolution + GRU |

The ST-GNN explicitly uses a junction adjacency matrix. If latitude/longitude columns are available, the graph is built from nearest spatial neighbours. Otherwise, a simple ring topology is used as a deterministic fallback.

## Multi-junction data schema

Required columns:

| column | description |
|---|---|
| `timestamp` | observation timestamp |
| `junction_id` | stable road junction/sensor identifier |
| `traffic_volume` | traffic count/flow target |

Recommended optional columns:

| column | description |
|---|---|
| `latitude` | junction latitude used to build spatial graph |
| `longitude` | junction longitude used to build spatial graph |
| other numeric columns | retained in raw data for future feature extensions |

Each timestamp must contain one observation for every junction.

Example:

```csv
timestamp,junction_id,traffic_volume,latitude,longitude
2026-01-01 00:00:00,J01,320,22.80,86.24
2026-01-01 00:00:00,J02,280,22.83,86.22
2026-01-01 01:00:00,J01,305,22.80,86.24
2026-01-01 01:00:00,J02,270,22.83,86.22
```

## Installation

Core baseline only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Full comparative framework:

```bash
pip install -e ".[benchmark,dev]"
```

The full benchmark extra installs XGBoost, LightGBM, and PyTorch.

## Quick multi-junction demo

Generate synthetic correlated traffic from six junctions:

```bash
python -m traffic_ml.generate_multijunction \
  --rows-per-junction 1080 \
  --junctions 6 \
  --output data/multijunction.csv
```

Run the complete comparison:

```bash
python -m traffic_ml.benchmark \
  --data data/multijunction.csv \
  --history 24 \
  --horizon 1 \
  --epochs 10
```

Run selected models:

```bash
python -m traffic_ml.benchmark \
  --data data/multijunction.csv \
  --models xgboost lightgbm lstm transformer stgnn \
  --history 24 \
  --epochs 20
```

## Benchmark outputs

By default results are written to `artifacts/benchmark/`:

```text
benchmark_config.json
leaderboard.csv
predictions.csv
random_forest.joblib
xgboost.joblib
lightgbm.joblib
lstm.pt
tcn.pt
transformer.pt
stgnn.pt
```

The leaderboard contains common metrics:

- MAE
- RMSE
- R²
- MAPE
- model training time
- number of train/test windows

`predictions.csv` stores actual and predicted traffic for each model, timestamp, and junction, which makes junction-level error analysis straightforward.

## Common experimental protocol

All models use the same preprocessing:

```text
multi-junction CSV
      ↓
validate complete timestamp × junction panel
      ↓
sort chronologically
      ↓
24-step historical windows
      ↓
traffic + cyclic hour/day calendar features
      ↓
chronological train/test split
      ↓
same target timestamps for every model
      ↓
common evaluation metrics
```

No random row-wise train/test split is used.

## Model details

### Random Forest / XGBoost / LightGBM

The complete historical window is flattened into a tabular feature vector. Each model predicts traffic for all junctions at the next forecast horizon. XGBoost and LightGBM are wrapped as multi-output regressors for consistent behaviour across junctions.

### LSTM

The historical network state is flattened per timestamp and passed through an LSTM. The final recurrent state predicts all junctions simultaneously.

### TCN

Dilated one-dimensional temporal convolutions capture short- and medium-range traffic dynamics across the historical window.

### Transformer

Each timestamp is projected into an embedding, combined with positional encoding, and processed by a Transformer encoder. The final temporal representation predicts all junctions.

### ST-GNN

For each historical timestamp:

1. node features are projected into a graph hidden space;
2. normalized adjacency propagates information between connected junctions;
3. each junction's graph-aware sequence is processed by a GRU;
4. the final hidden state predicts the next traffic value for that junction.

This gives the model an explicit spatial inductive bias absent from LSTM/TCN/Transformer baselines.

## Graph construction

When `latitude` and `longitude` are available, each junction is connected to its nearest neighbours. Self-loops are included and the adjacency matrix is symmetrically normalized:

```text
A_hat = D^(-1/2) A D^(-1/2)
```

The normalized matrix is stored in `benchmark_config.json` for reproducibility.

## Legacy single-junction baseline

The original single-location Random Forest workflow remains available:

```bash
python -m traffic_ml.generate_sample --output data/traffic.csv
python -m traffic_ml.train --data data/traffic.csv
python -m traffic_ml.predict --data data/traffic.csv
```

## Tests and CI

```bash
pytest
```

GitHub Actions runs:

1. core preprocessing/unit tests;
2. PyTorch model interface tests;
3. a full one-epoch smoke benchmark across all seven models.

## Research extensions

The framework is structured so additional experiments can be added without changing the evaluation contract. Natural next additions include multi-step forecasting, learned/dynamic adjacency, attention-based graph networks, external weather/events, uncertainty estimation, domain adaptation across cities, and optimization-guided model selection.
