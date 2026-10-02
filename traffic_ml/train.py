from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import (
    TARGET_COL,
    build_training_features,
    chronological_split,
    feature_columns,
)


def train_model(
    data_path: str,
    model_path: str,
    metrics_path: str,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    df = pd.read_csv(data_path)
    featured = build_training_features(df)
    train_df, test_df = chronological_split(featured, test_fraction=test_fraction)

    columns = feature_columns(featured)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(train_df[columns], train_df[TARGET_COL])

    predictions = model.predict(test_df[columns])

    metrics = {
        "mae": float(mean_absolute_error(test_df[TARGET_COL], predictions)),
        "rmse": float(mean_squared_error(test_df[TARGET_COL], predictions) ** 0.5),
        "r2": float(r2_score(test_df[TARGET_COL], predictions)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": columns,
        },
        model_output,
    )

    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the traffic forecasting baseline")
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--model", default="artifacts/traffic_model.joblib")
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_model(
        data_path=args.data,
        model_path=args.model,
        metrics_path=args.metrics,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
