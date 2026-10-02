from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .features import TARGET_COL, TIMESTAMP_COL, build_training_features


def predict_from_history(data_path: str, model_path: str, output_path: str) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    columns = bundle["feature_columns"]

    df = pd.read_csv(data_path)
    featured = build_training_features(df)

    missing = [col for col in columns if col not in featured.columns]
    if missing:
        raise ValueError(f"Input is missing model features: {missing}")

    predictions = model.predict(featured[columns])

    result = featured[[TIMESTAMP_COL, TARGET_COL]].copy()
    result["prediction"] = predictions

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate traffic-volume predictions")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="artifacts/traffic_model.joblib")
    parser.add_argument("--output", default="artifacts/predictions.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = predict_from_history(args.data, args.model, args.output)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
