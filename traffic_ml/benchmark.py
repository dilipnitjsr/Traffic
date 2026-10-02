from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .metrics import regression_metrics
from .models.tree_models import make_tree_model
from .multijunction import (
    TrafficScaler,
    build_adjacency,
    chronological_window_split,
    make_windows,
)

TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}
TORCH_MODELS = {"lstm", "tcn", "transformer", "stgnn"}
ALL_MODELS = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "lstm",
    "tcn",
    "transformer",
    "stgnn",
]


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def _train_tree(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
):
    model = make_tree_model(name, random_state=random_state)
    model.fit(_flatten(X_train), y_train)
    return model, np.asarray(model.predict(_flatten(X_test)), dtype=np.float32)


def _train_torch(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    adjacency: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install with: pip install -e '.[benchmark]'"
        ) from exc

    from .models.torch_models import make_torch_model

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    scaler = TrafficScaler.fit(X_train, y_train)
    X_train_scaled = scaler.transform_X(X_train)
    y_train_scaled = scaler.transform_y(y_train)
    X_test_scaled = scaler.transform_X(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32, device=device)

    model = make_torch_model(
        name,
        num_nodes=X_train.shape[2],
        num_features=X_train.shape[3],
        adjacency=adjacency_tensor,
    ).to(device)

    dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        ).cpu().numpy()

    return (
        {
            "state_dict": model.state_dict(),
            "model_name": name,
            "num_nodes": X_train.shape[2],
            "num_features": X_train.shape[3],
            "scaler": {"mean": scaler.mean, "std": scaler.std},
        },
        scaler.inverse_y(pred_scaled),
    )


def run_benchmark(
    data_path: str,
    output_dir: str = "artifacts/benchmark",
    models: list[str] | None = None,
    history: int = 24,
    horizon: int = 1,
    test_fraction: float = 0.2,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> pd.DataFrame:
    selected = models or ALL_MODELS
    unknown = sorted(set(selected).difference(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    df = pd.read_csv(data_path)
    windows = make_windows(df, history=history, horizon=horizon)
    train_idx, test_idx = chronological_window_split(
        windows, test_fraction=test_fraction
    )
    adjacency = build_adjacency(df)

    X_train, y_train = windows.X[train_idx], windows.y[train_idx]
    X_test, y_test = windows.X[test_idx], windows.y[test_idx]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    predictions = []

    for name in selected:
        start = time.perf_counter()

        if name in TREE_MODELS:
            model, y_pred = _train_tree(
                name, X_train, y_train, X_test, random_state=random_state
            )
            joblib.dump(model, output / f"{name}.joblib")
        else:
            bundle, y_pred = _train_torch(
                name,
                X_train,
                y_train,
                X_test,
                adjacency=adjacency,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            import torch

            torch.save(bundle, output / f"{name}.pt")

        elapsed = time.perf_counter() - start
        metrics = regression_metrics(y_test, y_pred)
        metrics.update(
            {
                "model": name,
                "train_seconds": round(elapsed, 4),
                "train_windows": int(len(train_idx)),
                "test_windows": int(len(test_idx)),
            }
        )
        rows.append(metrics)

        for sample_i, timestamp in enumerate(windows.timestamps[test_idx]):
            for node_i, junction in enumerate(windows.junction_ids):
                predictions.append(
                    {
                        "model": name,
                        "timestamp": str(timestamp),
                        "junction_id": junction,
                        "actual": float(y_test[sample_i, node_i]),
                        "prediction": float(y_pred[sample_i, node_i]),
                    }
                )

    leaderboard = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)

    metadata = {
        "history": history,
        "horizon": horizon,
        "test_fraction": test_fraction,
        "junction_ids": windows.junction_ids,
        "feature_names": windows.feature_names,
        "models": selected,
        "adjacency": adjacency.tolist(),
    }
    (output / "benchmark_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return leaderboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare tree, temporal deep-learning and spatiotemporal models"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="artifacts/benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
    )
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    leaderboard = run_benchmark(
        data_path=args.data,
        output_dir=args.output_dir,
        models=args.models,
        history=args.history,
        horizon=args.horizon,
        test_fraction=args.test_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
