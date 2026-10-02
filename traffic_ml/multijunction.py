from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TIMESTAMP_COL = "timestamp"
JUNCTION_COL = "junction_id"
TARGET_COL = "traffic_volume"


@dataclass(frozen=True)
class WindowedTraffic:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    junction_ids: list[str]
    feature_names: list[str]


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    required = {TIMESTAMP_COL, JUNCTION_COL, TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input data is empty")

    frame = df.copy()
    frame[TIMESTAMP_COL] = pd.to_datetime(frame[TIMESTAMP_COL], errors="raise")
    frame[JUNCTION_COL] = frame[JUNCTION_COL].astype(str)
    frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="raise")
    frame = frame.sort_values([TIMESTAMP_COL, JUNCTION_COL])
    frame = frame.drop_duplicates([TIMESTAMP_COL, JUNCTION_COL], keep="last")

    counts = frame.groupby(TIMESTAMP_COL)[JUNCTION_COL].nunique()
    expected = frame[JUNCTION_COL].nunique()
    if not (counts == expected).all():
        raise ValueError(
            "Multi-junction data must contain every junction at every timestamp"
        )
    return frame.reset_index(drop=True)


def _calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(float),
        ]
    ).astype(np.float32)


def make_windows(
    df: pd.DataFrame,
    history: int = 24,
    horizon: int = 1,
) -> WindowedTraffic:
    if history < 2:
        raise ValueError("history must be at least 2")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    frame = prepare_panel(df)
    junction_ids = sorted(frame[JUNCTION_COL].unique().tolist())
    target = (
        frame.pivot(index=TIMESTAMP_COL, columns=JUNCTION_COL, values=TARGET_COL)
        .reindex(columns=junction_ids)
        .sort_index()
    )

    timestamps = target.index
    volumes = target.to_numpy(dtype=np.float32)
    calendar = _calendar_features(timestamps)

    # Feature 0 is traffic volume. Calendar features are copied to every node.
    node_calendar = np.repeat(calendar[:, None, :], len(junction_ids), axis=1)
    features = np.concatenate([volumes[:, :, None], node_calendar], axis=2)

    X, y, target_times = [], [], []
    last_start = len(target) - history - horizon + 1
    for start in range(last_start):
        end = start + history
        target_index = end + horizon - 1
        X.append(features[start:end])
        y.append(volumes[target_index])
        target_times.append(timestamps[target_index].to_datetime64())

    if not X:
        raise ValueError("Not enough rows for the requested history/horizon")

    return WindowedTraffic(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        timestamps=np.asarray(target_times),
        junction_ids=junction_ids,
        feature_names=[
            "traffic_volume",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
        ],
    )


def chronological_window_split(
    windows: WindowedTraffic,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    n = len(windows.X)
    if n < 10:
        raise ValueError("At least 10 windows are required")
    split = max(1, min(int(n * (1 - test_fraction)), n - 1))
    return np.arange(split), np.arange(split, n)


def build_adjacency(df: pd.DataFrame, k: int = 2) -> np.ndarray:
    frame = prepare_panel(df)
    junction_ids = sorted(frame[JUNCTION_COL].unique().tolist())
    n = len(junction_ids)
    if n < 2:
        return np.ones((n, n), dtype=np.float32)

    metadata = frame.drop_duplicates(JUNCTION_COL).set_index(JUNCTION_COL)
    has_coords = {"latitude", "longitude"}.issubset(metadata.columns)

    adjacency = np.eye(n, dtype=np.float32)

    if has_coords:
        coords = metadata.reindex(junction_ids)[["latitude", "longitude"]].to_numpy(float)
        delta = coords[:, None, :] - coords[None, :, :]
        distance = np.sqrt((delta**2).sum(axis=2))
        k_eff = min(max(1, k), n - 1)
        for i in range(n):
            neighbours = np.argsort(distance[i])[1 : k_eff + 1]
            adjacency[i, neighbours] = 1.0
            adjacency[neighbours, i] = 1.0
    else:
        for i in range(n):
            adjacency[i, (i - 1) % n] = 1.0
            adjacency[i, (i + 1) % n] = 1.0

    degree = adjacency.sum(axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-8)))
    return (inv_sqrt @ adjacency @ inv_sqrt).astype(np.float32)


@dataclass
class TrafficScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, X_train: np.ndarray, y_train: np.ndarray) -> "TrafficScaler":
        traffic = np.concatenate(
            [X_train[..., 0].reshape(-1), y_train.reshape(-1)]
        )
        mean = float(traffic.mean())
        std = float(traffic.std())
        return cls(mean=mean, std=max(std, 1e-6))

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        result = X.copy()
        result[..., 0] = (result[..., 0] - self.mean) / self.std
        return result

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean
