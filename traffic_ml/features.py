from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TIMESTAMP_COL = "timestamp"
TARGET_COL = "traffic_volume"


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    rolling_windows: tuple[int, ...] = (3, 6, 12, 24)


def validate_input(df: pd.DataFrame, require_target: bool = True) -> None:
    required = {TIMESTAMP_COL}
    if require_target:
        required.add(TARGET_COL)

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Input data is empty")


def prepare_base_frame(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    validate_input(df, require_target=require_target)

    result = df.copy()
    result[TIMESTAMP_COL] = pd.to_datetime(result[TIMESTAMP_COL], errors="raise")
    result = result.sort_values(TIMESTAMP_COL).drop_duplicates(TIMESTAMP_COL, keep="last")
    result = result.reset_index(drop=True)

    if require_target:
        result[TARGET_COL] = pd.to_numeric(result[TARGET_COL], errors="raise")

    return result


def _calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    ts = frame[TIMESTAMP_COL]
    frame["hour"] = ts.dt.hour
    frame["day_of_week"] = ts.dt.dayofweek
    frame["day_of_month"] = ts.dt.day
    frame["month"] = ts.dt.month
    frame["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclic encodings help represent boundaries such as 23:00 → 00:00.
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    return frame


def build_training_features(
    df: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    frame = prepare_base_frame(df, require_target=True)
    frame = _calendar_features(frame)

    for lag in config.lags:
        frame[f"lag_{lag}"] = frame[TARGET_COL].shift(lag)

    # Shift before rolling so the current target is never used as a feature.
    shifted_target = frame[TARGET_COL].shift(1)
    for window in config.rolling_windows:
        frame[f"rolling_mean_{window}"] = shifted_target.rolling(window=window).mean()
        frame[f"rolling_std_{window}"] = shifted_target.rolling(window=window).std()

    return frame.dropna().reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {TIMESTAMP_COL, TARGET_COL}
    cols: list[str] = []

    for col in frame.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)

    if not cols:
        raise ValueError("No numeric feature columns are available")

    return cols


def chronological_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(frame) < 10:
        raise ValueError("At least 10 feature rows are required")

    split_index = int(len(frame) * (1 - test_fraction))
    split_index = max(1, min(split_index, len(frame) - 1))

    return (
        frame.iloc[:split_index].copy(),
        frame.iloc[split_index:].copy(),
    )
