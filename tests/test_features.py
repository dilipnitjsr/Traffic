import pandas as pd
import pytest

from traffic_ml.features import (
    TARGET_COL,
    build_training_features,
    chronological_split,
    feature_columns,
)


def sample_frame(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
            TARGET_COL: [100 + i for i in range(rows)],
            "temperature": [20.0 + (i % 5) for i in range(rows)],
        }
    )


def test_feature_engineering_is_chronological_and_leakage_aware():
    frame = build_training_features(sample_frame())

    assert frame["timestamp"].is_monotonic_increasing
    assert "lag_1" in frame.columns
    assert "rolling_mean_24" in frame.columns
    assert frame.iloc[0]["lag_1"] < frame.iloc[0][TARGET_COL]


def test_feature_columns_excludes_timestamp_and_target():
    frame = build_training_features(sample_frame())
    columns = feature_columns(frame)

    assert "timestamp" not in columns
    assert TARGET_COL not in columns
    assert "temperature" in columns


def test_chronological_split_preserves_order():
    frame = build_training_features(sample_frame())
    train, test = chronological_split(frame, test_fraction=0.25)

    assert train["timestamp"].max() < test["timestamp"].min()


def test_missing_target_rejected():
    with pytest.raises(ValueError):
        build_training_features(pd.DataFrame({"timestamp": ["2026-01-01"]}))
