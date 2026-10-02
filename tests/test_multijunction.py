import numpy as np

from traffic_ml.generate_multijunction import generate_multijunction_traffic
from traffic_ml.multijunction import (
    TrafficScaler,
    build_adjacency,
    chronological_window_split,
    make_windows,
)


def test_multijunction_window_shapes_and_split():
    df = generate_multijunction_traffic(rows_per_junction=120, junctions=4, seed=1)
    windows = make_windows(df, history=12, horizon=1)

    assert windows.X.shape[1:] == (12, 4, 6)
    assert windows.y.shape[1:] == (4,)
    assert len(windows.junction_ids) == 4

    train_idx, test_idx = chronological_window_split(windows, test_fraction=0.2)
    assert train_idx.max() < test_idx.min()


def test_adjacency_is_symmetric_and_normalized():
    df = generate_multijunction_traffic(rows_per_junction=120, junctions=5, seed=2)
    adjacency = build_adjacency(df, k=2)

    assert adjacency.shape == (5, 5)
    assert np.allclose(adjacency, adjacency.T)
    assert np.all(np.diag(adjacency) > 0)


def test_scaler_round_trip():
    df = generate_multijunction_traffic(rows_per_junction=120, junctions=3, seed=3)
    windows = make_windows(df, history=12)
    train_idx, _ = chronological_window_split(windows)
    scaler = TrafficScaler.fit(windows.X[train_idx], windows.y[train_idx])

    transformed = scaler.transform_y(windows.y[train_idx])
    restored = scaler.inverse_y(transformed)
    assert np.allclose(restored, windows.y[train_idx], atol=1e-4)
