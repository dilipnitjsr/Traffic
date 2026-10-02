from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)

    nonzero = np.abs(true) > 1e-8
    mape = (
        float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )

    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "r2": float(r2_score(true, pred)),
        "mape": mape,
    }
