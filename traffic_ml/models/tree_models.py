from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def make_tree_model(name: str, random_state: int = 42):
    key = name.lower()

    if key == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    if key == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is not installed. Install with: pip install -e '.[benchmark]'"
            ) from exc

        base = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        return MultiOutputRegressor(base)

    if key == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is not installed. Install with: pip install -e '.[benchmark]'"
            ) from exc

        base = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbose=-1,
        )
        return MultiOutputRegressor(base)

    raise ValueError(f"Unknown tree model: {name}")
