from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_traffic(rows: int = 24 * 60, seed: int = 42) -> pd.DataFrame:
    if rows < 72:
        raise ValueError("rows must be at least 72")

    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2026-01-01", periods=rows, freq="h")

    hour = timestamp.hour.to_numpy()
    weekday = timestamp.dayofweek.to_numpy()

    morning_peak = 450 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
    evening_peak = 520 * np.exp(-0.5 * ((hour - 18) / 2.2) ** 2)
    weekend_factor = np.where(weekday >= 5, 0.72, 1.0)
    baseline = 220 + morning_peak + evening_peak
    noise = rng.normal(0, 35, size=rows)

    volume = np.maximum(20, (baseline * weekend_factor + noise)).round().astype(int)
    temperature = 25 + 6 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.2, rows)
    rain_mm = np.maximum(0, rng.gamma(shape=0.5, scale=1.2, size=rows) - 0.5)

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "traffic_volume": volume,
            "temperature": temperature.round(2),
            "rain_mm": rain_mm.round(2),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic hourly traffic data")
    parser.add_argument("--rows", type=int, default=24 * 60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/traffic.csv")
    args = parser.parse_args()

    df = generate_synthetic_traffic(rows=args.rows, seed=args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
