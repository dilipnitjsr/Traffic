from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_multijunction_traffic(
    rows_per_junction: int = 24 * 45,
    junctions: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    if rows_per_junction < 96:
        raise ValueError("rows_per_junction must be at least 96")
    if junctions < 2:
        raise ValueError("junctions must be at least 2")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=rows_per_junction, freq="h")
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()

    common = (
        180
        + 320 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
        + 390 * np.exp(-0.5 * ((hour - 18) / 2.4) ** 2)
    )
    common *= np.where(dow >= 5, 0.76, 1.0)

    # Shared latent shock creates realistic cross-junction correlation.
    latent = rng.normal(0, 18, rows_per_junction)
    records = []

    center_lat, center_lon = 22.80, 86.20
    for j in range(junctions):
        angle = 2 * np.pi * j / junctions
        latitude = center_lat + 0.04 * np.sin(angle)
        longitude = center_lon + 0.04 * np.cos(angle)

        scale = 0.75 + 0.12 * j
        local_phase = 20 * np.sin(2 * np.pi * (hour + j) / 24)
        local_noise = rng.normal(0, 22 + 2 * j, rows_per_junction)

        volume = np.maximum(
            10,
            scale * common + local_phase + latent + local_noise,
        ).round().astype(int)

        temperature = 24 + 6 * np.sin(2 * np.pi * hour / 24) + rng.normal(
            0, 1.0, rows_per_junction
        )

        for idx, ts in enumerate(timestamps):
            records.append(
                {
                    "timestamp": ts,
                    "junction_id": f"J{j + 1:02d}",
                    "traffic_volume": int(volume[idx]),
                    "temperature": round(float(temperature[idx]), 2),
                    "latitude": round(latitude, 6),
                    "longitude": round(longitude, 6),
                }
            )

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-junction traffic")
    parser.add_argument("--rows-per-junction", type=int, default=24 * 45)
    parser.add_argument("--junctions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/multijunction.csv")
    args = parser.parse_args()

    df = generate_multijunction_traffic(
        rows_per_junction=args.rows_per_junction,
        junctions=args.junctions,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(
        f"Wrote {len(df)} rows for {args.junctions} junctions to {output}"
    )


if __name__ == "__main__":
    main()
