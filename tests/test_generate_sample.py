from traffic_ml.generate_sample import generate_synthetic_traffic


def test_generator_produces_expected_schema():
    df = generate_synthetic_traffic(rows=100, seed=1)

    assert list(df.columns) == [
        "timestamp",
        "traffic_volume",
        "temperature",
        "rain_mm",
    ]
    assert len(df) == 100
    assert (df["traffic_volume"] > 0).all()
