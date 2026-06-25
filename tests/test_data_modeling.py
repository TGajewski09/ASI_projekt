"""Testy jednostkowe wezlow z pipeline'u data_modeling."""
import pandas as pd

from new_kedro_project.pipelines.data_modeling.nodes import (
    build_features,
    compare_models,
)


def test_build_features_encodes_countries():
    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2000-01-15", "2005-06-20"]),
            "Latitude": [10.0, 20.0],
            "Longitude": [30.0, 40.0],
            "Country": ["Atlantis", "Narnia"],
            "AverageTemperature": [1.0, 2.0],
        }
    )

    features, encoder = build_features(df)

    assert encoder == {"Atlantis": 0, "Narnia": 1}
    assert list(features["country_code"]) == [0, 1]
    assert list(features["year"]) == [2000, 2005]
    assert "AverageTemperature" in features.columns


def test_compare_models_ranks_by_r2():
    lr = {"model": "LinearRegression", "r2": 0.5, "mae": 2.0, "rmse": 3.0}
    rf = {"model": "RandomForestRegressor", "r2": 0.9, "mae": 1.0, "rmse": 1.5}

    report = compare_models(lr, rf)

    assert report["best_model"] == "RandomForestRegressor"
    assert [r["model"] for r in report["ranking"]] == [
        "RandomForestRegressor",
        "LinearRegression",
    ]
