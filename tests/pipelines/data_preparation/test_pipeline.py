"""Testy jednostkowe wezlow z pipeline'u data_preparation."""
import pandas as pd

from new_kedro_project.pipelines.data_preparation.nodes import (
    parse_coordinates,
    remove_duplicates,
    remove_nulls,
)


def test_remove_nulls_drops_missing_temperature():
    df = pd.DataFrame({"AverageTemperature": [1.0, None, 3.0], "City": ["a", "b", "c"]})

    out = remove_nulls(df, {"signal": "ok"})

    assert len(out) == 2
    assert out["AverageTemperature"].isna().sum() == 0


def test_remove_duplicates_by_dt_city_country():
    df = pd.DataFrame(
        {
            "dt": ["2000-01-01", "2000-01-01", "2000-02-01"],
            "City": ["Warsaw", "Warsaw", "Warsaw"],
            "Country": ["Poland", "Poland", "Poland"],
            "AverageTemperature": [1.0, 1.0, 2.0],
        }
    )

    out = remove_duplicates(df)

    assert len(out) == 2


def test_parse_coordinates_handles_directions():
    df = pd.DataFrame(
        {"Latitude": ["57.05N", "10.00S"], "Longitude": ["20.00E", "30.00W"]}
    )

    out = parse_coordinates(df)

    # S i W oznaczaja wartosci ujemne
    assert list(out["Latitude"]) == [57.05, -10.0]
    assert list(out["Longitude"]) == [20.0, -30.0]
