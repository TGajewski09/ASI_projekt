"""Testy jednostkowe wezla make_features (pipeline feature_engineering)."""
import pandas as pd

from new_kedro_project.pipelines.feature_engineering.nodes import make_features


def test_make_features_creates_derived_columns():
    df = pd.DataFrame(
        {
            "dt": ["2013-05-01", "1990-08-01"],
            "Latitude": [52.0, -33.0],
            "Longitude": [21.0, 18.0],
            "AverageTemperature": [15.0, 20.0],
        }
    )

    out = make_features(df)

    assert list(out["year"]) == [2013, 1990]
    assert list(out["month"]) == [5, 8]
    # dekada to rok zaokraglony w dol do 10 lat
    assert list(out["decade"]) == [2010, 1990]
    # abs_latitude to odleglosc od rownika (zawsze dodatnia)
    assert list(out["abs_latitude"]) == [52.0, 33.0]
    # polozenie opisuja wspolrzedne - kraju juz nie kodujemy
    assert "country_label" not in out.columns
