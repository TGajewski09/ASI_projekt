"""Testy logiki wykrywania driftu w API (serve.py).

Import dziala bez pakietu autogluon - model laduje sie leniwie, a tutaj
sprawdzamy tylko czysta logike check_drift na podstawie zapisanych zakresow.
"""
from new_kedro_project.pipelines import serve


def test_drift_flags_year_before_training_range():
    # dane treningowe zaczynaja sie od 1850 roku
    drifted = serve.check_drift(1700, 6, 52.0, 21.0)
    assert "year" in drifted


def test_no_drift_for_typical_input():
    drifted = serve.check_drift(2010, 6, 52.0, 21.0)
    assert drifted == []


def test_drift_flags_impossible_latitude():
    # szerokosc geograficzna 200 jest poza zakresem treningowym
    drifted = serve.check_drift(2000, 6, 200.0, 21.0)
    assert "Latitude" in drifted
