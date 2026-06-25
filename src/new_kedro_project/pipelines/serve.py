import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI(title="Temperature Predictor API")

# Metryki Prometheus
PREDICTION_COUNTER = Counter("model_predictions_total", "Liczba predykcji")
PREDICTION_VALUE = Histogram(
    "model_prediction_value_celsius",
    "Rozklad przewidywanych temperatur",
    buckets=(-30, -20, -10, 0, 10, 20, 30, 40),
)
LATENCY_HISTOGRAM = Histogram("model_prediction_latency_seconds", "Czas predykcji")
DRIFT_COUNTER = Counter(
    "model_input_drift_total",
    "Liczba wejsc poza rozkladem treningowym",
    ["feature"],
)

# Katalog z modelem i artefaktami. W Dockerze model podajemy przez wolumen,
# wiec sciezke da sie podmienic zmienna srodowiskowa MODEL_DIR.
MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/06_models"))

# enkoder Country -> country_label oraz zakresy treningowe (do wykrywania driftu)
with open(MODEL_DIR / "country_label_encoder.json", encoding="utf-8") as f:
    country_encoder: dict = json.load(f)
with open(MODEL_DIR / "drift_baseline.json", encoding="utf-8") as f:
    baseline = json.load(f)

FEATURES = ["year", "month", "decade", "Latitude", "Longitude", "abs_latitude", "country_label"]

# Model AutoGluon ladujemy leniwie - dopiero przy pierwszym zapytaniu. Dzieki temu
# modul mozna zaimportowac (np. w testach CI) bez ciezkiego pakietu autogluon
# i bez samego modelu na dysku.
_predictor = None


def get_predictor():
    """Laduje i zapamietuje model AutoGluon (tylko przy pierwszym wywolaniu)."""
    global _predictor
    if _predictor is None:
        from autogluon.tabular import TabularPredictor

        _predictor = TabularPredictor.load(str(MODEL_DIR / "autogluon"))
        _predictor.persist()
    return _predictor


def check_drift(year: int, month: int, latitude: float, longitude: float, country: str) -> list[str]:
    """
    Sprawdza czy wejscia mieszcza sie w zakresie danych treningowych
    zwraca liste cech odbiegających od rozkładu treningowego
    """
    drifted = []
    values = {"year": year, "month": month, "Latitude": latitude, "Longitude": longitude}
    for feature, value in values.items():
        bounds = baseline["numeric"][feature]
        if value < bounds["min"] or value > bounds["max"]:
            drifted.append(feature)
            DRIFT_COUNTER.labels(feature=feature).inc()
    if country not in baseline["known_countries"]:
        drifted.append("Country")
        DRIFT_COUNTER.labels(feature="Country").inc()
    return drifted


@app.post("/predict")
def predict(year: int, month: int, latitude: float, longitude: float, country: str):
    with LATENCY_HISTOGRAM.time():
        drifted = check_drift(year, month, latitude, longitude, country)
        code = country_encoder.get(country, -1)

        input_data = pd.DataFrame([{
            "year": year,
            "month": month,
            "decade": (year // 10) * 10,
            "Latitude": latitude,
            "Longitude": longitude,
            "abs_latitude": abs(latitude),
            "country_label": code,
        }])[FEATURES]

        prediction = float(get_predictor().predict(input_data).iloc[0])
        PREDICTION_COUNTER.inc()
        PREDICTION_VALUE.observe(prediction)

        return {
            "avg_temperature_c": round(prediction, 3),
            "country_label": code,
            "drift_detected": bool(drifted),
            "drift_features": drifted,
        }


app.mount("/metrics", make_asgi_app())
