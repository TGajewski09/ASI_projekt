import json

import pandas as pd
from autogluon.tabular import TabularPredictor
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

# Model AutoGluon persist()
predictor = TabularPredictor.load("data/06_models/autogluon")
predictor.persist()

# enkoder Country - country_label
with open("data/06_models/country_label_encoder.json", encoding="utf-8") as f:
    country_encoder: dict = json.load(f)
with open("data/06_models/drift_baseline.json", encoding="utf-8") as f:
    baseline = json.load(f)

FEATURES = ["year", "month", "decade", "Latitude", "Longitude", "abs_latitude", "country_label"]


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

        prediction = float(predictor.predict(input_data).iloc[0])
        PREDICTION_COUNTER.inc()
        PREDICTION_VALUE.observe(prediction)

        return {
            "avg_temperature_c": round(prediction, 3),
            "country_label": code,
            "drift_detected": bool(drifted),
            "drift_features": drifted,
        }


app.mount("/metrics", make_asgi_app())
