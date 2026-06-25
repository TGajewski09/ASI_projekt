from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# słownik dostępnych modeli
_MODEL_REGISTRY = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy cechy do modelu na podstawie daty i wspolrzednych."""
    df = df.copy()
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month

    cols = ["year", "month", "Latitude", "Longitude", "AverageTemperature"]
    features_df = df[cols]

    print(f"[build_features] Przygotowano cechy dla {len(features_df)} wierszy.")
    return features_df


def split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dzieli dane na zbiór treningowy i testowy."""
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"[split_data] Trening: {len(train)} wierszy, Test: {len(test)} wierszy.")
    return train, test


def train_model(
    train_data: pd.DataFrame,
    features: list[str],
    target: str,
    model_config: dict,
) -> object:
    """Trenuje model na zbiorze treningowym."""
    model_type = model_config["type"]
    params = {k: v for k, v in model_config.items() if k != "type"}

    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Nieznany model '{model_type}'. Dostępne: {list(_MODEL_REGISTRY)}")

    model = _MODEL_REGISTRY[model_type](**params)
    model.fit(train_data[features], train_data[target])

    print(f"[train_model] Wytrenowano {model_type} na {len(train_data)} wierszach. Parametry: {params}")
    return model


def evaluate_model(
    model: object,
    test_data: pd.DataFrame,
    features: list[str],
    target: str,
) -> dict:
    """Ocenia model na zbiorze testowym i zwraca metryki."""
    y = test_data[target]
    y_pred = model.predict(test_data[features])

    model_name = type(model).__name__
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "test_rows": len(test_data),
        "mae": round(float(mean_absolute_error(y, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
        "r2": round(float(r2_score(y, y_pred)), 4),
    }

    mlflow.log_metrics({
        f"{model_name}_mae": metrics["mae"],
        f"{model_name}_rmse": metrics["rmse"],
        f"{model_name}_r2": metrics["r2"],
    })

    print(f"[evaluate_model] {model_name} - MAE={metrics['mae']}, RMSE={metrics['rmse']}, R2={metrics['r2']}")
    return metrics


def compare_models(metrics_lr: dict, metrics_rf: dict) -> dict:
    """Porównuje modele i układa je od najlepszego do najgorszego"""
    ranking = sorted(
        [metrics_lr, metrics_rf],
        key=lambda m: m["r2"],
        reverse=True,
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": ranking[0]["model"],
        "ranking": [
            {
                "rank": i + 1,
                "model": m["model"],
                "r2": m["r2"],
                "mae": m["mae"],
                "rmse": m["rmse"],
            }
            for i, m in enumerate(ranking)
        ],
    }

    print(f"[compare_models] Najlepszy model: {report['best_model']} (R2={ranking[0]['r2']})")
    return report
