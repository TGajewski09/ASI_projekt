"""Ponowne trenowanie modelu na probce danych (Continuous Training).

Uzywane przez workflow GitHub Actions (.github/workflows/continuous-training.yml).
Trenuje lekki model RandomForest na malej probce danych
(data/sample/temperatures_sample.csv), liczy metryki na zbiorze testowym
i zapisuje wytrenowany model oraz raport z metrykami.

Dzieki probce calosc dziala szybko i nie wymaga pobierania pelnych danych
z Kaggle - to demonstracja mechanizmu automatycznego retrenowania.

Uruchomienie (z katalogu projektu):
    python scripts/retrain.py
"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT_ROOT / "data/sample/temperatures_sample.csv"
MODEL_OUT = PROJECT_ROOT / "data/06_models/retrained_model.pkl"
REPORT_OUT = PROJECT_ROOT / "data/08_reporting/retrain_metrics.json"

TARGET = "AverageTemperature"
FEATURES = ["year", "month", "decade", "Latitude", "Longitude", "abs_latitude", "country_label"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy te same cechy co pipeline feature_engineering."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    df["decade"] = (df["year"] // 10) * 10
    df["abs_latitude"] = df["Latitude"].abs()
    df["country_label"], _ = pd.factorize(df["Country"])
    return df


def main() -> None:
    if not SAMPLE.exists():
        raise FileNotFoundError(f"Brak probki danych: {SAMPLE}")

    df = pd.read_csv(SAMPLE).dropna(subset=[TARGET])
    df = make_features(df)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=80, max_depth=15, random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURES], train[TARGET])

    pred = model.predict(test[FEATURES])
    metrics = {
        "retrained_at": datetime.now().isoformat(timespec="seconds"),
        "rows_total": len(df),
        "rows_train": len(train),
        "rows_test": len(test),
        "mae": round(float(mean_absolute_error(test[TARGET], pred)), 4),
        "rmse": round(float(mean_squared_error(test[TARGET], pred) ** 0.5), 4),
        "r2": round(float(r2_score(test[TARGET], pred)), 4),
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    REPORT_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[retrain] Wytrenowano RandomForest na {len(train)} wierszach.")
    print(f"[retrain] Metryki (test): MAE={metrics['mae']}, RMSE={metrics['rmse']}, R2={metrics['r2']}")
    print(f"[retrain] Zapisano model: {MODEL_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[retrain] Zapisano raport: {REPORT_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
