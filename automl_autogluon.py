"""AutoML z AutoGluon dla regresji temperatur
czyta dane z feature_engineering
trenuje AutoGluon na probce, liczy metryki na zbiorze testowym i loguje wynik do MLflow
"""
import json
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).parent
TARGET = "AverageTemperature"
FEATURES = ["year", "month", "decade", "Latitude", "Longitude", "abs_latitude", "country_label"]
SAMPLE_SIZE = 100_000
TIME_LIMIT = 300
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "temperature_regression"


def main() -> None:
    train = pd.read_parquet(PROJECT_ROOT / "data/05_model_input/engineered_train_data.parquet")
    test = pd.read_parquet(PROJECT_ROOT / "data/05_model_input/engineered_test_data.parquet")

    cols = FEATURES + [TARGET]
    train_sample = train[cols].sample(min(len(train), SAMPLE_SIZE), random_state=42)
    test = test[cols]

    print(f"[automl] Trening AutoGluon na {len(train_sample)} wierszach, limit {TIME_LIMIT}s")
    predictor = TabularPredictor(
        label=TARGET,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=str(PROJECT_ROOT / "data/06_models/autogluon"),
    ).fit(train_sample, time_limit=TIME_LIMIT, verbosity=2)

    y_true = test[TARGET]
    y_pred = predictor.predict(test[FEATURES])

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "AutoGluon",
        "best_model": predictor.model_best,
        "test_rows": len(test),
        "sample_rows": len(train_sample),
        "time_limit_s": TIME_LIMIT,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }

    leaderboard = predictor.leaderboard(test)
    leaderboard_records = leaderboard[["model", "score_test", "score_val"]].to_dict("records")

    reporting = PROJECT_ROOT / "data/08_reporting"
    (reporting / "automl_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (reporting / "automl_leaderboard.json").write_text(
        json.dumps(leaderboard_records, indent=2), encoding="utf-8"
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="autogluon_automl"):
        mlflow.log_params({
            "sample_rows": len(train_sample),
            "time_limit_s": TIME_LIMIT,
            "best_model": predictor.model_best,
        })
        mlflow.log_metrics({
            "AutoGluon_mae": metrics["mae"],
            "AutoGluon_rmse": metrics["rmse"],
            "AutoGluon_r2": metrics["r2"],
        })
        mlflow.log_artifact(str(reporting / "automl_metrics.json"))
        mlflow.log_artifact(str(reporting / "automl_leaderboard.json"))

    print(f"[automl] Gotowe. Najlepszy model: {predictor.model_best}, R2={metrics['r2']}, RMSE={metrics['rmse']}")


if __name__ == "__main__":
    main()
