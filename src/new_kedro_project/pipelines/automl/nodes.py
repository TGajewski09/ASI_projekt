from datetime import datetime

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_autogluon_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    automl_params: dict,
) -> tuple[dict, list[dict]]:
    from autogluon.tabular import TabularPredictor

    target = automl_params["target"]
    features = automl_params["features"]
    sample_size = automl_params["sample_size"]
    test_sample_size = automl_params["test_sample_size"]
    time_limit = automl_params["time_limit"]
    model_path = automl_params["model_path"]
    experiment = automl_params["experiment"]

    cols = features + [target]
    train_sample = train_data[cols].sample(
        min(len(train_data), sample_size),
        random_state=automl_params["random_state"],
    )
    test = test_data[cols].sample(
        min(len(test_data), test_sample_size),
        random_state=automl_params["random_state"],
    )

    print(
        f"[train_autogluon_model] Trening AutoGluon na {len(train_sample)} "
        f"wierszach, limit {time_limit}s"
    )
    predictor = TabularPredictor(
        label=target,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=model_path,
    ).fit(train_sample, time_limit=time_limit, verbosity=2)

    print(f"[train_autogluon_model] Ewaluacja AutoGluon na {len(test)} wierszach")
    y_true = test[target]
    y_pred = predictor.predict(test[features])

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "AutoGluon",
        "best_model": predictor.model_best,
        "test_rows": len(test),
        "sample_rows": len(train_sample),
        "test_sample_rows": len(test),
        "time_limit_s": time_limit,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }

    leaderboard = predictor.leaderboard(test)
    leaderboard_columns = [
        column
        for column in ["model", "score_test", "score_val"]
        if column in leaderboard.columns
    ]
    leaderboard_records = leaderboard[leaderboard_columns].to_dict("records")

    tracking_uri = automl_params.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="autogluon_automl", nested=True):
        mlflow.log_params({
            "sample_rows": len(train_sample),
            "time_limit_s": time_limit,
            "best_model": predictor.model_best,
        })
        mlflow.log_metrics({
            "AutoGluon_mae": metrics["mae"],
            "AutoGluon_rmse": metrics["rmse"],
            "AutoGluon_r2": metrics["r2"],
        })

    print(
        "[train_autogluon_model] Gotowe. "
        f"Najlepszy model: {predictor.model_best}, RMSE={metrics['rmse']}"
    )
    return metrics, leaderboard_records
