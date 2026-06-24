from datetime import datetime

import mlflow


def _pick(metrics: dict) -> dict:
    """Wyciaga trzy glowne metryki"""
    return {"mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"]}


def compare_all_models(
    metrics_lr: dict,
    metrics_rf: dict,
    engineered_metrics_rf: dict,
    tuned_metrics_rf: dict,
    automl_metrics: dict,
) -> dict:
    """Zbiorcze porownanie wszystkich modeli z pipelineow i AutoML"""
    entries = [
        {"model": "LinearRegression (data_modeling)", **_pick(metrics_lr)},
        {"model": "RandomForest (data_modeling)", **_pick(metrics_rf)},
        {"model": "RandomForest (feature_engineering)", **_pick(engineered_metrics_rf)},
        {"model": "RandomForest + RandomizedSearch", **_pick(tuned_metrics_rf)},
        {"model": "AutoGluon", **_pick(automl_metrics)},
    ]
    ranking = sorted(entries, key=lambda e: e["rmse"])
    for i, entry in enumerate(ranking):
        entry["rank"] = i + 1

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": ranking[0]["model"],
        "ranking": ranking,
    }

    mlflow.log_metric("best_overall_rmse", ranking[0]["rmse"])
    mlflow.log_metric("best_overall_r2", ranking[0]["r2"])

    print(f"[compare_all_models] Najlepszy model: {report['best_model']} (RMSE={ranking[0]['rmse']})")
    return report
