"""Testy jednostkowe wezla compare_all_models (pipeline evaluation)."""
from new_kedro_project.pipelines.evaluation import nodes


def _metrics(rmse, r2=0.9, mae=1.0):
    return {"mae": mae, "rmse": rmse, "r2": r2}


def test_compare_all_models_picks_lowest_rmse(mocker):
    # podmieniamy mlflow, zeby test nie wymagal aktywnego przebiegu MLflow
    mocker.patch.object(nodes, "mlflow")

    report = nodes.compare_all_models(
        metrics_lr=_metrics(9.0),
        metrics_rf=_metrics(1.5),
        engineered_metrics_rf=_metrics(1.45),
        tuned_metrics_rf=_metrics(1.4),
        automl_metrics=_metrics(1.3),
    )

    # najnizszy RMSE wygrywa
    assert report["best_model"] == "AutoGluon"
    assert report["ranking"][0]["rmse"] == 1.3
    assert report["ranking"][-1]["rmse"] == 9.0
