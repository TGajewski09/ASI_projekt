import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    # 1. Rodzielenie roku i miesiaca z daty
    df["dt"] = pd.to_datetime(df["dt"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month

    # 2. Tworzenie dekad, np. 2013 -> 2010
    df["decade"] = (df["year"] // 10) * 10

    # 3. Odleglosc od rownika
    df["abs_latitude"] = df["Latitude"].abs()

    # 4. Label Encoding dla kraju
    df["country_label"], countries = pd.factorize(df["Country"])

    cols = [
        "year",
        "month",
        "decade",
        "abs_latitude",
        "country_label",
        "AverageTemperature",
    ]
    features_df = df[cols]

    print(f"[make_features] Dodano cechy. Liczba krajow: {len(countries)}")
    return features_df


def split_features_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """Podział danych na treningowe i testowe"""
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    print(f"[split_features_data] Train: {len(train)}, test: {len(test)}")
    return train, test


def train_random_forest_model(
    train_data: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
) -> RandomForestRegressor:
    """Trenuje bazowy model Random Forest"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(train_data[features], train_data[target])

    print("[train_random_forest_model] Model zostal wytrenowany.")
    return model


def evaluate_model(
    model: RandomForestRegressor,
    test_data: pd.DataFrame,
    features: list[str],
    target: str,
) -> dict:

    """Podstawowe metryki modelu"""
    y_true = test_data[target]
    y_pred = model.predict(test_data[features])

    metrics = {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }

    print(f"[evaluate_model] Metryki: {metrics}")
    return metrics


def tune_random_forest_model(
    train_data: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
    tuning_sample_size: int,
    tuning_n_iter: int,
    tuning_cv: int,
) -> tuple[RandomForestRegressor, dict]:

    """Szuka lepszych parametrow dla Random Forest"""
    if len(train_data) > tuning_sample_size:
        train_sample = train_data.sample(tuning_sample_size, random_state=random_state)
    else:
        train_sample = train_data

    params = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    model = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        model,
        params,
        n_iter=tuning_n_iter,
        cv=tuning_cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )

    search.fit(train_sample[features], train_sample[target])

    tuning_report = {
        "method": "RandomizedSearchCV",
        "sample_rows": len(train_sample),
        "checked_sets": tuning_n_iter,
        "cv": tuning_cv,
        "best_params": search.best_params_,
        "best_score": round(float(search.best_score_), 4),
    }

    print(f"[tune_random_forest_model] Najlepsze parametry: {search.best_params_}")
    return search.best_estimator_, tuning_report


def compare_results(metrics_before: dict, metrics_after: dict, tuning_report: dict) -> dict:

    """Porownuje model przed i po optymalizacji — zwraca zbiorczy raport"""
    better = "tuned_model" if metrics_after["rmse"] < metrics_before["rmse"] else "baseline_model"

    report = {
        "feature_engineering": {
            "features": ["year", "month", "decade", "abs_latitude", "country_label"],
            "transformations": [
                "Wydzielenie roku z daty (dt -> year)",
                "Wydzielenie miesiaca z daty (dt -> month)",
                "Grupowanie lat w dekady (year -> decade)",
                "Bezwzgledna szerokosc geograficzna (Latitude -> abs_latitude)",
                "Label Encoding kraju (Country -> country_label)",
            ],
        },
        "baseline_model": {
            "params": {"n_estimators": 100, "max_depth": 15},
            "metrics": metrics_before,
        },
        "hyperparameter_tuning": {
            "method": tuning_report["method"],
            "sample_rows": tuning_report["sample_rows"],
            "n_iter": tuning_report["checked_sets"],
            "cv": tuning_report["cv"],
            "best_score_cv": tuning_report["best_score"],
            "best_params": tuning_report["best_params"],
        },
        "tuned_model": {
            "metrics": metrics_after,
        },
        "conclusion": {
            "better_model": better,
            "rmse_improvement": round(metrics_before["rmse"] - metrics_after["rmse"], 4),
            "r2_improvement": round(metrics_after["r2"] - metrics_before["r2"], 4),
        },
    }

    print(f"[compare_results] Lepszy model: {better}")
    return report
