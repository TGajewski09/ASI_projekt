import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy cechy pochodne na podstawie daty i wspolrzednych.

    Dla cech bazowych (year, month, Latitude, Longitude) tworzymy przeksztalcenia:
    - decade (grupy lat)
    - abs_latitude (odleglosc od rownika).

    Polozenie opisuja same wspolrzedne (Latitude, Longitude), dlatego nie
    korzystamy juz z kraju - byloby to nadmiarowe.
    """
    df = df.copy()

    # 1. Rodzielenie roku i miesiaca z daty
    df["dt"] = pd.to_datetime(df["dt"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month

    # 2. Tworzenie dekad, np. 2013 -> 2010
    df["decade"] = (df["year"] // 10) * 10

    # 3. Odleglosc od rownika
    df["abs_latitude"] = df["Latitude"].abs()

    cols = [
        "year",
        "month",
        "decade",
        "Latitude",
        "Longitude",
        "abs_latitude",
        "AverageTemperature",
    ]
    features_df = df[cols]

    print(f"[make_features] Przygotowano cechy dla {len(features_df)} wierszy.")
    return features_df


def analyze_feature_importance(
    train_data: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
    select_k_best: int,
) -> dict:
    """Selekcja cech ktore cechy realnie wnosza informacje"""

    # analiza na probce
    sample = train_data.sample(min(len(train_data), 100_000), random_state=random_state)
    X = sample[features]
    y = sample[target]

    # waznosci cech
    rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    rf_importances = {
        feat: round(float(imp), 4) for feat, imp in zip(features, rf.feature_importances_)
    }

    # SelectKBest
    k = min(select_k_best, len(features))
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    kbest_scores = {
        feat: round(float(score), 2) for feat, score in zip(features, selector.scores_)
    }
    selected = [feat for feat, keep in zip(features, selector.get_support()) if keep]

    report = {
        "features_analyzed": features,
        "sample_rows": len(sample),
        "random_forest_importance": rf_importances,
        "select_k_best_f_scores": kbest_scores,
        "select_k_best_selected": selected,
        "ranking_by_rf_importance": sorted(rf_importances, key=rf_importances.get, reverse=True),
    }

    print(f"[analyze_feature_importance] Ranking cech wg RF: {report['ranking_by_rf_importance']}")
    return report


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
    label: str,
) -> dict:

    """Podstawowe metryki modelu"""
    y_true = test_data[target]
    y_pred = model.predict(test_data[features])

    metrics = {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }

    mlflow.log_metrics({
        f"{label}_mae": metrics["mae"],
        f"{label}_rmse": metrics["rmse"],
        f"{label}_r2": metrics["r2"],
    })

    print(f"[evaluate_model] {label} - {metrics}")
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
