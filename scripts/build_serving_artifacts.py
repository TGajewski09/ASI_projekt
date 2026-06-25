"""Buduje pliki pomocnicze potrzebne API (serve.py).

Z gotowego zbioru `temperatures_primary` (wynik pipeline'u data_preparation)
tworzy dwa pliki:

1. country_label_encoder.json  - slownik {nazwa kraju: numer},
   dokladnie taki sam jak `country_label` uzyte przy trenowaniu modelu
   (pd.factorize na kolumnie Country w kolejnosci wystepowania).

2. drift_baseline.json - zakresy (min/max) cech liczbowych ze zbioru
   treningowego oraz lista znanych krajow. API uzywa tego do wykrywania
   driftu, czyli zapytan spoza rozkladu danych treningowych.

Uruchomienie (z katalogu projektu):
    python scripts/build_serving_artifacts.py
"""
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = PROJECT_ROOT / "data/03_primary/temperatures_primary.parquet"
OUT_DIR = PROJECT_ROOT / "data/06_models"


def main() -> None:
    if not PRIMARY.exists():
        raise FileNotFoundError(
            f"Brak pliku {PRIMARY}. Najpierw uruchom pipeline: `kedro run`."
        )

    df = pd.read_parquet(PRIMARY)

    # te same cechy co w feature_engineering.make_features
    df["dt"] = pd.to_datetime(df["dt"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month

    # kodowanie krajow w identycznej kolejnosci jak przy trenowaniu modelu
    _, uniques = pd.factorize(df["Country"])
    encoder = {country: idx for idx, country in enumerate(uniques)}

    baseline = {
        "numeric": {
            "year": {"min": int(df["year"].min()), "max": int(df["year"].max())},
            "month": {"min": int(df["month"].min()), "max": int(df["month"].max())},
            "Latitude": {
                "min": float(df["Latitude"].min()),
                "max": float(df["Latitude"].max()),
            },
            "Longitude": {
                "min": float(df["Longitude"].min()),
                "max": float(df["Longitude"].max()),
            },
        },
        "known_countries": sorted(encoder),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "country_label_encoder.json").write_text(
        json.dumps(encoder, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "drift_baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[build_serving_artifacts] Zapisano enkoder dla {len(encoder)} krajow.")
    print(f"[build_serving_artifacts] Zakresy cech: {baseline['numeric']}")


if __name__ == "__main__":
    main()
