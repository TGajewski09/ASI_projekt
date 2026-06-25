from datetime import datetime

import pandas as pd


def remove_nulls(df: pd.DataFrame, _ingestion_signal: dict) -> pd.DataFrame:
    """Usuwa wiersze gdzie brakuje temperatury."""
    before = len(df)
    df = df.dropna(subset=["AverageTemperature"])
    after = len(df)
    print(f"[remove_nulls] Usunięto {before - after} wierszy ({(before - after) / before * 100:.2f}%). Zostało: {after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa duplikaty na podstawie (dt, City, Country)."""
    before = len(df)
    df = df.drop_duplicates(subset=["dt", "City", "Country"])
    after = len(df)
    print(f"[remove_duplicates] Usunięto {before - after} duplikatów. Zostało: {after}")
    return df


def parse_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Zamienia współrzędne z tekstu (np. '57.05N') na liczby ze znakiem."""

    def _to_float(value: str, neg_suffix: str) -> float:
        # S i W oznaczają wartości ujemne
        sign = -1.0 if value.endswith(neg_suffix) else 1.0
        return sign * float(value[:-1])

    df = df.copy()
    df["Latitude"] = df["Latitude"].apply(lambda v: _to_float(v, "S"))
    df["Longitude"] = df["Longitude"].apply(lambda v: _to_float(v, "W"))
    print(f"[parse_coordinates] Zamieniono współrzędne dla {len(df)} wierszy.")
    return df


def filter_modern_era(df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Zostawia tylko dane od start_year wzwyż."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    before = len(df)
    df = df[df["dt"].dt.year >= start_year]
    after = len(df)
    print(f"[filter_modern_era] Zostało {after} wierszy (>= {start_year}). Odrzucono {before - after}.")
    return df


def generate_report(
    raw: pd.DataFrame,
    no_nulls: pd.DataFrame,
    no_duplicates: pd.DataFrame,
    primary: pd.DataFrame,
    start_year: int,
) -> dict:
    """Tworzy raport podsumowujący ile wierszy odrzucono na każdym kroku."""
    raw_count = len(raw)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "steps": [
            {
                "step": "raw",
                "rows": raw_count,
                "dropped": 0,
                "dropped_pct": 0.0,
            },
            {
                "step": "remove_nulls",
                "rows": len(no_nulls),
                "dropped": raw_count - len(no_nulls),
                "dropped_pct": round((raw_count - len(no_nulls)) / raw_count * 100, 2),
            },
            {
                "step": "remove_duplicates",
                "rows": len(no_duplicates),
                "dropped": len(no_nulls) - len(no_duplicates),
                "dropped_pct": round((len(no_nulls) - len(no_duplicates)) / raw_count * 100, 2),
            },
            {
                "step": f"filter_modern_era (>= {start_year})",
                "rows": len(primary),
                "dropped": len(no_duplicates) - len(primary),
                "dropped_pct": round((len(no_duplicates) - len(primary)) / raw_count * 100, 2),
            },
        ],
        "total_dropped": raw_count - len(primary),
        "total_dropped_pct": round((raw_count - len(primary)) / raw_count * 100, 2),
        "final_rows": len(primary),
        "final_unique_cities": int(primary["City"].nunique()),
        "final_unique_countries": int(primary["Country"].nunique()),
        "date_range": {
            "from": str(primary["dt"].min()),
            "to": str(primary["dt"].max()),
        },
    }
    print(f"[generate_report] Raport gotowy. Końcowy zbiór: {len(primary)} wierszy.")
    return report
