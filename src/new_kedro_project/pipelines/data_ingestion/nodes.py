import os
from datetime import datetime
from pathlib import Path


def download_dataset(parameters: dict) -> dict:
    """Pobiera dataset z Kaggle do data/01_raw, jeśli plików nie ma lokalnie.

    Zwraca raport z informacją, które pliki były już obecne, a które pobrano.
    Raport jest jednocześnie sygnałem zależności dla pipeline'u data_preparation,
    dzięki czemu Kedro wymusza kolejność: najpierw pobranie, potem czyszczenie.
    """
    dataset = parameters["dataset"]
    expected_files = parameters["expected_files"]
    raw_dir = Path(parameters["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    present = [f for f in expected_files if (raw_dir / f).exists()]
    missing = [f for f in expected_files if not (raw_dir / f).exists()]

    if not missing:
        print(f"[download_dataset] Wszystkie pliki ({len(present)}) są lokalnie, pomijam pobieranie.")
        downloaded = []
    else:
        print(f"[download_dataset] Brakuje {len(missing)} plików: {missing}. Pobieram z Kaggle...")
        _download_from_kaggle(dataset, raw_dir, parameters.get("credentials_dir"))
        downloaded = missing

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "raw_dir": str(raw_dir),
        "already_present": present,
        "downloaded": downloaded,
        "file_sizes_bytes": {
            f: (raw_dir / f).stat().st_size if (raw_dir / f).exists() else None
            for f in expected_files
        },
    }
    available = sum(1 for size in report["file_sizes_bytes"].values() if size)
    print(f"[download_dataset] Gotowe. Dostępnych plików: {available}/{len(expected_files)}.")
    return report


def _download_from_kaggle(dataset: str, raw_dir: Path, credentials_dir: str | None) -> None:
    if credentials_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = str(Path(credentials_dir).resolve())

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError("Brak pakietu 'kaggle'. Zainstaluj go: pip install kaggle") from exc

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(raw_dir), unzip=True)
