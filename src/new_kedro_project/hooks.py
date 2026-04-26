import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kedro.framework.hooks import hook_impl


DATASET = "berkeleyearth/climate-change-earth-surface-temperature-data"

EXPECTED_FILES = [
    "GlobalTemperatures.csv",
    "GlobalLandTemperaturesByCity.csv",
    # "GlobalLandTemperaturesByCountry.csv",
    # "GlobalLandTemperaturesByMajorCity.csv",
    # "GlobalLandTemperaturesByState.csv",
]


class DatasetDownloadHook:
    """Pobiera dane z Kaggle, jeśli nie ma ich lokalnie."""

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog) -> None:
        project_path = Path(run_params.get("project_path", Path.cwd())).resolve()
        raw_data_dir = project_path / "data" / "01_raw"
        missing_files = [
            filename
            for filename in EXPECTED_FILES
            if not (raw_data_dir / filename).exists()
        ]

        if not missing_files:
            print("[download_dataset] Dataset jest już dostępny lokalnie.")
            return

        raw_data_dir.mkdir(parents=True, exist_ok=True)

        local_config_dir = project_path / "conf" / "local"
        kaggle_config_path = local_config_dir / "kaggle.json"

        if not kaggle_config_path.exists():
            raise FileNotFoundError(
                "Brakuje pliku conf/local/kaggle.json. "
                "Wygeneruj token API na Kaggle i zapisz go w tym miejscu."
            )

        os.environ["KAGGLE_CONFIG_DIR"] = str(local_config_dir)

        print("[download_dataset] Pobieranie datasetu z Kaggle...")

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET, path=str(raw_data_dir), unzip=True)

        print("[download_dataset] Dataset pobrany do data/01_raw.")
