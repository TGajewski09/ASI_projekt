# Architektura systemu

Dokument opisuje przepływ danych i sposób uruchomienia projektu: od pobrania
danych z Kaggle, przez trening modeli, po API z monitoringiem.

Diagram do edycji znajduje się w pliku `architecture.drawio`, a wersja podglądowa
w pliku `architecture.svg`.


## Komponenty

| Komponent | Technologia | Rola |
|---|---|---|
| Pobieranie danych | Kaggle API | pobranie datasetu do `data/01_raw` |
| Pipeline ML | Kedro | uruchamianie etapów przygotowania danych i treningu |
| Eksperymenty | MLflow | zapisywanie metryk i artefaktów |
| AutoML | AutoGluon | trening modelu na danych po `feature_engineering` |
| API | FastAPI + Uvicorn | endpoint `POST /predict` |
| Monitoring | Prometheus | metryki API i licznik driftu |
| Konteneryzacja | Docker Compose | lokalne uruchomienie API i Prometheusa |
| Automatyzacja | GitHub Actions | testy, budowa obrazu i retrening |

## Przepływ pipeline'ów

Domyślne `kedro run` uruchamia:

```text
data_ingestion -> data_preparation -> data_modeling -> feature_engineering
```

Pełny pipeline `full` uruchamia:

```text
data_ingestion -> data_preparation -> data_modeling -> feature_engineering -> automl -> evaluation
```

Etapy:

| Pipeline | Opis |
|---|---|
| `data_ingestion` | pobiera pliki z Kaggle albo korzysta z danych dostępnych lokalnie |
| `data_preparation` | usuwa braki, usuwa duplikaty, przelicza współrzędne i filtruje dane od 1850 roku |
| `data_modeling` | trenuje LinearRegression i RandomForest |
| `feature_engineering` | tworzy dodatkowe cechy, stroi RandomForest i generuje `drift_baseline.json` |
| `automl` | trenuje AutoGluon na danych `engineered_train_data` |
| `evaluation` | porównuje wszystkie modele i zapisuje końcowy ranking |

## Warstwy danych

```text
01_raw -> 02_intermediate -> 03_primary -> 04_feature -> 05_model_input -> 06_models -> 08_reporting
surowe    częściowo czyste   gotowe do      cechy        train / test     modele       raporty
                             modelowania
```

Najważniejsze artefakty:

| Artefakt | Rola |
|---|---|
| `data/03_primary/temperatures_primary.parquet` | oczyszczony zbiór wejściowy |
| `data/05_model_input/engineered_train_data.parquet` | dane treningowe z dodatkowymi cechami |
| `data/05_model_input/engineered_test_data.parquet` | dane testowe z dodatkowymi cechami |
| `data/06_models/autogluon` | zapisany model AutoGluon |
| `data/06_models/drift_baseline.json` | zakresy cech używane do wykrywania driftu |
| `data/08_reporting/final_model_comparison.json` | końcowy ranking modeli |

## Decyzje architektoniczne

- Trening jest wsadowy. Model jest trenowany wcześniej, a API wykonuje tylko
  predykcję dla pojedynczego zapytania.
- AutoGluon działa po `feature_engineering`, dzięki czemu korzysta z tych samych
  cech co API: `year`, `month`, `decade`, `Latitude`, `Longitude`, `abs_latitude`.
- Model AutoGluon nie jest kopiowany bezpośrednio do obrazu Dockera. Katalog
  `data/06_models/autogluon` jest podłączany przez wolumen w `docker-compose.yml`.
- API ładuje model dopiero przy pierwszym zapytaniu. Dzięki temu testy mogą
  importować moduł API bez pełnego modelu na dysku.
- Cecha `Country` nie jest używana w modelu końcowym. Lokalizacja jest opisana
  przez `Latitude` i `Longitude`.

## Monitoring i drift

API wystawia metryki Prometheus na `/metrics`:

| Metryka | Znaczenie |
|---|---|
| `model_predictions_total` | liczba wykonanych predykcji |
| `model_prediction_value_celsius` | rozkład przewidywanych temperatur |
| `model_prediction_latency_seconds` | czas odpowiedzi API |
| `model_input_drift_total` | liczba wejść spoza zakresu treningowego |

Plik `drift_baseline.json` jest generowany przez node
`generate_drift_baseline_node` w pipeline `feature_engineering`. Zawiera minimalne
i maksymalne wartości cech liczbowych ze zbioru treningowego.

Przy każdym zapytaniu API sprawdza, czy `year`, `month`, `Latitude` i `Longitude`
mieszczą się w tych zakresach. Jeżeli nie, odpowiedź zawiera informację
o wykrytym drifcie.

## Uruchomienie pełnego flow

Komendy dla Windows PowerShell:

```powershell
cd C:\DEVELOPING\STUDIA\ASI2\ASI_projekt
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-automl.txt
```

Logowanie do Kaggle:

```powershell
$env:KAGGLE_CONFIG_DIR = "$PWD\conf\local"
kaggle auth login
```

Uruchomienie pełnego pipeline'u:

```powershell
kedro run --pipeline full
```

Sprawdzenie artefaktów:

```powershell
Test-Path data\06_models\autogluon
Test-Path data\06_models\drift_baseline.json
Test-Path data\08_reporting\automl_metrics.json
Test-Path data\08_reporting\final_model_comparison.json
```

Uruchomienie API i Prometheusa:

```powershell
docker compose up --build
```

Zapytanie do API:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/predict?year=2013&month=7&latitude=52.23&longitude=21.0"
```

Zapytanie pokazujące drift:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/predict?year=1700&month=7&latitude=52.23&longitude=21.0"
```

Adresy po uruchomieniu Dockera:

```text
API:        http://localhost:8000/docs
Prometheus: http://localhost:9090
```
