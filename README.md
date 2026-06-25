# Predykcja średniej temperatury - projekt ASI

## Opis problemu

System przewiduje średnią miesięczną temperaturę dla podanego roku, miesiąca
i współrzędnych geograficznych. Jest to problem regresji, ponieważ model zwraca
wartość liczbową w stopniach Celsjusza.

Przykładowe wejście do API:

```text
year=2013
month=7
latitude=52.23
longitude=21.0
```

Wynikiem jest przewidywana średnia temperatura oraz informacja, czy dane wejściowe
wychodzą poza zakres danych treningowych.

## Dane

W projekcie wykorzystano dataset
[Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
udostępniony przez Berkeley Earth. Głównym plikiem używanym w pipeline jest
`GlobalLandTemperaturesByCity.csv`.

Najważniejsze kolumny:

| Kolumna | Znaczenie |
|---|---|
| `dt` | data pomiaru |
| `AverageTemperature` | średnia temperatura, czyli wartość przewidywana przez model |
| `AverageTemperatureUncertainty` | niepewność pomiaru |
| `City`, `Country` | miasto i kraj |
| `Latitude`, `Longitude` | współrzędne geograficzne zapisane tekstowo |

Po czyszczeniu i odfiltrowaniu danych starszych niż 1850 rok zostaje
`6 695 755` wierszy. Pipeline przekształca współrzędne na wartości liczbowe
i tworzy cechy używane później przez modele.

## Architektura

Projekt składa się z kilku części:

| Część | Technologia | Rola |
|---|---|---|
| Pipeline danych i modeli | Kedro | uruchamianie etapów ML |
| Eksperymenty | MLflow | zapisywanie metryk i artefaktów |
| AutoML | AutoGluon | trening i wybór najlepszego modelu |
| API | FastAPI + Uvicorn | endpoint do predykcji temperatury |
| Monitoring | Prometheus | metryki API i wykrywanie driftu |
| Uruchomienie produkcyjne | Docker Compose | API oraz Prometheus lokalnie |
| Automatyzacja | GitHub Actions | CI, CD i Continuous Training |

Diagram architektury znajduje się w katalogu `docs/`:

```text
docs/architektura.md
docs/architectura.png
```

## Pipeline ML

Repozytorium zawiera kilka pipeline'ów Kedro.

Domyślny pipeline:

```text
data_ingestion -> data_preparation -> data_modeling -> feature_engineering
```

Pełny pipeline:

```text
data_ingestion -> data_preparation -> data_modeling -> feature_engineering -> automl -> evaluation
```

Opis etapów:

| Pipeline | Rola |
|---|---|
| `data_ingestion` | pobiera dane z Kaggle albo korzysta z plików dostępnych lokalnie |
| `data_preparation` | usuwa braki, usuwa duplikaty, przelicza współrzędne i filtruje dane od 1850 roku |
| `data_modeling` | trenuje modele bazowe: LinearRegression i RandomForest |
| `feature_engineering` | tworzy cechy `year`, `month`, `decade`, `Latitude`, `Longitude`, `abs_latitude`, wykonuje selekcję cech i strojenie RandomForest |
| `automl` | trenuje model AutoGluon na danych po `feature_engineering` |
| `evaluation` | tworzy wspólny ranking modeli |

AutoGluon zapisuje model do:

```text
data/06_models/autogluon
```

Pipeline `feature_engineering` generuje też plik:

```text
data/06_models/drift_baseline.json
```

Ten plik zawiera zakresy cech liczbowych ze zbioru treningowego i jest używany
przez API do prostego wykrywania driftu danych.

## Wyniki

Modele bazowe są porównywane na podstawie metryk MAE, RMSE i R2.
W projekcie trenowane są:

| Model | Miejsce w projekcie |
|---|---|
| LinearRegression | `data_modeling` |
| RandomForest | `data_modeling` |
| RandomForest z dodatkowymi cechami | `feature_engineering` |
| RandomForest po strojeniu | `feature_engineering` |
| AutoGluon | `automl` |

Końcowy ranking modeli jest zapisywany w:

```text
data/08_reporting/final_model_comparison.json
```

API korzysta z modelu AutoGluon zapisanego w `data/06_models/autogluon`.

## Struktura repozytorium

```text
ASI_projekt/
├── .github/workflows/          # CI, CD i Continuous Training
├── conf/                       # konfiguracja Kedro, katalog danych i parametry
├── data/                       # dane i artefakty pipeline'u
│   ├── 01_raw/                 # surowe dane z Kaggle
│   ├── 02_intermediate/        # dane po etapach czyszczenia
│   ├── 03_primary/             # główny oczyszczony zbiór
│   ├── 04_feature/             # dane z cechami
│   ├── 05_model_input/         # zbiory train/test
│   ├── 06_models/              # modele i pliki API
│   └── 08_reporting/           # metryki i raporty
├── docs/                       # dokumentacja i diagram architektury
├── notebooks/                  # notebook EDA
├── scripts/                    # skrypty pomocnicze
├── src/new_kedro_project/      # kod projektu Kedro i API
│   └── pipelines/              # pipeline'y Kedro
├── tests/                      # testy pytest
├── Dockerfile                  # obraz API
├── docker-compose.yml          # API + Prometheus
├── requirements.txt            # zależności główne
├── requirements-automl.txt     # zależności AutoML
└── requirements-serve.txt      # zależności API
```

## Uruchomienie całego flow

Komendy poniżej są przygotowane dla Windows PowerShell.

### 1. Wejście do projektu

```powershell
cd ...\ASI_projekt
```

### 2. Utworzenie i aktywacja środowiska

Zalecany jest Python 3.11 lub 3.12. Projekt był uruchamiany na Pythonie 3.12.
Nie należy używać Pythona 3.14, ponieważ zależności AutoGluon mogą się wtedy
nie zainstalować.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Jeżeli Python 3.12 nie jest dostępny, można użyć 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

### 3. Instalacja zależności

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-automl.txt
```

Zależności do API są instalowane osobno w obrazie Dockera z pliku
`requirements-serve.txt`.

### 4. Logowanie do Kaggle

Pipeline pobiera dane z Kaggle, jeżeli plików nie ma jeszcze w `data/01_raw`.

Dane indywidualne do zalogowania do Kaggle należy uzupełnić w pliku `conf/local/kaggle_temp.json`

```powershell
$env:KAGGLE_CONFIG_DIR = "$PWD\conf\local"
kaggle auth login
```

Po zalogowaniu w przeglądarce komenda powinna zakończyć się komunikatem
o poprawnym logowaniu.

Zamiast logowania można też ręcznie pobrać dataset z Kaggle i umieścić pliki:

```text
data/01_raw/GlobalLandTemperaturesByCity.csv
data/01_raw/GlobalTemperatures.csv
```

### 5. Uruchomienie pełnego pipeline'u

Najprostsza komenda do uruchomienia całego przepływu:

```powershell
kedro run --pipeline full
```

Pipeline `full` wykonuje pobranie danych, preprocessing, modele bazowe,
feature engineering, AutoML i końcową ewaluację.

AutoGluon może trenować kilka minut. Limit czasu jest ustawiony w:

```text
conf/base/parameters_automl.yml
```

Aktualne najważniejsze parametry:

```yaml
sample_size: 100000
test_sample_size: 50000
time_limit: 300
```

Do szybkiego testu można tymczasowo zmniejszyć `time_limit`, na przykład do `60`.

### 6. Sprawdzenie artefaktów po pipeline

Po zakończeniu pipeline'u warto sprawdzić, czy powstały najważniejsze pliki:

```powershell
Test-Path data\06_models\autogluon
Test-Path data\06_models\drift_baseline.json
Test-Path data\08_reporting\automl_metrics.json
Test-Path data\08_reporting\final_model_comparison.json
```

Każda komenda powinna zwrócić `True`.

### 7. MLflow

Podgląd eksperymentów:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Adres:

```text
http://127.0.0.1:5000
```

### 8. Kedro-Viz

Podgląd pipeline'ów:

```powershell
kedro viz
```

### 9. Uruchomienie API i monitoringu

API jest uruchamiane przez Docker Compose. Przed tym krokiem musi istnieć model
`data/06_models/autogluon` oraz plik `data/06_models/drift_baseline.json`.

```powershell
docker compose up --build
```

Po starcie dostępne są:

```text
API:        http://localhost:8000/docs
Prometheus: http://localhost:9090
```

### 10. Zapytanie do API

W drugim terminalu:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/predict?year=2013&month=7&latitude=52.23&longitude=21.0"
```

Przykład zapytania z danymi spoza zakresu treningowego:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/predict?year=1700&month=7&latitude=52.23&longitude=21.0"
```

W drugim przypadku API powinno zwrócić informację o wykrytym drifcie.

### 11. Zatrzymanie Dockera

```powershell
docker compose down
```

## Przydatne komendy Kedro

Uruchomienie domyślnego pipeline'u bez AutoML i końcowej ewaluacji:

```powershell
kedro run
```

Uruchomienie samego AutoML, gdy dane po `feature_engineering` już istnieją:

```powershell
kedro run --pipeline automl
```

Uruchomienie samej końcowej ewaluacji, gdy metryki modeli już istnieją:

```powershell
kedro run --pipeline evaluation
```

## Testy i linting

Instalacja narzędzi testowych:

```powershell
pip install pytest pytest-mock pytest-cov ruff
```

Uruchomienie testów i lintingu:

```powershell
pytest
ruff check .
```

## MLOps

W katalogu `.github/workflows/` znajdują się trzy workflowy:

| Workflow | Rola |
|---|---|
| `ci.yml` | uruchamia testy i linting przy pushu oraz pull requeście |
| `cd.yml` | buduje obraz Dockera z API i publikuje go w GHCR po zmianach na `main` |
| `continuous-training.yml` | cyklicznie lub ręcznie uruchamia retrening na próbce danych |

Continuous Training korzysta ze skryptu:

```text
scripts/retrain.py
```

Jest to lekki retrening demonstracyjny na próbce z `data/sample`, żeby workflow
mógł działać szybko w GitHub Actions.

## Demo projektu

Najprostsze demo działania projektu:

```powershell
kedro run --pipeline full
docker compose up --build
Invoke-RestMethod -Method Post "http://localhost:8000/predict?year=2013&month=7&latitude=52.23&longitude=21.0"
Invoke-RestMethod -Method Post "http://localhost:8000/predict?year=1700&month=7&latitude=52.23&longitude=21.0"
```

Pierwsze zapytanie pokazuje zwykłą predykcję. Drugie pokazuje wykrywanie driftu,
bo rok 1700 jest poza zakresem danych treningowych.
