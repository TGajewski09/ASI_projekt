# Predykcja średniej temperatury - projekt ASI

Projekt zaliczeniowy na przedmiot **ASI (Architektury rozwiązań SI)**. Celem projektu jest zbudowanie kompletnego systemu uczenia maszynowego: od pobrania i przygotowania danych, przez trenowanie oraz porównywanie modeli, aż po wystawienie API, monitoring i podstawową automatyzację MLOps.

## 1. Opis problemu

Celem systemu jest **przewidywanie średniej miesięcznej temperatury** dla wybranego miejsca i czasu.

Taki model może zostać wykorzystany na przykład do uzupełniania brakujących pomiarów historycznych albo jako uproszczony mechanizm szacowania temperatury dla wskazanej lokalizacji.

## 2. Opis danych

W projekcie wykorzystano dataset **[Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)** udostępniony przez Berkeley Earth. Głównym plikiem używanym w projekcie jest `GlobalLandTemperaturesByCity.csv`, którego rozmiar wynosi około 508 MB.

| Kolumna                         | Znaczenie                                                                         |
| ------------------------------- | --------------------------------------------------------------------------------- |
| `dt`                            | data pomiaru                                                                      |
| `AverageTemperature`            | średnia temperatura w stopniach Celsjusza, czyli wartość przewidywana przez model |
| `AverageTemperatureUncertainty` | niepewność pomiaru                                                                |
| `City`, `Country`               | miasto i kraj                                                                     |
| `Latitude`, `Longitude`         | współrzędne geograficzne zapisane tekstowo, np. `57.05N`                          |

Po oczyszczeniu danych w zbiorze pozostaje **6 695 755 wierszy** ze **159 krajów**. Zakres danych obejmuje lata **1850-2013**.

Proces czyszczenia danych wygląda następująco:

| Krok                                         | Liczba wierszy |
| -------------------------------------------- | -------------- |
| surowe dane                                  | 8 599 212      |
| po usunięciu braków temperatury              | 8 235 082      |
| po usunięciu duplikatów                      | 8 190 783      |
| po odfiltrowaniu lat wcześniejszych niż 1850 | **6 695 755**  |

## 3. Architektura systemu

Architektura projektu została opisana na diagramie znajdującym się w pliku `docs/architecture.drawio`.

System obejmuje kilka głównych części: pipeline danych i trenowania modeli, rejestrację eksperymentów w MLflow, API do wykonywania predykcji, konteneryzację z użyciem Dockera oraz monitoring z wykorzystaniem Prometheusa.

## 4. Pipeline ML

Podstawowy przebieg pipeline'u uruchamiany komendą `kedro run` składa się z trzech etapów:

1. **`data_ingestion`** - pobiera dane z Kaggle do katalogu `data/01_raw` albo korzysta z gotowego pliku CSV, jeżeli dane są już dostępne lokalnie.
2. **`data_preparation`** - usuwa braki i duplikaty, konwertuje współrzędne geograficzne na wartości liczbowe oraz filtruje dane od roku 1850.
3. **`data_modeling`** - buduje podstawowy zestaw cech, dzieli dane na zbiór treningowy i testowy, trenuje modele bazowe oraz porównuje ich wyniki.

Dodatkowo projekt zawiera osobno uruchamiane elementy:

* **`feature_engineering`** - tworzy dodatkowe cechy, takie jak `decade` i `abs_latitude`, wykonuje selekcję cech z użyciem RandomForest oraz SelectKBest, a także przeprowadza strojenie hiperparametrów z użyciem `RandomizedSearchCV`.
* **AutoML** - uruchamiany skryptem `python automl_autogluon.py`. Ten etap wykorzystuje AutoGluon i działa w osobnym środowisku.
* **`evaluation`** - tworzy wspólny ranking wszystkich modeli na podstawie metryki RMSE.

Pełna kolejność uruchamiania eksperymentów wygląda następująco:

```bash
kedro run
kedro run --pipeline feature_engineering
python automl_autogluon.py
kedro run --pipeline evaluation
```

## 5. Wyniki

Modele bazowe zostały ocenione na zbiorze testowym zawierającym **1 339 151 wierszy**.

| Model            | MAE      | RMSE     | R²        |
| ---------------- | -------- | -------- | --------- |
| LinearRegression | 6.99     | 8.81     | 0.238     |
| **RandomForest** | **0.99** | **1.39** | **0.981** |

Najlepszy wynik spośród modeli bazowych uzyskał **RandomForest**. Model przewiduje średnią miesięczną temperaturę z błędem około 1 stopnia Celsjusza według MAE. W pełnym porównaniu, łącznie z AutoML, najlepszy okazał się **AutoGluon** (R² ≈ 0.99, RMSE ≈ 1.05) i to on jest serwowany przez API.

Regresja liniowa wypada znacznie słabiej, ponieważ zależność temperatury od lokalizacji, miesiąca i czasu nie ma prostego charakteru liniowego. Temperatura silnie zależy między innymi od sezonowości, szerokości geograficznej oraz lokalnych warunków klimatycznych, dlatego model nieliniowy radzi sobie w tym zadaniu znacznie lepiej.

## 6. Struktura repozytorium

```text
ASI_projekt/
├── conf/                  # konfiguracja Kedro, katalogi danych, parametry i MLflow
├── data/                  # warstwy danych od 01_raw do 08_reporting, ignorowane w gicie
│   └── sample/            # mała próbka danych używana w Continuous Training
├── docs/                  # dokumentacja i diagram architektury
├── notebooks/             # notebooki, między innymi EDA i demo API
├── scripts/               # skrypty pomocnicze, artefakty API i retrening
├── src/new_kedro_project/
│   ├── pipelines/         # pipeline'y Kedro oraz serve.py dla API
│   └── ...
├── tests/                 # testy jednostkowe pytest
├── .github/workflows/     # CI, CD i Continuous Training
├── automl_autogluon.py    # AutoML z użyciem AutoGluon
├── Dockerfile             # obraz API
└── docker-compose.yml     # API oraz Prometheus
```

## 7. Jak uruchomić projekt

### Wymagania

Wymagany jest Python w wersji **3.10 lub nowszej**.

### Instalacja

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Instalacja zależności:

```bash
pip install -r requirements.txt
```

Do pobrania danych z Kaggle wymagany jest token API zapisany w pliku:

```text
conf/local/kaggle.json
```

### Uruchomienie pipeline'u

Podstawowy pipeline:

```bash
kedro run
```

Pipeline z dodatkowymi cechami i strojeniem modelu:

```bash
kedro run --pipeline feature_engineering
```

AutoML z użyciem AutoGluon:

```bash
python automl_autogluon.py
```

Ewaluacja i ranking modeli:

```bash
kedro run --pipeline evaluation
```

Zalecana pełna kolejność uruchomienia:

```bash
kedro run
kedro run --pipeline feature_engineering
python automl_autogluon.py
kedro run --pipeline evaluation
```

### Podgląd eksperymentów w MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Interfejs MLflow będzie dostępny pod adresem:

```text
http://127.0.0.1:5000
```

### Wizualizacja pipeline'u w Kedro-Viz

```bash
kedro viz
```

### API do predykcji temperatury

API można uruchomić przez Docker Compose:

```bash
docker compose up --build
```

Po uruchomieniu dostępne są:

```text
API:        http://localhost:8000/docs
Prometheus: http://localhost:9090
```

Przykładowe zapytanie do API:

```bash
curl -X POST "http://localhost:8000/predict?year=2013&month=7&latitude=52.23&longitude=21.0"
```

### Testy i linting

```bash
pytest
ruff check .
```

## 8. MLOps: CI, CD i Continuous Training

W katalogu `.github/workflows/` znajdują się workflowy GitHub Actions odpowiedzialne za automatyzację projektu:

* **`ci.yml`** - uruchamiany przy każdym pushu i pull requeście. Wykonuje `ruff check` oraz `pytest`.
* **`cd.yml`** - uruchamiany po wejściu zmian na gałąź `main`. Buduje obraz Dockera z API i publikuje go w GitHub Container Registry.
* **`continuous-training.yml`** - uruchamiany ręcznie albo cyklicznie raz w tygodniu. Ponownie trenuje model przy użyciu skryptu `scripts/retrain.py` na próbce danych i zapisuje wynik jako artefakt.

## 9. Demo

Działanie projektu można sprawdzić na dwa sposoby:

* przez **API**, uruchamiając `docker compose up --build` i korzystając z dokumentacji pod adresem `http://localhost:8000/docs`,
* przez **notebook** `notebooks/demo_api.ipynb`, który zawiera przykładowe predykcje oraz prostą demonstrację wykrywania driftu.
