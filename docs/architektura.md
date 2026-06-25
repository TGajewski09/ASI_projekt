# Architektura systemu

Dokument opisuje, z czego składa się projekt i jak przepływają dane: od
surowego pliku z Kaggle, przez trening modeli, po API z monitoringiem.



## Komponenty

| Komponent | Technologia | Rola |
|---|---|---|
| Pobieranie danych | Kaggle API | ściąga dataset do `data/01_raw` |
| Pipeline ML | Kedro | czyszczenie, cechy, trening i ewaluacja |
| Śledzenie eksperymentów | MLflow (`kedro-mlflow`) | metryki i artefakty modeli |
| AutoML | AutoGluon | trening modelu na danych po `feature_engineering` |
| API | FastAPI + Uvicorn | endpoint `POST /predict` |
| Monitoring | Prometheus | metryki API i wykrywanie driftu |
| Konteneryzacja | Docker, docker-compose | uruchomienie API + Prometheus |
| Automatyzacja | GitHub Actions | CI / CD / Continuous Training |

## Przepływ danych i pipeline'y

Główny pipeline `full` uruchamia kolejne etapy:

```
data_ingestion -> data_preparation -> data_modeling -> feature_engineering -> automl -> evaluation
```

Domyślne `kedro run` uruchamia część bez AutoML i końcowej ewaluacji zbiorczej:

```
data_ingestion -> data_preparation -> data_modeling -> feature_engineering
```

Dane przechodzą przez kolejne katalogi w `data/` zgodnie z konwencją Kedro:

```
01_raw -> 02_intermediate -> 03_primary -> 04_feature -> 05_model_input -> 06_models -> 08_reporting
surowe    częściowo czyste   gotowe do      cechy        train / test     modele       raporty,
                             modelowania                                                metryki
```

- **01_raw** - oryginalny CSV z Kaggle.
- **02_intermediate** - kolejne etapy czyszczenia, bez braków, bez duplikatów
  i z policzonymi współrzędnymi.
- **03_primary** - `temperatures_primary`, czyli czysty zbiór wejściowy do modeli.
- **04_feature / 05_model_input** - dodatkowe cechy oraz podział na zbiór
  treningowy i testowy. Z tych danych korzysta też AutoGluon.
- **06_models** - zapisane modele sklearn, model AutoGluon oraz pliki pomocnicze API.
- **08_reporting** - raporty i metryki, część z nich jest logowana też do MLflow.

## Decyzje architektoniczne

- **Batch, nie real-time.** Trening odbywa się wsadowo na danych historycznych.
  API obsługuje pojedyncze zapytania w czasie rzeczywistym, ale korzysta z modelu
  wytrenowanego wcześniej.
- **AutoML po `feature_engineering`.** AutoGluon dostaje te same cechy, których
  później używa API: `year`, `month`, `decade`, `Latitude`, `Longitude`
  i `abs_latitude`.
- **Model serwowany przez wolumen.** Model AutoGluon nie jest wpisany bezpośrednio
  w obraz Dockera. Obraz zawiera kod API i małe pliki pomocnicze, a katalog
  `data/06_models/autogluon` jest podłączany przez wolumen w `docker-compose.yml`.
- **Leniwe ładowanie modelu.** `serve.py` ładuje model dopiero przy pierwszym
  zapytaniu, więc moduł można importować w testach bez pełnego modelu na dysku.
- **Bez cechy `Country`.** Położenie opisują współrzędne (`Latitude`, `Longitude`),
  więc kraj nie jest używany jako cecha modelu.

## Monitoring i drift

API wystawia metryki Prometheus na `/metrics`:

- `model_predictions_total` - liczba wykonanych predykcji,
- `model_prediction_value_celsius` - rozkład przewidywanych temperatur,
- `model_prediction_latency_seconds` - czas odpowiedzi,
- `model_input_drift_total` - liczba zapytań z danymi spoza zakresu treningowego.

Plik `drift_baseline.json` jest generowany przez node `generate_drift_baseline_node`
w pipeline `feature_engineering` na podstawie `engineered_train_data`. Zawiera
minimalne i maksymalne wartości cech liczbowych używanych w API.

Drift wykrywamy prosto: dla każdego zapytania sprawdzamy, czy rok, miesiąc
i współrzędne mieszczą się w zakresach ze zbioru treningowego. Jeśli nie, API
zwraca informację o wykrytym drifcie i zlicza go w metryce Prometheus.
