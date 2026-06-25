FROM python:3.13-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/new_kedro_project/pipelines/serve.py ./serve.py

# Male artefakty trafiaja do obrazu, a duzy model AutoGluon montujemy dopiero
# w runtime przez wolumen (patrz docker-compose.yml: ./data/06_models/autogluon).
# Dzieki temu obraz buduje sie szybko i bez kilku GB modelu w srodku.
COPY data/06_models/country_label_encoder.json ./data/06_models/country_label_encoder.json
COPY data/06_models/drift_baseline.json ./data/06_models/drift_baseline.json

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
