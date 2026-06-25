"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines(raise_errors=True)
    default_pipeline = (
        pipelines["data_ingestion"]
        + pipelines["data_preparation"]
        + pipelines["data_modeling"]
        + pipelines["feature_engineering"]
    )
    pipelines["__default__"] = default_pipeline
    pipelines["full"] = default_pipeline + pipelines["automl"] + pipelines["evaluation"]
    return pipelines
