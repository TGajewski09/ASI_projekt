"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines(raise_errors=True)
    pipelines["__default__"] = (
        pipelines["data_ingestion"]
        + pipelines["data_preparation"]
        + pipelines["data_modeling"]
    )
    return pipelines
