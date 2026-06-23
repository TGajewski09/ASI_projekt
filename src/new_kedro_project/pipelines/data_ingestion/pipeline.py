from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_dataset,
            inputs="params:kaggle",
            outputs="ingestion_report",
            name="download_dataset_node",
        ),
    ])
