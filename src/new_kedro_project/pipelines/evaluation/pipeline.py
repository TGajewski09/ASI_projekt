from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compare_all_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compare_all_models,
            inputs=[
                "metrics_lr",
                "metrics_rf",
                "engineered_metrics_rf",
                "tuned_metrics_rf",
                "automl_metrics",
            ],
            outputs="final_model_comparison",
            name="compare_all_models_node",
        ),
    ])
