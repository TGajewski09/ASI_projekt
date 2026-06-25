from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_autogluon_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_autogluon_model,
            inputs=[
                "engineered_train_data",
                "engineered_test_data",
                "params:automl",
            ],
            outputs=["automl_metrics", "automl_leaderboard"],
            name="train_autogluon_model_node",
        ),
    ])
