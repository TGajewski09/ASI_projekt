from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_features,
    compare_models,
    evaluate_model,
    split_data,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_features,
            inputs="temperatures_primary",
            outputs="temperatures_features",
            name="build_features_node",
        ),
        node(
            func=split_data,
            inputs=["temperatures_features", "params:test_size", "params:random_state"],
            outputs=["train_data", "test_data"],
            name="split_data_node",
        ),
        # --- LinearRegression ---
        node(
            func=train_model,
            inputs=["train_data", "params:features", "params:target", "params:linear_regression"],
            outputs="model_lr",
            name="train_lr_node",
        ),
        node(
            func=evaluate_model,
            inputs=["model_lr", "test_data", "params:features", "params:target"],
            outputs="metrics_lr",
            name="evaluate_lr_node",
        ),
        # --- RandomForestRegressor ---
        node(
            func=train_model,
            inputs=["train_data", "params:features", "params:target", "params:random_forest"],
            outputs="model_rf",
            name="train_rf_node",
        ),
        node(
            func=evaluate_model,
            inputs=["model_rf", "test_data", "params:features", "params:target"],
            outputs="metrics_rf",
            name="evaluate_rf_node",
        ),
        # --- Porównanie wszystkich modeli ---
        node(
            func=compare_models,
            inputs=["metrics_lr", "metrics_rf"],
            outputs="model_comparison",
            name="compare_models_node",
        ),
    ])
