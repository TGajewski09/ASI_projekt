from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    filter_modern_era,
    generate_report,
    parse_coordinates,
    remove_duplicates,
    remove_nulls,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=remove_nulls,
            inputs=["temperatures_raw", "ingestion_report"],
            outputs="temperatures_no_nulls",
            name="remove_nulls_node",
        ),
        node(
            func=remove_duplicates,
            inputs="temperatures_no_nulls",
            outputs="temperatures_no_duplicates",
            name="remove_duplicates_node",
        ),
        node(
            func=parse_coordinates,
            inputs="temperatures_no_duplicates",
            outputs="temperatures_coords_parsed",
            name="parse_coordinates_node",
        ),
        node(
            func=filter_modern_era,
            inputs=["temperatures_coords_parsed", "params:start_year"],
            outputs="temperatures_primary",
            name="filter_modern_era_node",
        ),
        node(
            func=generate_report,
            inputs=[
                "temperatures_raw",
                "temperatures_no_nulls",
                "temperatures_no_duplicates",
                "temperatures_primary",
                "params:start_year",
            ],
            outputs="data_preparation_report",
            name="generate_report_node",
        ),
    ])
