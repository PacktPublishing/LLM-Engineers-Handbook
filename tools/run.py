from datetime import datetime as dt
from pathlib import Path

import click
from loguru import logger

from llm_engineering import settings
from pipelines import (
    digital_data_etl,
    end_to_end_data,
    evaluating,
    export_artifact_to_json,
    feature_engineering,
    generate_datasets,
    training,
)


@click.command(
    help="""
LLM Engineering project CLI v0.0.1. 

Main entry point for the pipeline execution. 
This entrypoint is where everything comes together.

Run the ZenML LLM Engineering project pipelines with various options.

Run a pipeline with the required parameters. This executes
all steps in the pipeline in the correct order using the orchestrator
stack component that is configured in your active ZenML stack.

Examples:

  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache
  
  \b
  # Run only the ETL pipeline
  python run.py --only-etl

"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-end-to-end-data",
    is_flag=True,
    default=False,
    help="Whether to run all the data pipelines in one go.",
)
@click.option(
    "--run-etl",
    is_flag=True,
    default=False,
    help="Whether to run the ETL pipeline.",
)
@click.option(
    "--run-export-artifact-to-json",
    is_flag=True,
    default=False,
    help="Whether to run the Artifact -> JSON pipeline",
)
@click.option(
    "--etl-config-filename",
    default="digital_data_etl_paul_iusztin.yaml",
    help="Filename of the ETL config file.",
)
@click.option(
    "--run-feature-engineering",
    is_flag=True,
    default=False,
    help="Whether to run the FE pipeline.",
)
@click.option(
    "--run-generate-instruct-datasets",
    is_flag=True,
    default=False,
    help="Whether to run the instruct dataset generation pipeline.",
)
@click.option(
    "--run-generate-preference-datasets",
    is_flag=True,
    default=False,
    help="Whether to run the preference dataset generation pipeline.",
)
@click.option(
    "--run-training",
    is_flag=True,
    default=False,
    help="Whether to run the training pipeline.",
)
@click.option(
    "--run-evaluation",
    is_flag=True,
    default=False,
    help="Whether to run the evaluation pipeline.",
)
@click.option(
    "--export-settings",
    is_flag=True,
    default=False,
    help="Whether to export your settings to ZenML or not.",
)
def main(
    no_cache: bool = False,
    run_end_to_end_data: bool = False,
    run_etl: bool = False,
    etl_config_filename: str = "digital_data_etl_paul_iusztin.yaml",
    run_export_artifact_to_json: bool = False,
    run_feature_engineering: bool = False,
    run_generate_instruct_datasets: bool = False,
    run_generate_preference_datasets: bool = False,
    run_training: bool = False,
    run_evaluation: bool = False,
    export_settings: bool = False,
) -> None:
    assert (
        run_end_to_end_data
        or run_etl
        or run_export_artifact_to_json
        or run_feature_engineering
        or run_generate_instruct_datasets
        or run_generate_preference_datasets
        or run_training
        or run_evaluation
        or export_settings
    ), "Please specify an action to run."

    if export_settings:
        logger.info("Exporting settings to ZenML secrets.")
        settings.export()

    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent

    if run_end_to_end_data:
        run_args_end_to_end = {}
        pipeline_args["config_path"] = root_dir / "configs" / "end_to_end_data.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"end_to_end_data_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        end_to_end_data.with_options(**pipeline_args)(**run_args_end_to_end)

    if run_etl:
        run_args_etl = {}
        pipeline_args["config_path"] = root_dir / "configs" / etl_config_filename
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"digital_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        digital_data_etl.with_options(**pipeline_args)(**run_args_etl)

    if run_export_artifact_to_json:
        run_args_etl = {}
        pipeline_args["config_path"] = root_dir / "configs" / "export_artifact_to_json.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"export_artifact_to_json_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        export_artifact_to_json.with_options(**pipeline_args)(**run_args_etl)

    if run_feature_engineering:
        run_args_fe = {}
        pipeline_args["config_path"] = root_dir / "configs" / "feature_engineering.yaml"
        pipeline_args["run_name"] = f"feature_engineering_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        feature_engineering.with_options(**pipeline_args)(**run_args_fe)

    if run_generate_instruct_datasets:
        run_args_cd = {}
        pipeline_args["config_path"] = root_dir / "configs" / "generate_instruct_datasets.yaml"
        pipeline_args["run_name"] = f"generate_instruct_datasets_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        generate_datasets.with_options(**pipeline_args)(**run_args_cd)

    if run_generate_preference_datasets:
        run_args_cd = {}
        pipeline_args["config_path"] = root_dir / "configs" / "generate_preference_datasets.yaml"
        pipeline_args["run_name"] = f"generate_preference_datasets_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        generate_datasets.with_options(**pipeline_args)(**run_args_cd)

    if run_training:
        run_args_cd = {}
        pipeline_args["config_path"] = root_dir / "configs" / "training.yaml"
        pipeline_args["run_name"] = f"training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        training.with_options(**pipeline_args)(**run_args_cd)

    if run_evaluation:
        run_args_cd = {}
        pipeline_args["config_path"] = root_dir / "configs" / "evaluating.yaml"
        pipeline_args["run_name"] = f"evaluation_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        evaluating.with_options(**pipeline_args)(**run_args_cd)


if __name__ == "__main__":
    main()
