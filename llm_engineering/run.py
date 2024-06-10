from datetime import datetime as dt
from pathlib import Path

import click

from llm_engineering.pipelines import digital_data_etl


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
    "--only-etl",
    is_flag=True,
    default=False,
    help="Whether to run only ETL pipeline.",
)
def main(
    no_cache: bool = False,
    only_etl: bool = False,
) -> None:
    pipeline_args = {
        "enable_cache": not no_cache,
    }

    # Execute digital data ETL
    run_args_etl = {}
    pipeline_args["config_path"] = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "digital_data_etl_paul_iusztin.yaml"
    )
    pipeline_args["run_name"] = (
        f"digital_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )
    digital_data_etl.with_options(**pipeline_args)(**run_args_etl)
    if only_etl is True:
        return


if __name__ == "__main__":
    main()
