from zenml import pipeline
from zenml.client import Client

from llm_engineering.interfaces.orchestrator.steps import training as training_steps


@pipeline
def training() -> None:

    instruct_datasets = Client().get_artifact_version(name_id_or_prefix="instruct_datasets")
    
    training_steps.tokenize(instruct_datasets=instruct_datasets)
