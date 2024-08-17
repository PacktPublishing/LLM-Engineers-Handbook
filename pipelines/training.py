from zenml import pipeline
from zenml.client import Client

from steps import training as training_steps


@pipeline
def training() -> None:
    # NOTE: This is a placeholder pipeline for the training logic.

    # Here is how you can access the generated instruct datasets by the generate_instruct_datasets pipeline.
    # 'instruct_datasets' is the ID of the artifact.
    instruct_datasets = Client().get_artifact_version(name_id_or_prefix="instruct_datasets")

    # Based on that you can retrieve other artifacts such as: raw_documents, cleaned_documents or embedded_documents

    # Here is an example of how to start the training logic with the tokenization step.
    training_steps.tokenize(instruct_datasets=instruct_datasets)
