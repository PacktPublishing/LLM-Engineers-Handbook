from typing import Any

from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step

from llm_engineering.application.dataset.generation import DatasetGenerator
from llm_engineering.domain.dataset import TrainTestSplit
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory


@step
def generate(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"],
    test_split_size: Annotated[float, "test_split_size"],
    mock: Annotated[bool, "mock_generation"] = False,
) -> Annotated[
    TrainTestSplit,
    ArtifactConfig(
        name="instruct_datasets",
        tags=["dataset", "instruct", "cleaned"],
    ),
]:
    instruct_datasets = DatasetGenerator.generate(prompts, test_size=test_split_size, mock=mock)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="instruct_datasets", metadata=_get_metadata(instruct_datasets))

    return instruct_datasets


def _get_metadata(instruct_datasets: TrainTestSplit) -> dict[str, Any]:
    instruct_dataset_categories = list(instruct_datasets.train.keys())
    train_num_samples = {
        category: instruct_dataset.num_samples for category, instruct_dataset in instruct_datasets.train.items()
    }
    test_num_samples = {
        category: instruct_dataset.num_samples for category, instruct_dataset in instruct_datasets.test.items()
    }

    return {
        "data_categories": instruct_dataset_categories,
        "train_num_samples": train_num_samples,
        "test_num_samples": test_num_samples,
        "test_split_size": instruct_datasets.test_split_size,
    }
