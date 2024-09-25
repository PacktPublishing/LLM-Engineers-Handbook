from typing import Any

from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step

from llm_engineering.application.dataset import generation
from llm_engineering.domain.dataset import DatasetType, InstructTrainTestSplit
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory


@step
def generate_intruction_dataset(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"],
    test_split_size: Annotated[float, "test_split_size"],
    mock: Annotated[bool, "mock_generation"] = False,
) -> Annotated[
    InstructTrainTestSplit,
    ArtifactConfig(
        name="instruct_datasets",
        tags=["dataset", "instruct", "cleaned"],
    ),
]:
    dataset_generator = generation.get_dataset_generator(DatasetType.INSTRUCTION)
    datasets = dataset_generator.generate(prompts, test_size=test_split_size, mock=mock)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="instruct_datasets", metadata=_get_metadata_instruct_dataset(datasets))

    return datasets


def _get_metadata_instruct_dataset(datasets: InstructTrainTestSplit) -> dict[str, Any]:
    instruct_dataset_categories = list(datasets.train.keys())
    train_num_samples = {
        category: instruct_dataset.num_samples for category, instruct_dataset in datasets.train.items()
    }
    test_num_samples = {category: instruct_dataset.num_samples for category, instruct_dataset in datasets.test.items()}

    return {
        "data_categories": instruct_dataset_categories,
        "test_split_size": datasets.test_split_size,
        "train_num_samples_per_category": train_num_samples,
        "test_num_samples_per_category": test_num_samples,
    }
