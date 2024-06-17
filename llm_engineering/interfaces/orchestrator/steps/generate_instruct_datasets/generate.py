from typing import Any

from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step

from llm_engineering.application.dataset.generation import DatasetGenerator
from llm_engineering.domain.dataset import InstructDataset
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory


@step
def generate(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"],
) -> Annotated[
    dict[DataCategory, InstructDataset],
    ArtifactConfig(
        name="instruct_datasets",
        tags=["dataset", "instruct", "cleaned"],
    ),
]:
    instruct_datasets = DatasetGenerator.generate(prompts, mock=True)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="instruct_datasets", metadata=_get_metadata(instruct_datasets))

    return instruct_datasets


def _get_metadata(instruct_datasets: dict[DataCategory, InstructDataset]) -> dict[str, Any]:
    instruct_dataset_categories = list(instruct_datasets.keys())
    instruct_dataset_num_samples = {
        category: instruct_dataset.num_samples for category, instruct_dataset in instruct_datasets.items()
    }

    return {"data_categories": instruct_dataset_categories, "data_categories_num_samples": instruct_dataset_num_samples}
