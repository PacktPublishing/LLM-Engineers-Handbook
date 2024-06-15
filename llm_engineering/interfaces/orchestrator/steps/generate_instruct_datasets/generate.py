from typing_extensions import Annotated
from zenml import step

from llm_engineering.application.dataset.generation import DatasetGenerator
from llm_engineering.domain.dataset import InstructDataset
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory


@step
def generate(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"],
) -> Annotated[dict[DataCategory, InstructDataset], "instruct_datasets"]:
    instruct_datasets = DatasetGenerator.generate(prompts)

    return instruct_datasets
