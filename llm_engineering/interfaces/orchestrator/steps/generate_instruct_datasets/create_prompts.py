from typing_extensions import Annotated
from zenml import step

from llm_engineering.application.dataset import generation
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory


@step
def create_prompts(
    documents: Annotated[list, "cleaned_documents"],
) -> Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompt_templates"]:
    grouped_prompts = generation.DatasetGenerator.get_prompts(documents)

    return grouped_prompts
