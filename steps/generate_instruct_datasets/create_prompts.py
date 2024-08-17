from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.dataset import generation
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory


@step
def create_prompts(
    documents: Annotated[list, "queried_cleaned_documents"],
) -> Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"]:
    grouped_prompts = generation.DatasetGenerator.get_prompts(documents)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="prompts", metadata=_get_metadata(grouped_prompts))

    return grouped_prompts


def _get_metadata(grouped_prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]]) -> dict:
    prompt_categories = list(grouped_prompts.keys())
    prompt_num_samples = {category: len(prompts) for category, prompts in grouped_prompts.items()}

    return {"data_categories": prompt_categories, "data_categories_num_prompts": prompt_num_samples}
