from zenml import pipeline

from llm_engineering.domain.dataset import DatasetType
from steps import generate_datasets as cd_steps


@pipeline
def generate_datasets(
    dataset_type: DatasetType = DatasetType.INSTRUCTION,
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
    wait_for: str | list[str] | None = None,
) -> None:
    cleaned_documents = cd_steps.query_feature_store(after=wait_for)
    prompts = cd_steps.create_prompts(documents=cleaned_documents, dataset_type=dataset_type)
    if dataset_type == DatasetType.INSTRUCTION:
        dataset = cd_steps.generate_intruction_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock)
    elif dataset_type == DatasetType.PREFERENCE:
        dataset = cd_steps.generate_preference_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if push_to_huggingface:
        cd_steps.push_to_huggingface(dataset=dataset, dataset_id=dataset_id)
