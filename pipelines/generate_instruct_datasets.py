from zenml import pipeline

from steps import generate_instruct_datasets as cd_steps


@pipeline
def generate_instruct_datasets(
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
    wait_for: str | list[str] | None = None,
) -> None:
    cleaned_documents = cd_steps.query_feature_store(after=wait_for)
    prompts = cd_steps.create_prompts(documents=cleaned_documents)
    instruct_dataset = cd_steps.generate(prompts=prompts, test_split_size=test_split_size, mock=mock)
    if push_to_huggingface:
        cd_steps.push_to_huggingface(dataset=instruct_dataset, dataset_id=dataset_id)
