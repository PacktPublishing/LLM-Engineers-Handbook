from zenml import pipeline

from .digital_data_etl import digital_data_etl
from .feature_engineering import feature_engineering
from .generate_datasets import generate_datasets


@pipeline
def end_to_end_data(
    author_links: list[dict[str, str | list[str]]],
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
) -> None:
    wait_for_ids = []
    for author_data in author_links:
        last_step_invocation_id = digital_data_etl(
            user_full_name=author_data["user_full_name"], links=author_data["links"]
        )

        wait_for_ids.append(last_step_invocation_id)

    author_full_names = [author_data["user_full_name"] for author_data in author_links]
    wait_for_ids = feature_engineering(author_full_names=author_full_names, wait_for=wait_for_ids)

    generate_datasets(
        test_split_size=test_split_size,
        push_to_huggingface=push_to_huggingface,
        dataset_id=dataset_id,
        mock=mock,
        wait_for=wait_for_ids,
    )
