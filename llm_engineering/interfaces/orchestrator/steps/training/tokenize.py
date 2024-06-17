from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.domain.dataset import InstructDataset
from llm_engineering.domain.types import DataCategory


@step
def tokenize(
    instruct_datasets: Annotated[dict[DataCategory, InstructDataset], "instruct_datasets"],
) -> dict:
    tokenized_datasets = {}
    for category, instruct_dataset in instruct_datasets.items():  # noqa: B007
        logger.info(f"Tokenizing instruct dataset for category: {category}")

        # Do your own thing here

    return tokenized_datasets
