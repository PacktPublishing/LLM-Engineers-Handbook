from loguru import logger

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets
except ImportError:
    logger.warning("Huggingface datasets not installed. Install with `pip install datasets`")

from pydantic import BaseModel

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class InstructDatasetSample(VectorBaseDocument):
    instruction: str
    answer: str

    class Config:
        category = DataCategory.INSTRUCT_DATASET_SAMPLES


class InstructDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[InstructDatasetSample]

    class Config:
        category = DataCategory.INSTRUCT_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]

        return Dataset.from_dict(
            {"instruction": [d["instruction"] for d in data], "answer": [d["answer"] for d in data]}
        )


class TrainTestSplit(BaseModel):
    train: dict[DataCategory, InstructDataset]
    test: dict[DataCategory, InstructDataset]
    test_split_size: float

    def to_huggingface(self, flatten: bool = False) -> "DatasetDict":
        train_datasets = {category.value: dataset.to_huggingface() for category, dataset in self.train.items()}
        test_datasets = {category.value: dataset.to_huggingface() for category, dataset in self.test.items()}

        if flatten:
            train_datasets = concatenate_datasets(list(train_datasets.values()))
            test_datasets = concatenate_datasets(list(test_datasets.values()))
        else:
            train_datasets = Dataset.from_dict(train_datasets)
            test_datasets = Dataset.from_dict(test_datasets)

        return DatasetDict({"train": train_datasets, "test": test_datasets})
