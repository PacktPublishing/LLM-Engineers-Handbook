from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class InstructDatasetSample(VectorBaseDocument):
    instruction: str
    response: str | None = None

    class Config:
        category = DataCategory.INSTRUCT_DATASET_SAMPLES


class InstructDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[InstructDatasetSample]

    class Config:
        category = DataCategory.INSTRUCT_DATASET_SAMPLES

    @property
    def num_samples(self) -> int:
        return len(self.samples)
