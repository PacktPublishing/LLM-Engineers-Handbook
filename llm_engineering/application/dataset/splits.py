from sklearn.model_selection import train_test_split

from llm_engineering.domain.dataset import InstructDataset, InstructDatasetSample, TrainTestSplit
from llm_engineering.domain.types import DataCategory


def create_train_test_split(
    data: dict[DataCategory, InstructDataset], test_size=0.2, random_state=42
) -> TrainTestSplit:
    train_data = {}
    test_data = {}

    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        train_samples_dicts, test_samples_dicts = train_test_split(
            samples_dicts, test_size=test_size, random_state=random_state
        )

        train_samples = [InstructDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
        test_samples = [InstructDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]

        train_dataset = InstructDataset(category=category, samples=train_samples)
        test_dataset = InstructDataset(category=category, samples=test_samples)

        train_data[category] = train_dataset
        test_data[category] = test_dataset

    return TrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)
