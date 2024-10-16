from sklearn.model_selection import train_test_split

from llm_engineering.application.preprocessing.operations.chunking import chunk_document
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.dataset import (
    InstructDataset,
    InstructDatasetSample,
    InstructTrainTestSplit,
    PreferenceDataset,
    PreferenceDatasetSample,
    PreferenceTrainTestSplit,
)
from llm_engineering.domain.types import DataCategory


def create_instruct_train_test_split(
    data: dict[DataCategory, InstructDataset], test_size=0.2, random_state=42
) -> InstructTrainTestSplit:
    train_data = {}
    test_data = {}

    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )
            train_samples = [InstructDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [InstructDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]
        else:
            train_samples = []
            test_samples = []

        train_dataset = InstructDataset(category=category, samples=train_samples)
        test_dataset = InstructDataset(category=category, samples=test_samples)

        train_data[category] = train_dataset
        test_data[category] = test_dataset

    return InstructTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)


def create_preference_train_test_split(
    data: dict[DataCategory, PreferenceDataset], test_size=0.2, random_state=42
) -> PreferenceTrainTestSplit:
    train_data = {}
    test_data = {}

    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )
            train_samples = [PreferenceDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [PreferenceDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]
        else:
            train_samples = []
            test_samples = []

        train_dataset = PreferenceDataset(category=category, samples=train_samples)
        test_dataset = PreferenceDataset(category=category, samples=test_samples)

        train_data[category] = train_dataset
        test_data[category] = test_dataset

    return PreferenceTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)


def filter_short_answers(
    data: dict[DataCategory, PreferenceDataset], min_length: int = 100
) -> dict[DataCategory, PreferenceDataset]:
    def is_long_enough(example: PreferenceDatasetSample) -> bool:
        return len(example.chosen) >= min_length

    filtered_data = {}
    for category, dataset in data.items():
        filetered_dataset_samples = list(filter(is_long_enough, dataset.samples))
        filtered_dataset = PreferenceDataset(category=category, samples=filetered_dataset_samples)

        filtered_data[category] = filtered_dataset

    return filtered_data


def filter_answer_format(data: dict[DataCategory, PreferenceDataset]) -> dict[DataCategory, PreferenceDataset]:
    def is_valid_format(example: PreferenceDatasetSample) -> bool:
        chosen = example.chosen

        return len(chosen) > 0 and chosen[0].isupper() and chosen[-1] in (".", "!", "?")

    filtered_data = {}
    for category, dataset in data.items():
        filetered_dataset_samples = list(filter(is_valid_format, dataset.samples))
        filtered_dataset = PreferenceDataset(category=category, samples=filetered_dataset_samples)

        filtered_data[category] = filtered_dataset

    return filtered_data


def extract_substrings(
    documents: list[CleanedDocument], min_length: int = 1000, max_length: int = 2000
) -> list[CleanedDocument]:
    extracts = []
    for document in documents:
        document_extracts = chunk_document(document.content, min_length, max_length)
        for extract in document_extracts:
            subdocument = document.model_copy()
            subdocument.content = extract

            extracts.append(subdocument)

    return extracts
