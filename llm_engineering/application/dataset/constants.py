from llm_engineering.domain.dataset import DatasetType

MOCKED_RESPONSE_INSTRUCT = """
[
    {"instruction": "<mocked generated instruction> 1", "answer": "<mocked generated answer> 1"},
    {"instruction": "<mocked generated instruction> 2", "answer": "<mocked generated answer> 2"},
    {"instruction": "<mocked generated instruction> 3", "answer": "<mocked generated answer> 3"}
]
"""

MOCKED_RESPONSE_PREFERENCE = """
[
    {"instruction": "<mocked generated instruction> 1", "rejected": "<mocked generated answer> 1", "chosen": "Mocked extracted extracted extracted extracted extracted extracted extracted extracted extracted extracted answer 1."},
    {"instruction": "<mocked generated instruction> 2", "rejected": "<mocked generated answer> 2", "chosen": "Mocked extracted extracted extracted extracted extracted extracted extracted extracted extracted extracted answer 2."},
    {"instruction": "<mocked generated instruction> 3", "rejected": "<mocked generated answer> 3", "chosen": "Mocked extracted answer 3"}
]
"""


def get_mocked_response(dataset_type: DatasetType) -> str:
    if dataset_type == DatasetType.INSTRUCTION:
        return MOCKED_RESPONSE_INSTRUCT
    elif dataset_type == DatasetType.PREFERENCE:
        return MOCKED_RESPONSE_PREFERENCE
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
