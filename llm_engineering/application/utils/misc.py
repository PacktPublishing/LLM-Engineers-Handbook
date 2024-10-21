from typing import Generator

from transformers import AutoTokenizer

from llm_engineering.settings import settings


def flatten(nested_list: list) -> list:
    """Flatten a list of lists into a single list."""

    return [item for sublist in nested_list for item in sublist]


def batch(list_: list, size: int) -> Generator[list, None, None]:
    yield from (list_[i : i + size] for i in range(0, len(list_), size))


def compute_num_tokens(text: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(settings.HF_MODEL_ID)

    return len(tokenizer.encode(text, add_special_tokens=False))
