from .chunking import chunk_text
from .cleaning import clean_text
from .embeddings import embedd_repositories, embedd_text

__all__ = [
    "chunk_text",
    "clean_text",
    "embedd_text",
    "embedd_repositories",
]


def flatten(nested_list: list) -> list:
    """Flatten a list of lists into a single list."""

    return [item for sublist in nested_list for item in sublist]
