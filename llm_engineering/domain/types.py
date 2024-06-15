from enum import StrEnum


class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"
    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"

    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
