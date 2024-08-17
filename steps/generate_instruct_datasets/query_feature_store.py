from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from qdrant_client.http import exceptions
from typing_extensions import Annotated
from zenml import step

from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)


@step
def query_feature_store() -> Annotated[list, "queried_cleaned_documents"]:
    logger.info("Querying feature store.")

    results = fetch_all_data()

    cleaned_documents = [doc for query_result in results.values() for doc in query_result]

    return cleaned_documents


def fetch_all_data() -> dict[str, list[NoSQLBaseDocument]]:
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(
                __fetch_articles,
            ): "articles",
            executor.submit(
                __fetch_posts,
            ): "posts",
            executor.submit(
                __fetch_repositories,
            ): "repositories",
        }

        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")

                results[query_name] = []

    return results


def __fetch_articles() -> list[CleanedDocument]:
    return __fetch(CleanedArticleDocument)


def __fetch_posts() -> list[CleanedDocument]:
    return __fetch(CleanedPostDocument)


def __fetch_repositories() -> list[CleanedDocument]:
    return __fetch(CleanedRepositoryDocument)


def __fetch(cleaned_document_type: type[CleanedDocument], limit: int = 1) -> list[CleanedDocument]:
    try:
        cleaned_documents, next_offset = cleaned_document_type.bulk_find(limit=limit)
    except exceptions.UnexpectedResponse:
        return []

    while next_offset:
        documents, next_offset = cleaned_document_type.bulk_find(limit=limit, offset=next_offset)
        cleaned_documents.extend(documents)

    return cleaned_documents
