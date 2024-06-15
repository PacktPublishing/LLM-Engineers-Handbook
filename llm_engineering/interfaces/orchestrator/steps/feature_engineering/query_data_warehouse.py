from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.application import utils
from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.documents import (
    ArticleDocument,
    PostDocument,
    RepositoryDocument,
    UserDocument,
)


@step
def query_data_warehouse(
    user_full_name: str,
) -> Annotated[list, "raw_documents"]:
    logger.info(f"Querying data warehouse for user: {user_full_name}")

    first_name, last_name = utils.split_user_full_name(user_full_name)
    logger.info(f"First name: {first_name}, Last name: {last_name}")
    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
    
    results = fetch_all_data(user)
    
    user_documents = [doc for query_result in results.values() for doc in query_result]

    return user_documents


def fetch_all_data(user: UserDocument)-> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles, user_id): 'articles',
            executor.submit(__fetch_posts, user_id): 'posts',
            executor.submit(__fetch_repositories, user_id): 'repositories'
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

def __fetch_articles(user_id) -> list[NoSQLBaseDocument]:
    return ArticleDocument.bulk_find(author_id=user_id)

def __fetch_posts(user_id) -> list[NoSQLBaseDocument]:
    return PostDocument.bulk_find(author_id=user_id)

def __fetch_repositories(user_id) -> list[NoSQLBaseDocument]:
    return RepositoryDocument.bulk_find(author_id=user_id)
