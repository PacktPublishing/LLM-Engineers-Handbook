from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.application import utils
from llm_engineering.domain.documents import (
    ArticleDocument,
    BaseDocument,
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

    # TODO: Run these queries in parallel using a ThreadPoolExecutor
    articles = ArticleDocument.bulk_find(author_id=str(user.id))
    posts = PostDocument.bulk_find(author_id=str(user.id))
    repositories = RepositoryDocument.bulk_find(owner_id=str(user.id))

    user_documents = [*articles, *posts, *repositories]

    return user_documents
