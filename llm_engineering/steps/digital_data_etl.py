from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.data.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.data.db.documents import UserDocument
from llm_engineering.exceptions import ImproperlyConfigured


def user_id_to_names(user: str | None) -> tuple[str, str]:
    if user is None:
        raise ImproperlyConfigured("User name is empty")

    name_tokens = user.split(" ")
    if len(name_tokens) == 0:
        raise ImproperlyConfigured("User name is empty")
    elif len(name_tokens) == 1:
        first_name, last_name = name_tokens[0], name_tokens[0]
    else:
        first_name, last_name = " ".join(name_tokens[:-1]), name_tokens[-1]

    return first_name, last_name


@step
def get_or_create_user(user_id: str) -> Annotated[UserDocument, "user"]:
    logger.info(f"Getting or creating user with id: {user_id}")
    
    first_name, last_name = user_id_to_names(user_id)

    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    return user


@step
def crawl_links(user: UserDocument, links: list[str]):
    dispatcher = (
        CrawlerDispatcher.build()
        .register_linkedin()
        .register_medium()
        .register_github()
    )
    
    logger.info(f"Starting to crawl {len(links)} links.")

    successfull_crawls = 0
    for link in links:
        successfull_crawls += _crawl_link(dispatcher, link, user)
        
    logger.info(f"Successfully crawled {successfull_crawls} / {len(links)} links.")


def _crawl_link(dispatcher: CrawlerDispatcher, link: str, user: UserDocument) -> bool:
    crawler = dispatcher.get_crawler(link)

    try:
        crawler.extract(link=link, user=user)
        
        return True
    except Exception as e:
        logger.error(f"An error occurred while crowling: {str(e)}")
        
        return False
