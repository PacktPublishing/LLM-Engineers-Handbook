from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.data.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.data.db.documents import UserDocument
from llm_engineering.exceptions import ImproperlyConfigured


def user_to_names(user: str | None) -> tuple[str, str]:
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
def get_or_create_user() -> Annotated[UserDocument, "user"]:
    first_name, last_name = user_to_names("Paul Iusztin")

    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    return user


@step
def crawl_links(user: UserDocument) -> Annotated[dict, "result"]:
    dispatcher = (
        CrawlerDispatcher.build()
        .register_linkedin()
        .register_medium()
        .register_github()
    )

    # TODO: Fix LI crawler
    # link = "https://www.linkedin.com/in/vesaalexandru/"
    link = "https://medium.com/decodingml/architect-scalable-and-cost-effective-llm-rag-inference-pipelines-73b94ef82a99"
    crawler = dispatcher.get_crawler(link)

    try:
        crawler.extract(link=link, user=user)

        logger.info(f"Link processed successfully: {link}")

        return {"statusCode": 200, "body": "Link processed successfully"}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

        return {"statusCode": 500, "body": f"An error occurred: {str(e)}"}
