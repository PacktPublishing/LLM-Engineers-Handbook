from loguru import logger
from zenml import step

from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.domain.documents import UserDocument


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
