import re

from .base import BaseCrawler
from .github import GithubCrawler
from .linkedin import LinkedInCrawler
from .medium import MediumCrawler


class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {}

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        dispatcher = cls()
        
        return dispatcher

    def register_medium(self) -> "CrawlerDispatcher":
        self.register("medium", MediumCrawler)
        
        return self

    def register_linkedin(self) -> "CrawlerDispatcher":
        self.register("linkedin", LinkedInCrawler)
        
        return self

    def register_github(self) -> "CrawlerDispatcher":
        self.register("github", GithubCrawler)
        
        return self

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        self._crawlers[r"https://(www\.)?{}.com/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            raise ValueError(f"No crawler found for the provided link: {url}")
