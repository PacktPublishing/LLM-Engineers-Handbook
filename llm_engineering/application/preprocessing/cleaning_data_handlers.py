from abc import ABC, abstractmethod

from llm_engineering.domain.base import DataModel
from llm_engineering.domain.clean import (
    ArticleCleanedModel,
    PostCleanedModel,
    RepositoryCleanedModel,
)
from llm_engineering.domain.documents import (
    ArticleDocument,
    BaseDocument,
    PostDocument,
    RepositoryDocument,
)

from .operations import clean_text


class CleaningDataHandler(ABC):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: BaseDocument) -> DataModel:
        pass


class PostCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PostDocument) -> PostCleanedModel:
        return PostCleanedModel(
            id=data_model.id,
            platform=data_model.platform,
            cleaned_content=clean_text("".join(data_model.content.values())),
            author_id=data_model.author_id,
            image=data_model.image if data_model.image else None,
        )


class ArticleCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ArticleDocument) -> ArticleCleanedModel:
        return ArticleCleanedModel(
            id=data_model.id,
            platform=data_model.platform,
            link=data_model.link,
            cleaned_content=clean_text("".join(data_model.content.values())),
            author_id=data_model.author_id,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> RepositoryCleanedModel:
        return RepositoryCleanedModel(
            id=data_model.id,
            name=data_model.name,
            link=data_model.link,
            cleaned_content=clean_text("".join(data_model.content.values())),
            owner_id=data_model.owner_id,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> RepositoryCleanedModel:
        return RepositoryCleanedModel(
            id=data_model.id,
            name=data_model.name,
            link=data_model.link,
            cleaned_content=clean_text("".join(data_model.content.values())),
            owner_id=data_model.owner_id,
        )
