from abc import ABC, abstractmethod

from llm_engineering.domain.base import DataModel
from llm_engineering.domain.clean import ArticleCleanedModel, PostCleanedModel, RepositoryCleanedModel
from llm_engineering.domain.raw import ArticleRawModel, PostsRawModel, RepositoryRawModel
from .operations import clean_text


class CleaningDataHandler(ABC):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DataModel) -> DataModel:
        pass


class PostCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PostsRawModel) -> PostCleanedModel:
        return PostCleanedModel(
            entry_id=data_model.entry_id,
            platform=data_model.platform,
            cleaned_content=clean_text("".join(data_model.content.values())),
            author_id=data_model.author_id,
            image=data_model.image if data_model.image else None,
            type=data_model.type,
        )


class ArticleCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ArticleRawModel) -> ArticleCleanedModel:
        return ArticleCleanedModel(
            entry_id=data_model.entry_id,
            platform=data_model.platform,
            link=data_model.link,
            cleaned_content=clean_text("".join(data_model.content.values())),
            author_id=data_model.author_id,
            type=data_model.type,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryRawModel) -> RepositoryCleanedModel:
        return RepositoryCleanedModel(
            entry_id=data_model.entry_id,
            name=data_model.name,
            link=data_model.link,
            cleaned_content=clean_text("".join(data_model.content.values())),
            owner_id=data_model.owner_id,
            type=data_model.type,
        )
