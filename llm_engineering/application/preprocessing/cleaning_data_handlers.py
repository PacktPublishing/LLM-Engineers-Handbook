from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.cleaned_documents import (CleanedArticle,
                                                      CleanedDocument,
                                                      CleanedPost,
                                                      RepositoryCleanedModel)
from llm_engineering.domain.documents import (ArticleDocument, Document,
                                              PostDocument, RepositoryDocument)

from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass


class PostCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PostDocument) -> CleanedPost:
        return CleanedPost(
            id=data_model.id,
            content=clean_text("".join(data_model.content.values())),
            platform=data_model.platform,
            author_id=data_model.author_id,
            image=data_model.image if data_model.image else None,
        )


class ArticleCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ArticleDocument) -> CleanedArticle:
        return CleanedArticle(
            id=data_model.id,
            content=clean_text("".join(data_model.content.values())),
            platform=data_model.platform,
            link=data_model.link,
            author_id=data_model.author_id,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> RepositoryCleanedModel:
        return RepositoryCleanedModel(
            id=data_model.id,
            content=clean_text("".join(data_model.content.values())),
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            author_id=data_model.author_id,
        )
