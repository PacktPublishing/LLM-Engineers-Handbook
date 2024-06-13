from abc import ABC, abstractmethod

from llm_engineering.domain.base import DataModel
from llm_engineering.domain.chunks import (
    ArticleChunkModel,
    PostChunkModel,
    RepositoryChunkModel,
)
from llm_engineering.domain.embedded_chunks import (
    ArticleEmbeddedChunkModel,
    PostEmbeddedChunkModel,
    RepositoryEmbeddedChunkModel,
)

from .operations import embedd_text


class EmbeddingDataHandler(ABC):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    @abstractmethod
    def embedd(self, data_model: DataModel) -> DataModel:
        pass


class PostEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: PostChunkModel) -> PostEmbeddedChunkModel:
        return PostEmbeddedChunkModel(
            id=data_model.id,
            content=data_model.content,
            embedding=embedd_text(data_model.content).tolist(),
            platform=data_model.platform,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
        )


class ArticleEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: ArticleChunkModel) -> ArticleEmbeddedChunkModel:
        return ArticleEmbeddedChunkModel(
            id=data_model.id,
            content=data_model.content,
            embedding=embedd_text(data_model.content).tolist(),
            platform=data_model.platform,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
        )


class RepositoryEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: RepositoryChunkModel) -> RepositoryEmbeddedChunkModel:
        return RepositoryEmbeddedChunkModel(
            id=data_model.id,
            content=data_model.content,
            embedding=embedd_text(data_model.content).tolist(),
            name=data_model.name,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
        )
