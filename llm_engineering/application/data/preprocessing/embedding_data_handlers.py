from abc import ABC, abstractmethod

from llm_engineering.domain.base import DataModel
from llm_engineering.domain.chunk import ArticleChunkModel, PostChunkModel, RepositoryChunkModel
from llm_engineering.domain.embedded_chunk import (
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
            entry_id=data_model.entry_id,
            platform=data_model.platform,
            chunk_id=data_model.chunk_id,
            chunk_content=data_model.chunk_content,
            embedded_content=embedd_text(data_model.chunk_content),
            author_id=data_model.author_id,
            type=data_model.type,
        )


class ArticleEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: ArticleChunkModel) -> ArticleEmbeddedChunkModel:
        return ArticleEmbeddedChunkModel(
            entry_id=data_model.entry_id,
            platform=data_model.platform,
            link=data_model.link,
            chunk_content=data_model.chunk_content,
            chunk_id=data_model.chunk_id,
            embedded_content=embedd_text(data_model.chunk_content),
            author_id=data_model.author_id,
            type=data_model.type,
        )


class RepositoryEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: RepositoryChunkModel) -> RepositoryEmbeddedChunkModel:
        return RepositoryEmbeddedChunkModel(
            entry_id=data_model.entry_id,
            name=data_model.name,
            link=data_model.link,
            chunk_id=data_model.chunk_id,
            chunk_content=data_model.chunk_content,
            embedded_content=embedd_text(data_model.chunk_content),
            owner_id=data_model.owner_id,
            type=data_model.type,
        )
