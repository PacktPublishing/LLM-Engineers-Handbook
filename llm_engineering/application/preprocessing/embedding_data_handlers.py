from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.chunks import (ArticleChunk, Chunk, PostChunk,
                                           RepositoryChunk)
from llm_engineering.domain.embedded_chunks import (
    ArticleEmbeddedChunkModel, EmbeddedChunk, PostEmbeddedChunkModel,
    RepositoryEmbeddedChunkModel)

from .operations import embedd_text

ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    @abstractmethod
    def embedd(self, data_model: ChunkT) -> EmbeddedChunkT:
        pass


class PostEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: PostChunk) -> PostEmbeddedChunkModel:
        return PostEmbeddedChunkModel(
            id=data_model.id,
            content=data_model.content,
            embedding=embedd_text(data_model.content).tolist(),
            platform=data_model.platform,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
        )


class ArticleEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: ArticleChunk) -> ArticleEmbeddedChunkModel:
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
    def embedd(self, data_model: RepositoryChunk) -> RepositoryEmbeddedChunkModel:
        return RepositoryEmbeddedChunkModel(
            id=data_model.id,
            content=data_model.content,
            embedding=embedd_text(data_model.content).tolist(),
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
        )
