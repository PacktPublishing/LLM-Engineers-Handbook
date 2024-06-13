import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from llm_engineering.domain.chunks import (
    ArticleChunk,
    Chunk,
    PostChunk,
    RepositoryChunk,
)
from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)

from .operations import chunk_text

CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)


class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass


class PostChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: CleanedPostDocument) -> list[PostChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content)

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = PostChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                document_id=data_model.id,
                author_id=data_model.author_id,
                image=data_model.image if data_model.image else None,
            )
            data_models_list.append(model)

        return data_models_list


class ArticleChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: CleanedArticleDocument) -> list[ArticleChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content)

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ArticleChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
            )
            data_models_list.append(model)

        return data_models_list


class RepositoryChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: CleanedRepositoryDocument) -> list[RepositoryChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content)

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = RepositoryChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                name=data_model.name,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
            )
            data_models_list.append(model)

        return data_models_list
