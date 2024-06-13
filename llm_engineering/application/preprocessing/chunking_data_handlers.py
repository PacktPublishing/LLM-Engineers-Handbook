import hashlib
from abc import ABC, abstractmethod
from uuid import UUID

from llm_engineering.domain.base import DataModel
from llm_engineering.domain.chunks import (
    ArticleChunkModel,
    PostChunkModel,
    RepositoryChunkModel,
)
from llm_engineering.domain.cleaned_documents import (
    ArticleCleanedModel,
    PostCleanedModel,
    RepositoryCleanedModel,
)

from .operations import chunk_text


class ChunkingDataHandler(ABC):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @abstractmethod
    def chunk(self, data_model: DataModel) -> list[DataModel]:
        pass


class PostChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: PostCleanedModel) -> list[PostChunkModel]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content)

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = PostChunkModel(
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
    def chunk(self, data_model: ArticleCleanedModel) -> list[ArticleChunkModel]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content)

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ArticleChunkModel(
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
    def chunk(self, data_model: RepositoryCleanedModel) -> list[RepositoryChunkModel]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content)

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = RepositoryChunkModel(
                id=UUID(chunk_id, version=4),
                content=chunk,
                name=data_model.name,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
            )
            data_models_list.append(model)

        return data_models_list
