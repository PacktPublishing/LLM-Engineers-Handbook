from abc import ABC

from pydantic import UUID4

from llm_engineering.domain.types import DataCategory

from .base import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: list
    platform: str
    document_id: UUID4
    author_id: str


class PostEmbeddedChunkModel(EmbeddedChunk):
    class Config:
        name = "embedded_posts"
        category = DataCategory.POSTS
        use_vector_index = True


class ArticleEmbeddedChunkModel(EmbeddedChunk):
    link: str

    class Config:
        name = "embedded_articles"
        category = DataCategory.ARTICLES
        use_vector_index = True


class RepositoryEmbeddedChunkModel(EmbeddedChunk):
    name: str
    link: str

    class Config:
        name = "embedded_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = True
