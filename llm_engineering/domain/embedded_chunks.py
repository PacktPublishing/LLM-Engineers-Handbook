from pydantic import UUID4
from llm_engineering.domain.types import DataCategory

from .base import VectorDBDataModel


class PostEmbeddedChunkModel(VectorDBDataModel):
    content: str
    embedding: list
    platform: str
    document_id: UUID4
    author_id: str

    class Config:
        name = "embedded_posts"
        category = DataCategory.POSTS
        use_vector_index = True


class ArticleEmbeddedChunkModel(VectorDBDataModel):
    content: str
    embedding: list
    platform: str
    link: str
    document_id: UUID4
    author_id: str

    class Config:
        name = "embedded_articles"
        category = DataCategory.ARTICLES
        use_vector_index = True


class RepositoryEmbeddedChunkModel(VectorDBDataModel):
    content: str
    embedding: list
    name: str
    link: str
    document_id: UUID4
    author_id: str

    class Config:
        name = "embedded_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = True
