import numpy as np

from llm_engineering.domain.types import DataCategory

from .base import VectorDBDataModel


class PostEmbeddedChunkModel(VectorDBDataModel):
    platform: str
    chunk_id: str
    chunk_content: str
    embedding: list
    author_id: str

    class Config:
        name = "embedded_posts"
        category = DataCategory.POSTS
        use_vector_index = True


class ArticleEmbeddedChunkModel(VectorDBDataModel):
    platform: str
    link: str
    chunk_id: str
    chunk_content: str
    embedding: list
    author_id: str

    class Config:
        name = "embedded_articles"
        category = DataCategory.ARTICLES
        use_vector_index = True


class RepositoryEmbeddedChunkModel(VectorDBDataModel):
    name: str
    link: str
    chunk_id: str
    chunk_content: str
    embedding: list
    owner_id: str

    class Config:
        name = "embedded_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = True
