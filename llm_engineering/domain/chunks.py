from typing import Optional

from pydantic import UUID4

from llm_engineering.domain.base import DataModel
from llm_engineering.domain.types import DataCategory


class PostChunkModel(DataModel):
    content: str
    platform: str
    document_id: UUID4
    author_id: str
    image: Optional[str] = None

    class Config:
        category = DataCategory.POSTS


class ArticleChunkModel(DataModel):
    content: str
    platform: str
    link: str
    document_id: UUID4
    author_id: str

    class Config:
        category = DataCategory.ARTICLES


class RepositoryChunkModel(DataModel):
    content: str
    name: str
    link: str
    document_id: UUID4
    author_id: str

    class Config:
        category = DataCategory.REPOSITORIES
