from typing import Optional

from llm_engineering.domain.base import DataModel
from llm_engineering.domain.types import DataCategory


class PostChunkModel(DataModel):
    platform: str
    chunk_id: str
    chunk_content: str
    author_id: str
    image: Optional[str] = None

    class Config:
        category = DataCategory.POSTS


class ArticleChunkModel(DataModel):
    platform: str
    link: str
    chunk_id: str
    chunk_content: str
    author_id: str

    class Config:
        category = DataCategory.ARTICLES


class RepositoryChunkModel(DataModel):
    name: str
    link: str
    chunk_id: str
    chunk_content: str
    owner_id: str

    class Config:
        category = DataCategory.REPOSITORIES
