from typing import Optional

from .base import DataModel


class PostChunkModel(DataModel):
    entry_id: str
    platform: str
    chunk_id: str
    chunk_content: str
    author_id: str
    image: Optional[str] = None
    type: str


class ArticleChunkModel(DataModel):
    entry_id: str
    platform: str
    link: str
    chunk_id: str
    chunk_content: str
    author_id: str
    type: str


class RepositoryChunkModel(DataModel):
    entry_id: str
    name: str
    link: str
    chunk_id: str
    chunk_content: str
    owner_id: str
    type: str
