from typing import Optional

from .base import DataModel


class RepositoryRawModel(DataModel):
    entry_id: str
    name: str
    link: str
    content: dict
    owner_id: str
    type: str


class ArticleRawModel(DataModel):
    entry_id: str
    platform: str
    link: str
    content: dict
    author_id: str
    type: str


class PostsRawModel(DataModel):
    entry_id: str
    platform: str
    content: dict
    author_id: str
    image: Optional[str] = None
    type: str
