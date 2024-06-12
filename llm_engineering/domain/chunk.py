from typing import Optional

from llm_engineering.domain.types import DataType

from .base import DataModel


class PostChunkModel(DataModel):
    platform: str
    chunk_id: str
    chunk_content: str
    author_id: str
    image: Optional[str] = None
    
    @property
    def type(self) -> DataType:
        return DataType.POSTS


class ArticleChunkModel(DataModel):
    platform: str
    link: str
    chunk_id: str
    chunk_content: str
    author_id: str

    @property
    def type(self) -> DataType:
        return DataType.ARTICLES

class RepositoryChunkModel(DataModel):
    name: str
    link: str
    chunk_id: str
    chunk_content: str
    owner_id: str

    @property
    def type(self) -> DataType:
        return DataType.REPOSITORIES
