from typing import Tuple

import numpy as np

from llm_engineering.domain.types import DataType

from .base import VectorDBDataModel


class PostEmbeddedChunkModel(VectorDBDataModel):
    platform: str
    chunk_id: str
    chunk_content: str
    embedded_content: np.ndarray
    author_id: str

    class Config:
        arbitrary_types_allowed = True
        
    @property
    def type(self) -> DataType:
        return DataType.POSTS

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.id,
            "platform": self.platform,
            "content": self.chunk_content,
            "owner_id": self.author_id,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data


class ArticleEmbeddedChunkModel(VectorDBDataModel):
    platform: str
    link: str
    chunk_id: str
    chunk_content: str
    embedded_content: np.ndarray
    author_id: str

    class Config:
        arbitrary_types_allowed = True
        
    @property
    def type(self) -> DataType:
        return DataType.ARTICLES

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.id,
            "platform": self.platform,
            "content": self.chunk_content,
            "link": self.link,
            "author_id": self.author_id,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data


class RepositoryEmbeddedChunkModel(VectorDBDataModel):
    name: str
    link: str
    chunk_id: str
    chunk_content: str
    embedded_content: np.ndarray
    owner_id: str

    class Config:
        arbitrary_types_allowed = True
        
    @property
    def type(self) -> DataType:
        return DataType.REPOSITORIES

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.id,
            "name": self.name,
            "content": self.chunk_content,
            "link": self.link,
            "owner_id": self.owner_id,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data
