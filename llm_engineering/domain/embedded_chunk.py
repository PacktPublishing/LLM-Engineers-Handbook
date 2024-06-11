from typing import Tuple

import numpy as np

from .base import VectorDBDataModel


class PostEmbeddedChunkModel(VectorDBDataModel):
    entry_id: str
    platform: str
    chunk_id: str
    chunk_content: str
    embedded_content: np.ndarray
    author_id: str
    type: str

    class Config:
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.entry_id,
            "platform": self.platform,
            "content": self.chunk_content,
            "owner_id": self.author_id,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data


class ArticleEmbeddedChunkModel(VectorDBDataModel):
    entry_id: str
    platform: str
    link: str
    chunk_id: str
    chunk_content: str
    embedded_content: np.ndarray
    author_id: str
    type: str

    class Config:
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.entry_id,
            "platform": self.platform,
            "content": self.chunk_content,
            "link": self.link,
            "author_id": self.author_id,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data


class RepositoryEmbeddedChunkModel(VectorDBDataModel):
    entry_id: str
    name: str
    link: str
    chunk_id: str
    chunk_content: str
    embedded_content: np.ndarray
    owner_id: str
    type: str

    class Config:
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.entry_id,
            "name": self.name,
            "content": self.chunk_content,
            "link": self.link,
            "owner_id": self.owner_id,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data
