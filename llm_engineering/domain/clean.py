from typing import Optional, Tuple

from pydantic import UUID4

from .base import VectorDBDataModel
from .types import DataType


class PostCleanedModel(VectorDBDataModel):
    platform: str
    cleaned_content: str
    author_id: str
    image: Optional[str] = None
    
    @property
    def type(self) -> DataType:
        return DataType.POSTS

    def to_payload(self) -> Tuple[UUID4, dict]:
        data = {
            "platform": self.platform,
            "author_id": self.author_id,
            "cleaned_content": self.cleaned_content,
            "image": self.image,
        }

        return self.id, data


class ArticleCleanedModel(VectorDBDataModel):
    platform: str
    link: str
    cleaned_content: str
    author_id: str
    
    @property
    def type(self) -> DataType:
        return DataType.ARTICLES

    def to_payload(self) -> Tuple[UUID4, dict]:
        data = {
            "platform": self.platform,
            "link": self.link,
            "cleaned_content": self.cleaned_content,
            "author_id": self.author_id,
        }

        return self.id, data


class RepositoryCleanedModel(VectorDBDataModel):
    name: str
    link: str
    cleaned_content: str
    owner_id: str
    
    @property
    def type(self) -> DataType:
        return DataType.REPOSITORIES

    def to_payload(self) -> Tuple[UUID4, dict]:
        data = {
            "name": self.name,
            "link": self.link,
            "cleaned_content": self.cleaned_content,
            "owner_id": self.owner_id,
        }

        return self.id, data
