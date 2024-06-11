from typing import Optional, Tuple

from .base import VectorDBDataModel


class PostCleanedModel(VectorDBDataModel):
    entry_id: str
    platform: str
    cleaned_content: str
    author_id: str
    image: Optional[str] = None
    type: str

    def to_payload(self) -> Tuple[str, dict]:
        data = {
            "platform": self.platform,
            "author_id": self.author_id,
            "cleaned_content": self.cleaned_content,
            "image": self.image,
            "type": self.type,
        }

        return self.entry_id, data


class ArticleCleanedModel(VectorDBDataModel):
    entry_id: str
    platform: str
    link: str
    cleaned_content: str
    author_id: str
    type: str

    def to_payload(self) -> Tuple[str, dict]:
        data = {
            "platform": self.platform,
            "link": self.link,
            "cleaned_content": self.cleaned_content,
            "author_id": self.author_id,
            "type": self.type,
        }

        return self.entry_id, data


class RepositoryCleanedModel(VectorDBDataModel):
    entry_id: str
    name: str
    link: str
    cleaned_content: str
    owner_id: str
    type: str

    def to_payload(self) -> Tuple[str, dict]:
        data = {
            "name": self.name,
            "link": self.link,
            "cleaned_content": self.cleaned_content,
            "owner_id": self.owner_id,
            "type": self.type,
        }

        return self.entry_id, data
