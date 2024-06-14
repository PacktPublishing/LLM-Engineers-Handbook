from abc import ABC

from pydantic import UUID4

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class Query(VectorBaseDocument, ABC):
    content: str
    author_id: UUID4 | None = None

    class Config:
        category = DataCategory.QUERIES
        
    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))
    
    def replace_content(self, new_content: str) -> "Query":
        return Query(id=self.id, content=new_content, author_id=self.author_id)


class EmbeddedQuery(Query):
    embedding: list[float]

    class Config:
        category = DataCategory.QUERIES
