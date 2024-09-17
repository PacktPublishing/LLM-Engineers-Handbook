from abc import ABC, abstractmethod
from typing import Any

from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from llm_engineering.domain.queries import Query


class PromptTemplateFactory(ABC, BaseModel):
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass


class RAGStep(ABC):
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock

    @abstractmethod
    def generate(self, query: Query, *args, **kwargs) -> Any:
        pass
