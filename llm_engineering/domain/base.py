from abc import ABC, abstractmethod

from pydantic import UUID4, BaseModel

from llm_engineering.domain.types import DataType


class DataModel(BaseModel, ABC):
    """
    Abstract class for all data model
    """

    id: UUID4
    
    @property
    @abstractmethod
    def type(self) -> DataType:
        pass


class VectorDBDataModel(DataModel, ABC):
    """
    Abstract class for all data models that need to be saved into a vector DB (e.g. Qdrant)
    """
    
    @abstractmethod
    def to_payload(self) -> tuple:
        pass
