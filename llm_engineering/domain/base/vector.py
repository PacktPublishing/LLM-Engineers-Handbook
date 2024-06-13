from abc import ABC

import numpy as np
from pydantic import UUID4, BaseModel
from qdrant_client.models import PointStruct

from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.domain.types import DataCategory
from llm_engineering.infrastructure.db.qdrant import connection


class VectorBaseDocument(BaseModel, ABC):
    id: UUID4

    def to_point(self, **kwargs) -> PointStruct:
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        payload = self.dict(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        _id = str(payload.pop("id"))
        vector = payload.pop("embedding", {})
        if vector and isinstance(vector, np.ndarray):
            vector = vector.tolist()

        return PointStruct(id=_id, vector=vector, payload=payload)

    @classmethod
    def bulk_insert(cls, documents: list["VectorBaseDocument"]):
        # TODO: Move this to a separate step
        connection.get_or_create_collection(
            collection_name=cls.get_collection_name(),
            use_vector_index=cls.get_use_vector_index(),
        )

        points = [doc.to_point() for doc in documents]

        return connection.upsert(
            collection_name=cls.get_collection_name(), points=points
        )

    @classmethod
    def get_category(cls) -> DataCategory:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "category"):
            raise ImproperlyConfigured(
                "The class should define a Config class with"
                "the 'category' property that reflects the collection's data category."
            )

        return cls.Config.category

    @classmethod
    def get_collection_name(cls) -> str:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "name"):
            raise ImproperlyConfigured(
                "The class should define a Config class with"
                "the 'name' property that reflects the collection's name."
            )

        return cls.Config.name

    @classmethod
    def get_use_vector_index(cls) -> bool:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "use_vector_index"):
            return True

        return cls.Config.use_vector_index

    @classmethod
    def group_by_collection(
        cls, documents: list["VectorBaseDocument"]
    ) -> dict["VectorBaseDocument", list["VectorBaseDocument"]]:
        grouped = {}
        for doc in documents:
            collection_name = doc.get_collection_name()
            data_model_class = cls.collection_name_to_class(collection_name)

            if data_model_class not in grouped:
                grouped[data_model_class] = []
            grouped[data_model_class].append(doc)

        return grouped

    @classmethod
    def collection_name_to_class(
        cls, collection_name: str
    ) -> type["VectorBaseDocument"]:
        for subclass in cls.__subclasses__():
            try:
                if subclass.get_collection_name() == collection_name:
                    return subclass
            except ImproperlyConfigured:
                pass

            try:
                return subclass.collection_name_to_class(collection_name)
            except ValueError:
                continue

        raise ValueError(f"No subclass found for collection name: {collection_name}")
