import uuid
from typing import List, Optional

from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import errors

from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.domain.types import DataCategory
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings

_database = connection.get_database(settings.DATABASE_NAME)


# TODO: Move this to base?
class BaseDocument(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    @classmethod
    def from_mongo(cls, data: dict) -> "BaseDocument":
        """Convert "_id" (str object) into "id" (UUID object)."""

        if not data:
            raise ValueError("Data is empty.")

        id = data.pop("_id", None)
        return cls(**dict(data, id=id))

    def to_mongo(self, **kwargs) -> dict:
        """Convert "id" (UUID object) into "_id" (str object)."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.dict(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        return parsed

    def save(self, **kwargs):
        collection = _database[self.get_collection_name()]
        try:
            result = collection.insert_one(self.to_mongo(**kwargs))
            return result.inserted_id
        except errors.WriteError as e:
            logger.error(f"Failed to insert document {e}")
            return None

    # TODO: Add generics to this method & return type
    @classmethod
    def get_or_create(cls, **filter_options) -> "BaseDocument":
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)

            new_instance = cls(**filter_options)
            new_instance = new_instance.save()

            return cls.from_mongo(new_instance)
        except errors.OperationFailure:
            logger.exception(
                f"Failed to retrieve document with filter options: {filter_options}"
            )

            raise

    @classmethod
    def bulk_insert(cls, documents: List, **kwargs) -> Optional[List[str]]:
        collection = _database[cls.get_collection_name()]
        try:
            result = collection.insert_many(
                [doc.to_mongo(**kwargs) for doc in documents]
            )
            return result.inserted_ids
        except errors.WriteError as e:
            logger.error(f"Failed to insert document {e}")
            return None

    @classmethod
    def bulk_find(cls, **filter_options) -> list["BaseDocument"]:
        collection = _database[cls.get_collection_name()]
        try:
            instances = collection.find(filter_options)
            return [
                document
                for instance in instances
                if (document := cls.from_mongo(instance)) is not None
            ]
        except errors.OperationFailure as e:
            logger.error(f"Failed to retrieve document: {e}")

            return []

    @classmethod
    def get_collection_name(cls) -> str:
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define an Settings configuration class with the name of the collection."
            )

        return cls.Settings.name


class UserDocument(BaseDocument):
    first_name: str
    last_name: str

    class Settings:
        name = "users"


class RepositoryDocument(BaseDocument):
    name: str
    link: str
    content: dict
    author_id: str = Field(alias="author_id")

    class Settings:
        name = DataCategory.REPOSITORIES


class PostDocument(BaseDocument):
    platform: str
    content: dict
    author_id: str = Field(alias="author_id")
    image: Optional[str] = None

    class Settings:
        name = DataCategory.POSTS


class ArticleDocument(BaseDocument):
    platform: str
    link: str
    content: dict
    author_id: str = Field(alias="author_id")

    class Settings:
        name = DataCategory.ARTICLES
