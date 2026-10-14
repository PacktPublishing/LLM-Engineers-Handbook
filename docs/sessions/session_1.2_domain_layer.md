# Session 1.2: Domain Layer - Data Modeling

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand the two base document hierarchies: `NoSQLBaseDocument` (ODM for MongoDB) and `VectorBaseDocument` (OVM for Qdrant)
- Model data with Pydantic v2 and `Generic[T]`
- Implement MongoDB CRUD through the **ODM** (object-document mapping) class methods
- Implement Qdrant vector operations through the **OVM** (object-vector mapping) class methods
- Know every concrete entity in `llm_engineering/domain/`
- Understand the `_id`/`id` and `payload`/`vector` translation seams, and their failure modes
- Reason about what belongs in Mongo versus Qdrant, and why the same data lives in both

---

## 🏗️ Architecture Overview

### The Domain Layer at a Glance

The domain layer is the innermost layer. It has no knowledge of pipelines, steps, or AWS. It only knows business entities and how to persist them. Everything else depends on it.

```
┌───────────────────────────────────────────────────────────────┐
│                      domain/ (innermost)                       │
│                                                                │
│                    VectorBaseDocument (Qdrant / OVM)          │
│                    NoSQLBaseDocument  (MongoDB / ODM)         │
│                            ▲                                   │
│             ┌──────────────┼───────────────┬───────────────┐   │
│             │              │               │               │   │
│        documents.py    chunks.py    embedded_chunks.py  dataset.py
│        (raw Mongo)     (Qdrant)     (Qdrant+vector)     (datasets)│
│             │              │               │               │   │
│   User/Post/Article   Post/Article     EmbeddedPost     Instruct  │
│   /Repository        /Repository       /Article/Repo    Preference│
└───────────────────────────────────────────────────────────────┘
```

There are also smaller, inference-side value objects: `queries.py` (`Query`, `EmbeddedQuery`) and `prompt.py` (`Prompt`, `GenerateDatasetSamplesPrompt`). They follow the OVM base but are not part of the storage lifecycle in the same way.

### The Two Dimensions of the Domain (per the book)

The book models the domain along two axes:

- **Data category**: post, article, repository
- **Data state**: cleaned, chunked, embedded

The design decision is to make the *base class per state* (`CleanedDocument`, `Chunk`, `EmbeddedChunk`) and the *subclass per category* (`...Post`, `...Article`, `...Repository`). The rationale: states are stable and few; categories are the axis likely to grow (add X, GitLab). To add a category you inherit the existing state bases; you do not restructure storage.

```
                 CleanedDocument   Chunk   EmbeddedChunk
Post             CleanedPost       PostChunk   EmbeddedPostChunk
Article          CleanedArticle    ArticleChunk EmbeddedArticleChunk
Repository       CleanedRepo       RepoChunk    EmbeddedRepoChunk
```

Nine concrete document-type combinations, all persisting to Qdrant, plus raw Mongo documents and dataset wrappers.

### Why Two Base Classes?

The system stores the same conceptual data in two engines with different constraints:

| Concern | MongoDB (`NoSQLBaseDocument`) | Qdrant (`VectorBaseDocument`) |
|---------|-------------------------------|-------------------------------|
| Identity | `_id` string ↔ `id` UUID | point `id` string ↔ `id` UUID |
| Payload | full document | `payload` + optional `embedding` vector |
| Query | `find_one` / `find` | `search` / `scroll` |
| Bulk | `insert_many` | `upsert` points |
| Config | inner `Settings` class | inner `Config` class |
| Fail-soft read | returns `None` / `[]` | returns `[]` / `( [], None )` |

Both expose class methods (`find`, `bulk_find`, `save`, `bulk_insert`) so callers write the same code against either store. This is the *generic repository* pattern without a repository class: persistence lives on the entity.

**Naming (book terminology)**: the book calls the MongoDB mapper the **ODM** (object-document mapping) and the Qdrant mapper the **OVM** (object-vector mapping), because it maps objects to embeddings and vectors instead of SQL/structured tables. Both descend from the same ORM concept: CRUD hidden behind an object.

### Where does the data actually live? (worked example)

For a single cleaned Medium article, the article ends up as three records across two systems:

| Stage | Python class | Store | Collection / name | Carries vector? |
|-------|--------------|-------|-------------------|-----------------|
| Crawled | `ArticleDocument` | MongoDB | `articles` | no |
| Cleaned | `CleanedArticleDocument` | Qdrant | `cleaned_articles` | no (`use_vector_index=False`) |
| Chunked | `ArticleChunk` | Qdrant | (category `articles`) | no |
| Embedded | `EmbeddedArticleChunk` | Qdrant | `embedded_articles` | yes (`use_vector_index=True`) |

The `id` is the same UUID threaded through each stage where the object is *the same entity transformed*; `document_id` is the link from a chunk back to its parent cleaned document.

---

## 📁 Key Files Explained

### 1. `llm_engineering/domain/types.py` - Data Categories

**Purpose**: A single `StrEnum` that tags every entity and collection in the system.

```python
# llm_engineering/domain/types.py  (verbatim)
from enum import StrEnum


class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"

    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
```

**Key Concepts**:
- **`StrEnum`** members behave as `str`, so they serialize cleanly to MongoDB and Qdrant payloads.
- The values double as **collection names** (`Settings.name`, `Config.name`) and as the grouping key in `group_by_category`.
- Notice the enum spans the whole lifecycle: raw data (`POSTS`, `ARTICLES`, `REPOSITORIES`), generated prompts (`PROMPT`, `QUERIES`), and datasets (`INSTRUCT_*`, `PREFERENCE_*`).
- Because `Config.category` is the enum and `Config.name` is a plain string like `"cleaned_articles"`, the two are related but not identical. Do not assume name == category.

---

### 2. `llm_engineering/domain/base/nosql.py` - MongoDB Base Document (ODM)

**Purpose**: A generic, Pydantic-based ODM base for MongoDB.

```python
# llm_engineering/domain/base/nosql.py  (verbatim, head)
import uuid
from abc import ABC
from typing import Generic, Type, TypeVar

from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import errors

from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings

_database = connection.get_database(settings.DATABASE_NAME)

T = TypeVar("T", bound="NoSQLBaseDocument")


class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)
```

**Identity and hashing** (verbatim):

```python
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)
```

Equality is by `id`, not by all fields. This makes de-duplication trivial:

```python
documents = list({doc.id: doc for doc in documents}.values())
```

It also means two documents with identical content but different ids are *not* equal, and two documents that differ in every field but share an id *are* equal. That is the intended "entity identity" semantics.

**The Mongo ⇄ Pydantic translation layer** (verbatim):

```python
    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        """Convert "_id" (str object) into "id" (UUID object)."""

        if not data:
            raise ValueError("Data is empty.")

        id = data.pop("_id")

        return cls(**dict(data, id=id))

    def to_mongo(self: T, **kwargs) -> dict:
        """Convert "id" (UUID object) into "_id" (str object)."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        return parsed
```

**Why this matters**: MongoDB's primary key is `_id` (a string/ObjectId), while the Python model uses `id: UUID4`. The two methods form a clean seam so no other layer worries about the difference.

There is also a `model_dump` override that stringifies UUIDs recursively *at the top level* for Mongo:

```python
    def model_dump(self: T, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)

        for key, value in dict_.items():
            if isinstance(value, uuid.UUID):
                dict_[key] = str(value)

        return dict_
```

> **Subtlety**: this Mongo `model_dump` stringifies only top-level UUIDs. The Qdrant OVM's version (below) recurses into lists and nested dicts. If you add nested UUID fields to a Mongo document, the top-level loop will miss them and `save()` may fail to serialize. Keep Mongo documents flat, or extend this method.

**Full CRUD surface** (verbatim):

```python
    def save(self: T, **kwargs) -> T | None:
        collection = _database[self.get_collection_name()]
        try:
            collection.insert_one(self.to_mongo(**kwargs))

            return self
        except errors.WriteError:
            logger.exception("Failed to insert document.")

            return None

    @classmethod
    def get_or_create(cls: Type[T], **filter_options) -> T:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)

            new_instance = cls(**filter_options)
            new_instance = new_instance.save()

            return new_instance
        except errors.OperationFailure:
            logger.exception(f"Failed to retrieve document with filter options: {filter_options}")

            raise

    @classmethod
    def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
        collection = _database[cls.get_collection_name()]
        try:
            collection.insert_many([doc.to_mongo(**kwargs) for doc in documents])

            return True
        except (errors.WriteError, errors.BulkWriteError):
            logger.error(f"Failed to insert documents of type {cls.__name__}")

            return False

    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)

            return None
        except errors.OperationFailure:
            logger.error("Failed to retrieve document")

            return None

    @classmethod
    def bulk_find(cls: Type[T], **filter_options) -> list[T]:
        collection = _database[cls.get_collection_name()]
        try:
            instances = collection.find(filter_options)
            return [document for instance in instances if (document := cls.from_mongo(instance)) is not None]
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents")

            return []
```

**Collection name resolution** (verbatim):

```python
    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define an Settings configuration class with the name of the collection."
            )

        return cls.Settings.name
```

**Key Concepts**:
- **Generic + ABC**: `Generic[T]` gives subclasses a precise return type; `ABC` prevents direct instantiation.
- **Class methods as an ODM**: There is no repository class. Persistence lives on the entity itself.
- **Fail soft**: Read helpers log and return `None`/`[]`; the caller decides what to do.
- **`get_collection_name` guard**: A subclass that forgets its `Settings.name` fails immediately with `ImproperlyConfigured`, not with a cryptic key error later.
- **`get_or_create` semantics**: it is a *read-then-write* with no unique index guarantee. Two concurrent callers can both miss and both insert. It is named per its Mongo filter, not a database upsert.

> **Naming quirk in the book's prose**: the book's SQLAlchemy example uses a `users` table. Here the concept is a *collection*. The method and field names (`_id`, `find_one`, `insert_many`) are PyMongo's, not SQL's.

---

### 3. `llm_engineering/domain/base/vector.py` - Qdrant Base Document (OVM)

**Purpose**: The vector-store counterpart, called the **OVM** (object-vector mapping) in the book. Same ODM-style pattern, different engine and naming. Full file is 269 lines; the load-bearing parts follow.

```python
# llm_engineering/domain/base/vector.py  (head, verbatim)
import uuid
from abc import ABC
from typing import Any, Callable, Dict, Generic, Type, TypeVar
from uuid import UUID

import numpy as np
from loguru import logger
from pydantic import UUID4, BaseModel, Field
from qdrant_client.http import exceptions
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import CollectionInfo, PointStruct, Record

from llm_engineering.application.networks.embeddings import EmbeddingModelSingleton
from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.domain.types import DataCategory
from llm_engineering.infrastructure.db.qdrant import connection

T = TypeVar("T", bound="VectorBaseDocument")


class VectorBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)
```

**Record ⇄ Point translation** (verbatim):

```python
    @classmethod
    def from_record(cls: Type[T], point: Record) -> T:
        _id = UUID(point.id, version=4)
        payload = point.payload or {}

        attributes = {
            "id": _id,
            **payload,
        }
        if cls._has_class_attribute("embedding"):
            attributes["embedding"] = point.vector or None

        return cls(**attributes)

    def to_point(self: T, **kwargs) -> PointStruct:
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        payload = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        _id = str(payload.pop("id"))
        vector = payload.pop("embedding", {})
        if vector and isinstance(vector, np.ndarray):
            vector = vector.tolist()

        return PointStruct(id=_id, vector=vector, payload=payload)
```

> **Book vs repo discrepancy (FACT)**: the printed book's `from_record` writes `payload["embedding"] = point.vector or None` and `to_point` uses `self.dict(...)`. In this repo the correct code writes to `attributes` and uses `self.model_dump(...)`. `self.dict()` is the deprecated Pydantic v1 API; `model_dump()` is Pydantic v2. The `attributes` version matters: writing into `payload` would leak `embedding` into the stored payload and, for classes without an `embedding` field, would crash construction. Trust the repo.

**Why `_has_class_attribute("embedding")`**: Not every vector document carries an embedding. `CleanedDocument` (a vector collection used as a staging store) has no `embedding` field, so `from_record` must not inject one. The helper walks the MRO:

```python
    @classmethod
    def _has_class_attribute(cls: Type[T], attribute_name: str) -> bool:
        if attribute_name in cls.__annotations__:
            return True

        for base in cls.__bases__:
            if hasattr(base, "_has_class_attribute") and base._has_class_attribute(attribute_name):
                return True

        return False
```

**UUID stringification (recursive)**:

```python
    def model_dump(self: T, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)

        dict_ = self._uuid_to_str(dict_)

        return dict_

    def _uuid_to_str(self, item: Any) -> Any:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, UUID):
                    item[key] = str(value)
                elif isinstance(value, list):
                    item[key] = [self._uuid_to_str(v) for v in value]
                elif isinstance(value, dict):
                    item[key] = {k: self._uuid_to_str(v) for k, v in value.items()}

        return item
```

Unlike the Mongo base, this recursion reaches `metadata` dicts and lists, which is why Qdrant payloads like `metadata={"html_url": ..., "parent_id": ...}` stay serializable.

**Bulk insert with just-in-time collection creation** (verbatim):

```python
    @classmethod
    def bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> bool:
        try:
            cls._bulk_insert(documents)
        except exceptions.UnexpectedResponse:
            logger.info(
                f"Collection '{cls.get_collection_name()}' does not exist. Trying to create the collection and reinsert the documents."
            )

            cls.create_collection()

            try:
                cls._bulk_insert(documents)
            except exceptions.UnexpectedResponse:
                logger.error(f"Failed to insert documents in '{cls.get_collection_name()}'.")

                return False

        return True

    @classmethod
    def _bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> None:
        points = [doc.to_point() for doc in documents]

        connection.upsert(collection_name=cls.get_collection_name(), points=points)
```

**Search and scroll** (verbatim):

```python
    @classmethod
    def bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[list[T], UUID | None]:
        try:
            documents, next_offset = cls._bulk_find(limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            logger.error(f"Failed to search documents in '{cls.get_collection_name()}'.")

            documents, next_offset = [], None

        return documents, next_offset

    @classmethod
    def _bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[list[T], UUID | None]:
        collection_name = cls.get_collection_name()

        offset = kwargs.pop("offset", None)
        offset = str(offset) if offset else None

        records, next_offset = connection.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", False),
            offset=offset,
            **kwargs,
        )
        documents = [cls.from_record(record) for record in records]
        if next_offset is not None:
            next_offset = UUID(next_offset, version=4)

        return documents, next_offset

    @classmethod
    def search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]:
        try:
            documents = cls._search(query_vector=query_vector, limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            logger.error(f"Failed to search documents in '{cls.get_collection_name()}'.")

            documents = []

        return documents

    @classmethod
    def _search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]:
        collection_name = cls.get_collection_name()
        records = connection.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", False),
            **kwargs,
        )
        documents = [cls.from_record(record) for record in records]

        return documents
```

`_bulk_find` wraps Qdrant's `scroll`, returning a `(documents, next_offset)` tuple so callers can paginate through a large collection. `search` runs actual vector similarity; `scroll` does *not*.

**Collection management** (verbatim):

```python
    @classmethod
    def get_or_create_collection(cls: Type[T]) -> CollectionInfo:
        collection_name = cls.get_collection_name()

        try:
            return connection.get_collection(collection_name=collection_name)
        except exceptions.UnexpectedResponse:
            use_vector_index = cls.get_use_vector_index()

            collection_created = cls._create_collection(
                collection_name=collection_name, use_vector_index=use_vector_index
            )
            if collection_created is False:
                raise RuntimeError(f"Couldn't create collection {collection_name}") from None

            return connection.get_collection(collection_name=collection_name)

    @classmethod
    def create_collection(cls: Type[T]) -> bool:
        collection_name = cls.get_collection_name()
        use_vector_index = cls.get_use_vector_index()

        return cls._create_collection(collection_name=collection_name, use_vector_index=use_vector_index)

    @classmethod
    def _create_collection(cls, collection_name: str, use_vector_index: bool = True) -> bool:
        if use_vector_index is True:
            vectors_config = VectorParams(size=EmbeddingModelSingleton().embedding_size, distance=Distance.COSINE)
        else:
            vectors_config = {}

        return connection.create_collection(collection_name=collection_name, vectors_config=vectors_config)
```

**Key Concepts**:
- **Just-in-time collection creation**: `bulk_insert` creates the collection on first use instead of requiring a migration step.
- **`use_vector_index`**: collections used only as string stores set `use_vector_index = False`; the vector size comes from the embedding singleton, so the collection dimension can never drift from the model dimension.
- **Dimension source of truth**: `EmbeddingModelSingleton().embedding_size` for `all-MiniLM-L6-v2` is **384**. A Qdrant collection with `size=384` rejects anything else.

**Config accessors** (verbatim):

```python
    @classmethod
    def get_category(cls: Type[T]) -> DataCategory:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "category"):
            raise ImproperlyConfigured(
                "The class should define a Config class with"
                "the 'category' property that reflects the collection's data category."
            )

        return cls.Config.category

    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "name"):
            raise ImproperlyConfigured(
                "The class should define a Config class withthe 'name' property that reflects the collection's name."
            )

        return cls.Config.name

    @classmethod
    def get_use_vector_index(cls: Type[T]) -> bool:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "use_vector_index"):
            return True

        return cls.Config.use_vector_index
```

`get_use_vector_index` defaults to `True` when the attribute is absent. That is why the embedded chunk classes can write it explicitly and the chunk classes can omit it (chunks are never inserted directly in the current flow, but if they were, they would be treated as vector collections).

**Grouping utilities** (verbatim):

```python
    @classmethod
    def group_by_class(
        cls: Type["VectorBaseDocument"], documents: list["VectorBaseDocument"]
    ) -> Dict["VectorBaseDocument", list["VectorBaseDocument"]]:
        return cls._group_by(documents, selector=lambda doc: doc.__class__)

    @classmethod
    def group_by_category(cls: Type[T], documents: list[T]) -> Dict[DataCategory, list[T]]:
        return cls._group_by(documents, selector=lambda doc: doc.get_category())

    @classmethod
    def _group_by(cls: Type[T], documents: list[T], selector: Callable[[T], Any]) -> Dict[Any, list[T]]:
        grouped = {}
        for doc in documents:
            key = selector(doc)

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(doc)

        return grouped
```

**Reverse lookup** (verbatim):

```python
    @classmethod
    def collection_name_to_class(cls: Type["VectorBaseDocument"], collection_name: str) -> type["VectorBaseDocument"]:
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
```

This is used by the RAG retriever to map a query back to the right collection class. Note it recurses over *direct subclasses only*; if you nest your class hierarchy deeper than the current one level, verify this still finds your class.

---

### 4. `llm_engineering/domain/documents.py` - Raw Documents (MongoDB)

**Purpose**: The entities the crawlers write, before any cleaning.

```python
# llm_engineering/domain/documents.py  (verbatim)
from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from .base import NoSQLBaseDocument
from .types import DataCategory


class UserDocument(NoSQLBaseDocument):
    first_name: str
    last_name: str

    class Settings:
        name = "users"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"


class Document(NoSQLBaseDocument, ABC):
    content: dict
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")


class RepositoryDocument(Document):
    name: str
    link: str

    class Settings:
        name = DataCategory.REPOSITORIES


class PostDocument(Document):
    image: Optional[str] = None
    link: str | None = None

    class Settings:
        name = DataCategory.POSTS


class ArticleDocument(Document):
    link: str

    class Settings:
        name = DataCategory.ARTICLES
```

**Key Concepts**:
- **`UserDocument` is the root entity**. Every document references `author_id` and `author_full_name`, which become the metadata used later for self-query filtering.
- **`Document` is abstract** and shared by the three platform documents.
- **`content` is a `dict`** at this stage. The crawler stores a platform-specific structure (title, subtitle, body) that cleaning will later collapse into a single string.
- **`Settings.name` uses `DataCategory` members** for the three document types, so collection names stay consistent: `posts`, `articles`, `repositories`. `UserDocument` uses the literal `"users"` because there is no `USERS` enum member.
- **`author_id`/`author_full_name` use aliases** identical to the field names. This is a forward-compatibility hook: `by_alias=True` is the default in `to_mongo`, so the alias is what lands in Mongo.

---

### 5. `llm_engineering/domain/cleaned_documents.py` - Cleaned Documents

**Purpose**: The output of the cleaning step, stored as vector-collection payloads but **without** embeddings.

```python
# llm_engineering/domain/cleaned_documents.py  (verbatim)
from abc import ABC
from typing import Optional

from pydantic import UUID4

from .base import VectorBaseDocument
from .types import DataCategory


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: UUID4
    author_full_name: str


class CleanedPostDocument(CleanedDocument):
    image: Optional[str] = None

    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


class CleanedArticleDocument(CleanedDocument):
    link: str

    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class CleanedRepositoryDocument(CleanedDocument):
    name: str
    link: str

    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False
```

**Key Concepts**:
- `content` is now a `str` (cleaned text), not a dict.
- **`use_vector_index = False`**: cleaned documents are a staging store in Qdrant, not a searchable vector index. This is the case that `to_point` / `from_record` must tolerate with no embedding.
- Transition `NoSQLBaseDocument → VectorBaseDocument` is deliberate: raw data lives in MongoDB, cleaned data lives in Qdrant.
- They keep `author_id` and `author_full_name` so downstream steps can build self-query filters and citations.

---

### 6. `llm_engineering/domain/chunks.py` - Chunks

**Purpose**: Cleaned documents split into retrieval-sized pieces.

```python
# llm_engineering/domain/chunks.py  (verbatim)
from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class Chunk(VectorBaseDocument, ABC):
    content: str
    platform: str
    document_id: UUID4
    author_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)


class PostChunk(Chunk):
    image: Optional[str] = None

    class Config:
        category = DataCategory.POSTS


class ArticleChunk(Chunk):
    link: str

    class Config:
        category = DataCategory.ARTICLES


class RepositoryChunk(Chunk):
    name: str
    link: str

    class Config:
        category = DataCategory.REPOSITORIES
```

**Key Concepts**:
- **`document_id` links a chunk back to its parent** cleaned document. This is the foreign key that enables citations.
- **`metadata`** is a free dict that carries chunker-specific info (for example the chunk index).
- The chunk has no `name`/`link` on the base class; platform-specific fields live on the subclasses.
- These `Config` classes define only `category`, not `name`. They are never persisted directly in the current pipeline (chunking produces chunks, embedding produces embedded chunks that *are* persisted), which is why the missing `name` does not raise at runtime. If you try to `bulk_insert` a `Chunk`, `get_collection_name()` raises `ImproperlyConfigured`.

---

### 7. `llm_engineering/domain/embedded_chunks.py` - Embedded Chunks

**Purpose**: Chunks plus their embedding. These are the searchable vectors used at inference time.

```python
# llm_engineering/domain/embedded_chunks.py  (verbatim)
from abc import ABC

from pydantic import UUID4, Field

from llm_engineering.domain.types import DataCategory

from .base import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: list[float] | None
    platform: str
    document_id: UUID4
    author_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Author: {chunk.author_full_name}
            Content: {chunk.content}\n
            """

        return context


class EmbeddedPostChunk(EmbeddedChunk):
    class Config:
        name = "embedded_posts"
        category = DataCategory.POSTS
        use_vector_index = True


class EmbeddedArticleChunk(EmbeddedChunk):
    link: str

    class Config:
        name = "embedded_articles"
        category = DataCategory.ARTICLES
        use_vector_index = True


class EmbeddedRepositoryChunk(EmbeddedChunk):
    name: str
    link: str

    class Config:
        name = "embedded_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = True
```

**Key Concepts**:
- **`use_vector_index = True`**: these are the only queryable vector collections in Qdrant.
- **`embedding: list[float] | None`**: the `_has_class_attribute("embedding")` check is what makes `from_record` populate this field only for embedded chunk classes.
- **`to_context`** is the bridge from retrieval to generation. It renders the retrieved chunks into the prompt context that the RAG chain passes to the LLM. Note it emits `Type` as the Python class name (e.g. `EmbeddedArticleChunk`), not the `platform`; it also emits both.
- `Type` and `Platform` are slightly redundant; keep this in mind if you tune the prompt.

---

### 8. `llm_engineering/domain/dataset.py` - Training Datasets

**Purpose**: The entities produced by dataset generation and consumed by training.

```python
# llm_engineering/domain/dataset.py  (verbatim, abridged imports)
from enum import Enum

from loguru import logger

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets
except ImportError:
    logger.warning("Huggingface datasets not installed. Install with `pip install datasets`")


from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class DatasetType(Enum):
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"


class InstructDatasetSample(VectorBaseDocument):
    instruction: str
    answer: str

    class Config:
        category = DataCategory.INSTRUCT_DATASET_SAMPLES


class PreferenceDatasetSample(VectorBaseDocument):
    instruction: str
    rejected: str
    chosen: str

    class Config:
        category = DataCategory.PREFERENCE_DATASET_SAMPLES


class InstructDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[InstructDatasetSample]

    class Config:
        category = DataCategory.INSTRUCT_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]

        return Dataset.from_dict(
            {"instruction": [d["instruction"] for d in data], "output": [d["answer"] for d in data]}
        )
```

```python
class TrainTestSplit(VectorBaseDocument):
    train: dict
    test: dict
    test_split_size: float

    def to_huggingface(self, flatten: bool = False) -> "DatasetDict":
        train_datasets = {category.value: dataset.to_huggingface() for category, dataset in self.train.items()}
        test_datasets = {category.value: dataset.to_huggingface() for category, dataset in self.test.items()}

        if flatten:
            train_datasets = concatenate_datasets(list(train_datasets.values()))
            test_datasets = concatenate_datasets(list(test_datasets.values()))
        else:
            train_datasets = Dataset.from_dict(train_datasets)
            test_datasets = Dataset.from_dict(test_datasets)

        return DatasetDict({"train": train_datasets, "test": test_datasets})


class InstructTrainTestSplit(TrainTestSplit):
    train: dict[DataCategory, InstructDataset]
    test: dict[DataCategory, InstructDataset]
    test_split_size: float

    class Config:
        category = DataCategory.INSTRUCT_DATASET
```

```python
class PreferenceDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[PreferenceDatasetSample]

    class Config:
        category = DataCategory.PREFERENCE_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]

        return Dataset.from_dict(
            {
                "prompt": [d["instruction"] for d in data],
                "rejected": [d["rejected"] for d in data],
                "chosen": [d["chosen"] for d in data],
            }
        )


class PreferenceTrainTestSplit(TrainTestSplit):
    train: dict[DataCategory, PreferenceDataset]
    test: dict[DataCategory, PreferenceDataset]
    test_split_size: float

    class Config:
        category = DataCategory.PREFERENCE_DATASET


def build_dataset(dataset_type, *args, **kwargs) -> InstructDataset | PreferenceDataset:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructDataset(*args, **kwargs)
    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDataset(*args, **kwargs)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
```

**Key Concepts**:
- **`to_huggingface()`** maps the internal schema to the exact column names the training stack expects: `instruction`/`output` for SFT, `prompt`/`chosen`/`rejected` for DPO.
- **`TrainTestSplit`** wraps `train` / `test` dicts keyed by `DataCategory` and can `flatten` them into a single `datasets.DatasetDict`.
- **`test_split_size`** is a float (fraction), stored so the split is reproducible and reportable in ZenML artifact metadata.
- **`build_dataset`** is a small factory so callers pass a `DatasetType` rather than branching.
- **Import guard**: if `datasets` is not installed, the module still imports but `to_huggingface` will fail. This keeps the domain importable in environments that only need the data models.
- Note these classes are `VectorBaseDocument` subclasses but are *not* Qdrant collections in the usual sense; their `Config` has only `category` (no `name`), so they are carried as ZenML artifacts, not inserted by `bulk_insert`.

---

### 9. `llm_engineering/domain/queries.py` and `prompt.py`

**Purpose**: The inference-side and generation-side value objects.

```python
# llm_engineering/domain/queries.py  (verbatim)
from pydantic import UUID4, Field

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class Query(VectorBaseDocument):
    content: str
    author_id: UUID4 | None = None
    author_full_name: str | None = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        category = DataCategory.QUERIES

    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))

    def replace_content(self, new_content: str) -> "Query":
        return Query(
            id=self.id,
            content=new_content,
            author_id=self.author_id,
            author_full_name=self.author_full_name,
            metadata=self.metadata,
        )


class EmbeddedQuery(Query):
    embedding: list[float]

    class Config:
        category = DataCategory.QUERIES
```

`replace_content` is what query expansion uses: it keeps the original `id` and metadata while swapping in a rewritten query, so traces stay grouped.

```python
# llm_engineering/domain/prompt.py  (verbatim)
from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.types import DataCategory


class Prompt(VectorBaseDocument):
    template: str
    input_variables: dict
    content: str
    num_tokens: int | None = None

    class Config:
        category = DataCategory.PROMPT


class GenerateDatasetSamplesPrompt(Prompt):
    data_category: DataCategory
    document: CleanedDocument
```

`GenerateDatasetSamplesPrompt` embeds the source `CleanedDocument` so a single prompt object carries both the rendered text and its provenance for the dataset generator.

---

## 🧠 Design Patterns Used

### 1. **Generic Repository (ODM / OVM)**

Instead of a separate `DocumentRepository` class, persistence is generic over `T`:

```python
T = TypeVar("T", bound="NoSQLBaseDocument")


class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None: ...
```

Callers get full type inference: `UserDocument.find(...)` returns `UserDocument | None`.

### 2. **Template Method**

The base defines the persist/insert skeleton; `to_mongo` / `to_point` are the overridable translation steps. Subclasses only declare fields and a `Config`/`Settings` class.

### 3. **Configuration via inner class**

`Settings` (Mongo) and `Config` (Qdrant) declare `name` and `category`. The base validates their presence at call time and raises `ImproperlyConfigured`.

### 4. **Public/Private split**

`bulk_insert` / `_bulk_insert`, `search` / `_search`, `bulk_find` / `_bulk_find`. The public method handles failures and logging; the private method does the work. This is the book's explicit recommendation ("one private function contains the logic, one public handles errors").

### 5. **Singleton accessor for connections**

`connection` is created at import time in `infrastructure/db/*.py`; the domain layer just imports it. (Covered in depth in Session 1.3.)

---

## 🛠️ Hands-On: Create a Custom Document Model

### Step 1: Define a new raw document

```python
# llm_engineering/domain/documents.py
class TweetDocument(Document):
    link: str
    hashtags: list[str] = []

    class Settings:
        name = "tweets"
```

### Step 2: Define its cleaned and chunked forms

```python
# cleaned_documents.py
class CleanedTweetDocument(CleanedDocument):
    link: str

    class Config:
        name = "cleaned_tweets"
        category = DataCategory.POSTS  # reuse a category if no new enum value exists
        use_vector_index = False
```

### Step 3: Persist and read it

```python
from llm_engineering.domain.documents import TweetDocument

doc = TweetDocument(
    content={"text": "Hello world"},
    platform="twitter",
    author_id=user.id,
    author_full_name=user.full_name,
    link="https://x.com/...",
    hashtags=["#llm"],
)
saved = doc.save()                 # writes to the "tweets" collection
found = TweetDocument.find(_id=str(doc.id))
print(found == doc)                # True (equality by id)
```

### Step 4: Inspect collections in MongoDB

```bash
# inside the container
docker compose exec mongo mongosh \
  "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017/twin" \
  --eval "db.getCollectionNames()"
```

### Step 5 (worked numeric example): verify a Qdrant collection

```python
from llm_engineering.domain.embedded_chunks import EmbeddedArticleChunk
from llm_engineering.application.networks.embeddings import EmbeddingModelSingleton

print("Embedding size:", EmbeddingModelSingleton().embedding_size)   # 384
info = EmbeddedArticleChunk.get_or_create_collection()
print("Vector size:", info.config.params.vectors.size)              # 384
print("Distance:   ", info.config.params.vectors.distance)          # COSINE
```

If `embedding_size` is 384 and a query vector has 768 elements, `search()` raises `UnexpectedResponse`, which the public `search()` catches and turns into `[]`. A silent empty result is the classic symptom of a wrong embedding model.

---

## 📝 Exercise 1: Trace the Lifecycle of One Entity

### Task

Follow a single `ArticleDocument` from creation to retrievable chunk.

1. Create an `ArticleDocument` and `save()` it → collection `articles`.
2. Clean it into a `CleanedArticleDocument` and `bulk_insert()` → collection `cleaned_articles`.
3. Chunk it into `ArticleChunk` objects → category `articles`.
4. Embed into `EmbeddedArticleChunk` and `bulk_insert()` → collection `embedded_articles`.
5. `EmbeddedArticleChunk.search(query_vector=...)` and inspect `metadata`, `document_id`, `author_full_name`.

**Goal**: See how `id`, `document_id`, and `author_id` thread through every stage.

---

## 📝 Exercise 2: Prove the Translation Seam Round-Trips

### Task

Write a script that proves `to_mongo`/`from_mongo` and `to_point`/`from_record` are lossless for the fields they own.

```python
import uuid
from llm_engineering.domain.documents import ArticleDocument
from llm_engineering.domain.embedded_chunks import EmbeddedArticleChunk

# 1. Mongo round trip (do not hit the DB)
doc = ArticleDocument(
    content={"text": "hello"},
    platform="medium",
    author_id=uuid.uuid4(),
    author_full_name="Paul Iusztin",
    link="https://example.com/a",
)
mongo_doc = doc.to_mongo()
assert "_id" in mongo_doc and "id" not in mongo_doc      # id renamed
assert isinstance(mongo_doc["_id"], str)                  # UUID stringified
restored = ArticleDocument.from_mongo(dict(mongo_doc))
assert restored == doc                                    # equality by id

# 2. Qdrant round trip (do not hit the DB)
chunk = EmbeddedArticleChunk(
    content="hello world",
    embedding=[0.1] * 384,
    platform="medium",
    document_id=doc.id,
    author_id=doc.author_id,
    author_full_name=doc.author_full_name,
    metadata={"index": 0},
    link="https://example.com/a",
)
point = chunk.to_point()
assert isinstance(point.id, str)
assert point.vector and len(point.vector) == 384
assert "embedding" not in point.payload                   # vector is not in payload
```

Then extend it: create a plain `CleanedArticleDocument` (no `embedding` field), call `to_point()`, and confirm `vector` is `{}`/empty and no `embedding` key appears. Finally call `CleanedArticleDocument.from_record` with a synthetic `Record` and confirm it constructs.

**Goal**: internalize the two seams and the `_has_class_attribute("embedding")` branch.

---

## ⚠️ Common Pitfalls

1. **The book's `from_record` differs from the repo.** The printed book writes `payload["embedding"] = ...`; the repo writes to `attributes`. Using the book's version leaks `embedding` into the payload and crashes classes that have no `embedding` field. Trust `domain/base/vector.py`.
2. **`self.dict()` is Pydantic v1.** The book's `to_point` shows `self.dict(...)`. The repo uses `self.model_dump(...)`. On Pydantic v2, `.dict()` is deprecated and warns; use `model_dump`.
3. **`from_mongo` mutates its input.** It calls `data.pop("_id")`, so the caller's dict loses `_id`. Pass a copy if you still need the raw document.
4. **Equality ignores everything but `id`.** `assert user == fetched` passes even if every other field differs. Do not use `==` to validate that a save preserved field values; compare `model_dump()`.
5. **`get_or_create` is not an atomic upsert.** Two concurrent callers can insert duplicates. There is no unique index assumed. For dedup, prefer a stable deterministic id or add a Mongo unique index.
6. **Qdrant point ids must be UUID or unsigned int.** `to_point` stringifies `id`, and `from_record` does `UUID(point.id, version=4)`. A non-UUID, non-integer id raises. Keep `id` a UUID4.
7. **Vector dimension must equal `embedding_size`.** `_create_collection` reads it from the embedding singleton (384 for `all-MiniLM-L6-v2`). Inserting a differently sized vector fails; a mismatched *query* vector returns `[]` silently.
8. **`use_vector_index` defaults to `True` when absent.** A `Config` that omits it is treated as a vector collection. Only set it to `False` when you truly want a payload-only store.
9. **Chunks have no `Config.name`.** `ArticleChunk.get_collection_name()` raises `ImproperlyConfigured`. Chunks are in-memory intermediates; embedded chunks are what get stored. Do not try to insert a bare `Chunk`.
10. **`metadata` must be JSON-serializable.** Nested UUIDs are handled by `_uuid_to_str`, but arbitrary Python objects are not. Qdrant will reject or coerce them.
11. **`bulk_find` on Qdrant is a scroll, not a search.** It returns in arbitrary order and ignores similarity. Use `search()` for retrieval.
12. **Reloading a class after changing `Config` does not recreate an existing collection.** Qdrant collections are created once with a fixed dimension. If you change the embedding model, delete and recreate the collection.
13. **`collection_name_to_class` only recurses direct subclasses.** Deep hierarchies may not resolve; test after restructuring.
14. **Import time connects.** `nosql.py` builds `_database` at import; unit tests need settings and a reachable/mocked Mongo client.

---

## 🎓 Knowledge Check

1. **Why does `NoSQLBaseDocument` override `model_dump`?**
   - Answer: To convert top-level `UUID` field values into plain `str` so they can be stored in Mongo.

2. **What is the difference between `to_mongo` and `to_point`?**
   - Answer: `to_mongo` emits a Mongo document with `_id`; `to_point` emits a Qdrant `PointStruct` with a separate `vector` and `payload`.

3. **Why do `CleanedDocument`s set `use_vector_index = False`?**
   - Answer: They are a staging payload store with no embedding and are never searched by vector similarity.

4. **What does `to_context` produce, and who consumes it?**
   - Answer: A formatted string of the retrieved chunks; the RAG prompt builder consumes it during generation.

5. **How is de-duplication across query variations implemented?**
   - Answer: Documents hash by `id`, so `{doc.id: doc for doc in docs}.values()` collapses duplicates.

6. **Where does a chunk get its link back to the source?**
   - Answer: `document_id` on `Chunk`, pointing at the parent cleaned document.

7. **What is the ODM and what is the OVM?**
   - Answer: ODM = object-document mapping for MongoDB (`NoSQLBaseDocument`); OVM = object-vector mapping for Qdrant (`VectorBaseDocument`).

8. **Why two base classes instead of one?**
   - Answer: The engines have different identity, payload, and query models; two bases isolate each store's translation seam.

9. **What is `_has_class_attribute("embedding")` for?**
   - Answer: To populate the `embedding` field from the point's vector only for classes that declare it (embedded chunks), not for cleaned docs.

10. **What does `get_use_vector_index` return if `Config.use_vector_index` is missing?**
    - Answer: `True` (treat as a vector collection).

11. **Why can `ArticleChunk.get_collection_name()` fail?**
    - Answer: `Chunk` subclasses define `Config.category` but not `Config.name`; the accessor requires `name`.

12. **How does `bulk_insert` handle a missing Qdrant collection?**
    - Answer: It catches `UnexpectedResponse`, calls `create_collection()`, and retries the insert once.

13. **What two things does `TrainTestSplit.to_huggingface` do differently when `flatten=True`?**
    - Answer: It `concatenate_datasets` across categories instead of building a nested dict.

14. **Which column names does `InstructDataset.to_huggingface` emit, and why?**
    - Answer: `instruction` and `output`, matching the SFT training stack's expected schema.

15. **Why does the domain layer keep no reference to ZenML?**
    - Answer: So entities and persistence are reusable and testable independent of the orchestrator.

---

## 📖 Glossary

- **ODM**: Object-Document Mapping; Python objects ⇄ MongoDB documents. The book's name for `NoSQLBaseDocument`.
- **OVM**: Object-Vector Mapping; Python objects ⇄ Qdrant points. The book's name for `VectorBaseDocument`.
- **Entity identity**: equality/hash by `id`, not field values.
- **Payload**: the non-vector attributes stored alongside a Qdrant point.
- **Point**: a Qdrant record: id + optional vector + payload.
- **Scroll**: Qdrant's paginated listing of points without similarity search.
- **Collection**: Mongo's grouping of documents / Qdrant's grouping of points.
- **`use_vector_index`**: whether a Qdrant collection is created with a vector index and dimension.
- **`_id`**: MongoDB primary key; stringified from the model's `id`.
- **`document_id`**: foreign key from a chunk to its parent cleaned document.
- **Cleaned / chunked / embedded**: the three data states of the domain.
- **`ImproperlyConfigured`**: raised when a subclass forgets its `Settings`/`Config` configuration.

---

## 🔗 Next Session

**Session 1.3**: [Infrastructure Layer - Database Connections](./session_1.3_infrastructure_layer.md)

We'll cover:
- `MongoDatabaseConnector` and `QdrantDatabaseConnector` singletons
- Connection lifecycle and lazy environment loading
- `Settings.load_settings()` and the ZenML secret store
- `JsonFileManager` and `docker-compose.yml`

---

## 📚 Additional Resources

- [Pydantic Generics](https://docs.pydantic.dev/latest/concepts/models/#generic-models)
- [PyMongo CRUD](https://pymongo.readthedocs.io/en/stable/tutorial.html)
- [Qdrant Points and Collections](https://qdrant.tech/documentation/concepts/points/)
- [Qdrant Scroll API](https://qdrant.tech/documentation/concepts/points/#scroll-points)
- [SQLAlchemy ORM (for contrast)](https://docs.sqlalchemy.org/en/20/orm/)
- [mongoengine (a real ODM)](https://mongoengine.org/)
- [Curriculum map](../CURRICULUM.md)

---

## 📎 References

- Iusztin, P., & Labonne, M. *LLM Engineer's Handbook*. Packt. Chapter 3 "The ORM and ODM software patterns" and "Implementing the ODM class" (pages 108-124); Chapter 4 "Pydantic domain entities" and "OVM" (pages 179-189).
- Repository source files read for this session: `llm_engineering/domain/base/nosql.py`, `llm_engineering/domain/base/vector.py`, `llm_engineering/domain/base/__init__.py`, `llm_engineering/domain/types.py`, `llm_engineering/domain/documents.py`, `llm_engineering/domain/cleaned_documents.py`, `llm_engineering/domain/chunks.py`, `llm_engineering/domain/embedded_chunks.py`, `llm_engineering/domain/dataset.py`, `llm_engineering/domain/queries.py`, `llm_engineering/domain/prompt.py`, `llm_engineering/domain/exceptions.py`, `llm_engineering/application/networks/embeddings.py`.

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Session 1.1

**Outcome**: You can design and persist domain entities against both MongoDB (ODM) and Qdrant (OVM) using the project's base classes, and you understand the translation seams and their failure modes.
