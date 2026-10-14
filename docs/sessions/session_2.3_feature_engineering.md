# Session 2.3: Feature Engineering Pipeline (Embedding and Loading)

> Book reference: Chapter 4, *RAG Feature Pipeline* (pages 168-203).
> Repo: `llm_engineering/application/networks/*.py`, `llm_engineering/application/preprocessing/embedding_data_handlers.py`, `steps/feature_engineering/*.py`, `pipelines/feature_engineering.py`.

## 🎯 Learning Objectives

By the end of this session, you will:
- Explain `SingletonMeta` and why model loading needs a lock.
- Read `EmbeddingModelSingleton` and `CrossEncoderModelSingleton` and their public surface.
- Trace chunk → embedded chunk through `EmbeddingDataHandler.embed_batch`.
- Understand `EmbeddingDispatcher`'s homogeneity assertion and single/list return shape.
- Read the `feature_engineering` ZenML pipeline end to end: query, clean, chunk+embed, load.
- Reason about batching, device placement, idempotency, and Qdrant point shape.
- Know the workstation constraints: `all-MiniLM-L6-v2` is 384-dim, 256-token, CPU by default.

---

## 🏗️ Architecture Overview

### The end-to-end feature pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                    feature_engineering pipeline                          │
│                                                                          │
│  query_data_warehouse(author_full_names, after=wait_for)                 │
│        │  UserDocument.get_or_create + 3 parallel Mongo queries          │
│        ▼  raw docs (articles + posts + repositories)                     │
│  clean_documents(raw_documents)                                          │
│        │  CleaningDispatcher → CleanedDocument                           │
│        ├────────────────────────────────────► load_to_vector_db          │
│        │                                      (cleaned, no vectors)      │
│        ▼                                                                 │
│  chunk_and_embed(cleaned_documents)      [steps/feature_engineering/rag.py]│
│        │  ChunkingDispatcher → chunks                                    │
│        │  EmbeddingDispatcher (batch=10) → EmbeddedChunk                 │
│        ▼                                                                 │
│  load_to_vector_db(embedded_documents)                                   │
│        │  group_by_class + batch(4) + bulk_insert (upsert)               │
│        ▼                                                                 │
│  Qdrant: cleaned_articles/posts/repos  +  embedded_articles/posts/repos  │
└────────────────────────────────────────────────────────────────────────┘
```

The pipeline loads **two** snapshot families into Qdrant:
1. **Cleaned documents** (`use_vector_index=False`) — unindexed payload store, used for fine-tuning.
2. **Embedded chunks** (`use_vector_index=True`) — searchable vectors, used for RAG.

### The DAG

```
query_data_warehouse
        │
        ▼
clean_documents
   ├──────────────► load_to_vector_db (cleaned)   → last_step_1
   │
   ▼
chunk_and_embed
        │
        ▼
load_to_vector_db (embedded)                     → last_step_2
```

`cleaned_documents` fans out to two consumers; both `load_to_vector_db` calls return bools.

### The logical feature store

```
Online serving (RAG)  ──► Qdrant (vectors + payload)
Offline training       ──► ZenML artifacts (datasets in later sessions)
```

Both halves read from the same cleaned/embedded snapshots, keeping the warehouse generic.

---

## 📁 Key Files Explained

### 1. `llm_engineering/application/networks/base.py` — thread-safe singleton

```python
# llm_engineering/application/networks/base.py
from threading import Lock
from typing import ClassVar


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances: ClassVar = {}

    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]
```

**Key concepts**:

- **`SingletonMeta` is a metaclass**: `class Foo(metaclass=SingletonMeta)` intercepts *construction* of `Foo`, so `Foo()` always returns the same instance.
- **The `Lock` prevents a race.** Two threads entering `__call__` at startup would otherwise both pass the `not in _instances` check and build two models. The lock serializes the check-and-create.
- **`_instances` is keyed by class**, so `EmbeddingModelSingleton` and `CrossEncoderModelSingleton` get separate instances, but each subclass shares its own.
- **`__init__` arguments are ignored after the first call** (the comment in the source says exactly this). This is why you cannot change model settings at runtime: change `.env` and restart.
- **Contrast with `MongoDatabaseConnector`** (Session 1.3), which uses the simpler `__new__`-based singleton without a lock. Model construction is slow and non-atomic (downloads weights, moves to device), so it needs the lock; a Mongo client does not.

> **Caveat**: `_lock` is a single class-level lock shared by all subclasses, so constructing one model blocks constructing another. That is fine at startup and is actually desirable (serialize heavy loads).

### 2. `llm_engineering/application/networks/embeddings.py` — the models

```python
# llm_engineering/application/networks/embeddings.py
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer

from llm_engineering.settings import settings

from .base import SingletonMeta


class EmbeddingModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = settings.TEXT_EMBEDDING_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model_id = model_id
        self._device = device

        self._model = SentenceTransformer(
            self._model_id,
            device=self._device,
            cache_folder=str(cache_dir) if cache_dir else None,
        )
        self._model.eval()

    @property
    def model_id(self) -> str:
        return self._model_id

    @cached_property
    def embedding_size(self) -> int:
        dummy_embedding = self._model.encode("")
        return dummy_embedding.shape[0]

    @property
    def max_input_length(self) -> int:
        return self._model.max_seq_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._model.tokenizer

    def __call__(
        self, input_text: str | list[str], to_list: bool = True
    ) -> NDArray[np.float32] | list[float] | list[list[float]]:
        try:
            embeddings = self._model.encode(input_text)
        except Exception:
            logger.error(f"Error generating embeddings for {self._model_id=} and {input_text=}")
            return [] if to_list else np.array([])

        if to_list:
            embeddings = embeddings.tolist()

        return embeddings


class CrossEncoderModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = settings.RERANKING_CROSS_ENCODER_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
    ) -> None:
        self._model_id = model_id
        self._device = device

        self._model = CrossEncoder(
            model_name=self._model_id,
            device=self._device,
        )
        self._model.model.eval()

    def __call__(self, pairs: list[tuple[str, str]], to_list: bool = True) -> NDArray[np.float32] | list[float]:
        scores = self._model.predict(pairs)

        if to_list:
            scores = scores.tolist()

        return scores
```

**Public surface and why each member exists**:

| Member | Kind | Purpose |
|--------|------|---------|
| `model_id` | property | provenance; stored on every vector's metadata |
| `embedding_size` | `cached_property` | sizes every Qdrant collection; probed once by encoding `""` |
| `max_input_length` | property | `max_seq_length`; feeds `chunk_text` token size (Session 2.2) |
| `tokenizer` | property | exposed for token counting/analysis |
| `__call__` | method | the encode function; returns list or ndarray |

**Key concepts**:

- **Inference only**: `self._model.eval()` switches off dropout etc. `SentenceTransformer.encode` already runs under `torch.no_grad()`, so the project does not wrap calls manually.
- **`embedding_size` is `cached_property`**: the dummy encode costs one forward pass and is cached; Qdrant collection creation reads it (`VectorBaseDocument._create_collection`).
- **`__call__` swallows errors**: on any exception it logs and returns `[]` (or an empty ndarray). The embedding handler then produces fewer documents rather than crashing the step. This pairs with `zip(..., strict=False)` downstream.
- **CrossEncoder is the reranker sibling**, used later (Session 4.1/4.2). `pairs` is a list of `(query, document)` tuples; it returns relevance scores. `self._model.model.eval()` reaches into the underlying `transformers` model.

**Settings defaults (verified in `settings.py`)**:

```python
TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
RAG_MODEL_DEVICE: str = "cpu"
```

`all-MiniLM-L6-v2` is 384-dimensional with a 256-token max sequence length — tiny and CPU-friendly.

> **Workstation note (RTX 5000, 16 GB)**: the default `RAG_MODEL_DEVICE="cpu"` keeps this off the GPU. If you point `TEXT_EMBEDDING_MODEL_ID` at a larger model and set `RAG_MODEL_DEVICE="cuda"`, batch size and `max_input_length` both scale activation VRAM. A 384-dim model at 256 tokens is trivial; a 1024-dim model at 512 tokens with batch 50 is not.

### 3. `llm_engineering/application/preprocessing/embedding_data_handlers.py` — chunks to vectors

```python
# llm_engineering/application/preprocessing/embedding_data_handlers.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.chunks import ArticleChunk, Chunk, PostChunk, RepositoryChunk
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from llm_engineering.domain.queries import EmbeddedQuery, Query

ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [data_model.content for data_model in data_model]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_model, embeddings, strict=False)
        ]

        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass
```

**Concrete mappers** (one per category). The article one:

```python
class ArticleEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: ArticleChunk, embedding: list[float]) -> EmbeddedArticleChunk:
        return EmbeddedArticleChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
```

The query handler:

```python
class QueryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Query, embedding: list[float]) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=data_model.id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            content=data_model.content,
            embedding=embedding,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
```

**Key concepts**:

- **`embed` vs `embed_batch`**: `embed` is a convenience wrapper around a single document; `embed_batch` is the real path. Encoding a whole list in one model call is far faster on GPU than one call per chunk, because it amortizes per-call overhead.
- **`id` is preserved** from the chunk. Because chunk ids are deterministic (Session 2.2), re-embedding overwrites the same Qdrant point.
- **Provenance metadata on every vector** (`embedding_model_id`, `embedding_size`, `max_input_length`). If the model changes, the collection dimension check and this metadata reveal the stale vectors.
- **`zip(..., strict=False)`** tolerates a length mismatch between inputs and embeddings, which is the deliberate pairing with the error-swallowing `__call__`. If embedding fails, fewer vectors are produced rather than raising.
- **`QueryEmbeddingHandler` output is never persisted**: `EmbeddedQuery` is embedded on the fly at query time and used only as the search vector.
- **Note the metadata overwrite**: `map_model` sets `metadata` to the embedding provenance only. The chunk's chunking metadata (`chunk_size`, etc.) lived on the `Chunk`, not carried onto the `EmbeddedChunk` by these handlers. The chunking metadata is captured instead in the step's ZenML output metadata (see `rag.py`).

### 4. `EmbeddingDispatcher` — batch entry point

```python
# llm_engineering/application/preprocessing/dispatchers.py (excerpt)
class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> EmbeddingDataHandler:
        if data_category == DataCategory.QUERIES:
            return QueryEmbeddingHandler()
        if data_category == DataCategory.POSTS:
            return PostEmbeddingHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleEmbeddingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type")


class EmbeddingDispatcher:
    factory = EmbeddingHandlerFactory

    @classmethod
    def dispatch(
        cls, data_model: VectorBaseDocument | list[VectorBaseDocument]
    ) -> VectorBaseDocument | list[VectorBaseDocument]:
        is_list = isinstance(data_model, list)
        if not is_list:
            data_model = [data_model]

        if len(data_model) == 0:
            return []

        data_category = data_model[0].get_category()
        assert all(data_model.get_category() == data_category for data_model in data_model), (
            "Data models must be of the same category."
        )
        handler = cls.factory.create_handler(data_category)

        embedded_chunk_model = handler.embed_batch(data_model)

        if not is_list:
            embedded_chunk_model = embedded_chunk_model[0]

        logger.info(
            "Data embedded successfully.",
            data_category=data_category,
        )

        return embedded_chunk_model
```

**Key concepts**:

- **Accepts a single document or a list**, returning the matching shape (or `[]` for an empty list). Keeps call sites clean.
- **Homogeneity assertion**: all documents in one batch must share a category, because each category maps to a different Qdrant collection with a different point payload shape.
- **Handles `QUERIES` too**, so the RAG retriever embeds a user query through the same code path.

### 5. `steps/feature_engineering/query_data_warehouse.py` — parallel extraction

```python
# steps/feature_engineering/query_data_warehouse.py
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.documents import ArticleDocument, Document, PostDocument, RepositoryDocument, UserDocument


@step
def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = []
    authors = []
    for author_full_name in author_full_names:
        logger.info(f"Querying data warehouse for user: {author_full_name}")

        first_name, last_name = utils.split_user_full_name(author_full_name)
        logger.info(f"First name: {first_name}, Last name: {last_name}")
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)

        results = fetch_all_data(user)
        user_documents = [doc for query_result in results.values() for doc in query_result]

        documents.extend(user_documents)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))

    return documents


def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }

        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")

                results[query_name] = []

    return results
```

**Key concepts**:

- **Three parallel Mongo queries** (`articles`, `posts`, `repositories`) via `ThreadPoolExecutor`; results are collected through `as_completed`.
- **Per-query failure isolation**: an exception in one future is logged and yields `[]`; the others still return.
- **`get_or_create`** makes the step re-runnable: an unknown author is created once, then reused.
- **`split_user_full_name`** (from `llm_engineering.application.utils`) handles multi-token names: `"Paul Iusztin"` → `("Paul", "Iusztin")`; a single token maps to itself as both parts.
- **`_get_metadata`** records per-collection counts and unique authors.

### 6. `steps/feature_engineering/rag.py` — chunk + embed

```python
# steps/feature_engineering/rag.py
@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    metadata = {"chunking": {}, "embedding": {}, "num_documents": len(cleaned_documents)}

    embedded_chunks = []
    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])

        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)

    metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"])
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="embedded_documents", metadata=metadata)

    return embedded_chunks
```

**Key concepts**:

- **`utils.misc.batch(chunks, 10)`**: embed 10 chunks per model call. This is the memory/speed dial. Larger batches are faster on GPU but use more VRAM.
- **Rich ZenML metadata**: per-category chunk counts, embedding provenance, and unique authors are attached to the step output.
- **Returns plain `EmbeddedChunk` objects**; persistence is a separate step.

`utils.misc.batch` is a generator:

```python
def batch(list_: list, size: int) -> Generator[list, None, None]:
    yield from (list_[i : i + size] for i in range(0, len(list_), size))
```

### 7. `steps/feature_engineering/load_to_vector_db.py` — upsert

```python
# steps/feature_engineering/load_to_vector_db.py
@step
def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> Annotated[bool, "successful"]:
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")
        for documents_batch in utils.misc.batch(documents, size=4):
            try:
                document_class.bulk_insert(documents_batch)
            except Exception:
                logger.error(f"Failed to insert documents into {document_class.get_collection_name()}")

                return False

    return True
```

**Key concepts**:

- **`group_by_class`** fans a mixed list into concrete classes (`CleanedArticleDocument`, `EmbeddedPostChunk`, ...). Each group goes to its own Qdrant collection.
- **Batch size 4** is deliberately small: Qdrant upserts are network-bound.
- **Returns a bool** so the pipeline can assert success downstream.

The underlying `VectorBaseDocument.bulk_insert` creates the collection on first use and upserts points:

```python
# llm_engineering/domain/base/vector.py (excerpt)
@classmethod
def bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> bool:
    try:
        cls._bulk_insert(documents)
    except exceptions.UnexpectedResponse:
        logger.info(f"Collection '{cls.get_collection_name()}' does not exist. Trying to create the collection ...")
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

`to_point()` splits the model into `id`, `vector` (the `embedding` field), and `payload` (everything else). Cleaned documents have no `embedding`, so their point vector is empty and the collection was created with `vectors_config = {}` (`use_vector_index=False`).

### 8. `pipelines/feature_engineering.py` — orchestration

```python
# pipelines/feature_engineering.py
from zenml import pipeline

from steps import feature_engineering as fe_steps


@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    return [last_step_1.invocation_id, last_step_2.invocation_id]
```

**Key concepts**:

- **`wait_for`** wires pipeline dependencies: `end_to_end_data` passes the ETL step invocation ids so feature engineering runs only after crawling completes.
- **Returns invocation ids** so the parent pipeline can chain dataset generation after it.

Config (`configs/feature_engineering.yaml`):

```yaml
settings:
  docker:
    parent_image: 992382797823.dkr.ecr.eu-central-1.amazonaws.com/zenml-rlwlcs:latest
    skip_build: True
  orchestrator.sagemaker:
    synchronous: false

parameters:
  author_full_names:
    - Maxime Labonne
    - Paul Iusztin
```

---

## 🛠️ Hands-On

### Step 1: Prerequisites

```bash
docker compose up -d
python -m tools.run --run-feature-engineering --no-cache
```

### Step 2: Inspect the collections

```python
from llm_engineering.infrastructure.db.qdrant import connection

for c in connection.get_collections().collections:
    info = connection.get_collection(c.name)
    print(c.name, info.points_count)
```

### Step 3: Embed and search a query

```python
from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.embedded_chunks import EmbeddedArticleChunk

model = EmbeddingModelSingleton()
query_vector = model("How does RAG work?", to_list=True)

results = EmbeddedArticleChunk.search(query_vector=query_vector, limit=3)
for r in results:
    print(r.author_full_name, "|", r.content[:80])
```

### Step 4: Confirm provenance metadata

```python
print(results[0].metadata)
# {'embedding_model_id': 'sentence-transformers/all-MiniLM-L6-v2', 'embedding_size': 384, 'max_input_length': 256}
```

### Step 5: Prove the singleton

```python
from llm_engineering.application.networks import EmbeddingModelSingleton

a = EmbeddingModelSingleton()
b = EmbeddingModelSingleton()
assert a is b
assert a.embedding_size == 384
```

---

## 📝 Exercise 1: Tune batching

**Task**: measure the effect of batch size on embedding throughput.

1. Wrap the embedding loop in `time.perf_counter()`.
2. Run with `batch(chunks, 1)`, `batch(chunks, 10)`, and `batch(chunks, 50)`.
3. Record wall-clock time and (if on GPU) peak VRAM via `nvidia-smi`.
4. Write down the tradeoff between throughput and memory.

**Goal**: understand why the pipeline uses 10 and where the ceiling is.

> **VRAM note**: with the default model (`all-MiniLM-L6-v2`, 384-dim, 256 tokens) and `RAG_MODEL_DEVICE=cpu`, this is a CPU exercise. If you switch to a larger model on `cuda`, batch size and `max_input_length` both scale activation VRAM on the 16 GB RTX 5000.

---

## 📝 Exercise 2: Detect stale vectors after a model swap

**Task**: build an operational check for embedding-model drift.

1. Pick a collection (e.g. `embedded_articles`) and read one point's `metadata`.
2. Compare `metadata["embedding_model_id"]` and `metadata["embedding_size"]` with the currently loaded `EmbeddingModelSingleton`.
3. Write a function that scrolls the collection and reports how many points have a mismatched model id.
4. Decide the remediation: re-embed (drop and rebuild) or versioned collections.

**Goal**: connect provenance metadata to a real MLOps concern — vectors are stale after any model change, and dimension mismatch will reject inserts.

---

## 🐛 Common Pitfalls

- **Device mismatch**: default device is `cpu`. Setting `RAG_MODEL_DEVICE="cuda"` without a compatible PyTorch/CUDA build fails at model load; the singleton then propagates the error to every consumer.
- **Stale vectors**: changing `TEXT_EMBEDDING_MODEL_ID` changes the vector dimension, so `bulk_insert` into the old collection fails (or silently mismatches). Re-embed into a fresh collection.
- **`embedding_size` probing cost**: it triggers one forward pass and is cached per process. Restarting the process re-probes.
- **`assert` for homogeneity**: assertions are stripped under `python -O`. Do not rely on them as the only guard in production code you run optimized.
- **Empty embedding list**: `__call__` returns `[]` on error, and `embed` indexes `[0]`, which would `IndexError` if called on a single failed item. `embed_batch` with `strict=False` is the safe path.
- **`wait_for` miswiring**: if feature engineering runs before ETL finishes, `query_data_warehouse` returns empty and the pipeline "succeeds" with zero documents.
- **CPU offload surprise**: embedding a large model on CPU is slow but memory-safe; on GPU it is fast but OOM-prone. Choose per model size.

---

## 🎓 Knowledge Check

1. **Why does `SingletonMeta` need a `Lock`?**
   - Answer: model construction is slow and non-atomic; without the lock two threads could build two models at startup.

2. **What sizes the Qdrant collections?**
   - Answer: `EmbeddingModelSingleton.embedding_size`, a cached property probed once from a dummy encoding.

3. **Why do embedding handlers attach the model id to each vector's metadata?**
   - Answer: to detect stale vectors if the embedding model changes.

4. **Why batch in `chunk_and_embed` and `load_to_vector_db`?**
   - Answer: to bound memory per model/network call while keeping throughput high.

5. **Why does `EmbeddingDispatcher` assert a single category per batch?**
   - Answer: each category maps to a different Qdrant collection with a different point shape.

6. **What two things does the pipeline load into Qdrant?**
   - Answer: cleaned documents (unindexed) and embedded chunks (searchable vectors).

7. **Why is `embed` implemented in terms of `embed_batch`?**
   - Answer: so there is one real code path; the single-document form just wraps a one-element batch.

8. **What does `zip(..., strict=False)` protect against here?**
   - Answer: a length mismatch when the model fails to embed some inputs and returns fewer vectors.

9. **Why is `all-MiniLM-L6-v2` cheap?**
   - Answer: 384 dimensions and 256-token max length, and it runs on CPU by default.

10. **How are the three Mongo queries in `query_data_warehouse` executed?**
    - Answer: in parallel via `ThreadPoolExecutor`, gathered through `as_completed`, with per-query failure isolation.

11. **What does `load_to_vector_db` return, and why?**
    - Answer: a bool; the pipeline can chain on success and the value is the step's output artifact.

12. **What does `wait_for` do in the pipeline signature?**
    - Answer: it lets a parent pipeline gate this pipeline on the completion of upstream steps (e.g. the ETL run).

13. **What is the point of the `tokenizer` property?**
    - Answer: to expose the model tokenizer for token counting and analysis.

14. **What happens when `EmbeddingDispatcher.dispatch` receives an empty list?**
    - Answer: it returns `[]` immediately.

15. **Why is the CrossEncoder a separate singleton from the embedding model?**
    - Answer: it is a different model type (pair scoring, not single-text encoding) with its own id and collection of use cases.

---

## 📖 Glossary

- **Singleton**: a class with exactly one instance per process.
- **Metaclass**: the class of a class; `SingletonMeta` customizes construction for its subclasses.
- **Embedding dimension**: the length of the vector a model outputs; it fixes the Qdrant collection size.
- **`max_seq_length`**: the model's token limit; drives chunk token size.
- **Cross-encoder / reranker**: scores a `(query, document)` pair directly; slower but more accurate than bi-encoder similarity.
- **Upsert**: insert-or-update by point id; with deterministic ids it is idempotent.
- **Payload**: the non-vector fields stored with a Qdrant point.
- **Provenance metadata**: metadata recording which model and parameters produced a vector.

---

## 🔗 Next Session

**Session 3.1: Instruction Dataset Creation**

We use the cleaned documents as seeds and generate instruction-answer pairs with GPT-4o-mini.

---

## 📚 Additional Resources

- [SentenceTransformers](https://www.sbert.net/)
- [Qdrant: points and upsert](https://qdrant.tech/documentation/concepts/points/#upload-points)
- [Python `concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)
- [Python `functools.cached_property`](https://docs.python.org/3/library/functools.html#functools.cached_property)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [ms-marco-MiniLM-L-4-v2 model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-4-v2)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 1.2, 1.3, 2.1, 2.2

**Outcome**: You can embed chunks, load them into Qdrant collection-by-collection, explain every step of the feature-engineering pipeline, and reason about batching, device placement, and stale-vector risk on the RTX 5000.
