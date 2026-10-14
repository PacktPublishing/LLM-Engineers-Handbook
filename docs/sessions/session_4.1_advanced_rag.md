# Session 4.1: Advanced RAG — The Retrieval Module

> Book: Chapter 9, "RAG Inference Pipeline" (pages 346-380).
> Repo: `llm_engineering/application/rag/` and `llm_engineering/application/networks/`.
> This is the *inference-side* retrieval module. The *ingestion* side (clean, chunk, embed, load) lives in the RAG feature pipeline (Session 2.3).

## 🎯 Learning Objectives

By the end of this session, you will:

- Understand why retrieval, not generation, is where most RAG engineering happens.
- Read the real `ContextRetriever.search()` end to end and trace every stage: self-query, query expansion, parallel filtered vector search, dedup, rerank.
- Explain the three advanced-RAG stages (pre-retrieval, retrieval, post-retrieval) and which class implements each.
- Use the actual `RAGStep` / `PromptTemplateFactory` abstractions instead of hand-wolfed prompt code.
- Compute latency and candidate-count arithmetic for an N-query, K-chunk retrieval.
- Recognize the failure modes of each stage and how the code guards them (`mock`, `assert k >= 3`, empty-result branch).
- Extend the pipeline with a router or hybrid search (exercises).

---

## 🏗️ Architecture Overview

### Where retrieval sits in RAG

An RAG system has two independent components:

```
┌──────────────────────────────┐        ┌──────────────────────────────┐
│   Ingestion (feature)        │        │   Inference (this session)   │
│   pipeline                   │        │                              │
│                              │        │   user query                 │
│  raw docs ──clean──chunk──   │        │        │                     │
│  embed──load ──► vector DB   │        │        ▼                     │
│                  (Qdrant)    │        │   ContextRetriever.search()  │
│                              │        │        │                     │
│   runs on a schedule         │        │   retrieved context          │
│   (offline, batch)           │        │        │                     │
└──────────────────────────────┘        │        ▼                     │
         populates the same DB ────────►│   LLM generates the answer   │
                                         └──────────────────────────────┘
```

The feature pipeline writes chunks on a schedule. The inference pipeline reads them on demand, once per user request. They share only the vector DB and the *same embedding model* (`EmbeddingModelSingleton`, via `EmbeddingDispatcher`). Using the same embedding model at ingestion and query time is mandatory: a query vector is only comparable to a stored vector if both came from the same encoder.

### The eight-step inference flow (book Figure 9.1)

```
User query
   │
   ▼
1. Query expansion      QueryExpansion.generate()   ── LLM, N variants
   │
   ▼
2. Self-querying        SelfQuery.generate()         ── LLM, extract author -> filter
   │
   ▼
3. Filtered vector search  _search() per category x per query
   │                        Qdrant Filter on author_id, limit = k//3
   ▼
4. Collect results      flatten + dedup (set)
   │
   ▼
5. Reranking            Reranker.generate()          ── cross-encoder, keep top k
   │
   ▼
6. Build prompt         prompt template + context + query   (Session 6.2)
   │
   ▼
7. Call LLM             SageMaker endpoint                  (Session 5.3/6.1)
   │
   ▼
8. Answer
```

Steps 1-2 are **pre-retrieval** (query optimization). Step 3 is **retrieval**. Steps 4-5 are **post-retrieval** (noise removal). Steps 6-8 are the generation half, covered in Session 6.2.

### Class map

```
                    ┌─────────────────────────┐
                    │  ContextRetriever       │  retriever.py
                    │  (orchestrator)         │
                    └───────────┬─────────────┘
                                │ owns
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  QueryExpansion          SelfQuery               Reranker
  (RAGStep)               (RAGStep)               (RAGStep)  reranking.py
        │                       │                       │
        │ uses                  │ uses                  │ uses
        ▼                       ▼                       ▼
  QueryExpansionTemplate  SelfQueryTemplate   CrossEncoderModelSingleton
  (PromptTemplateFactory) (PromptTemplateFactory)        networks/embeddings.py
        │                       │
        └──────────┬────────────┘
                   ▼
             RAGStep / PromptTemplateFactory   base.py
```

### Why the retrieval module carries the engineering

The generation call is one HTTP request to a fine-tuned LLM. The retrieval module is where you wrangle data so the retrieved context is actually relevant. The book is explicit: "At the retrieval step (and not when calling the LLM), you write most of the RAG inference code." Everything in this session optimizes that step.

---

## 📁 Key Files Explained

### 1. `base.py` — the two interfaces

Every advanced step subclasses `RAGStep`; every prompt subclasses `PromptTemplateFactory`. This is the contract that lets you swap implementations without touching `ContextRetriever`.

```python
# llm_engineering/application/rag/base.py
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
```

**Why two separate ABCs.** `PromptTemplateFactory` is `ABC + BaseModel` so templates can carry the prompt string as a Pydantic field (see `prompt_templates.py`). `RAGStep` is a plain ABC with a single `mock` flag. The `mock` flag is the cheapest test seam in the whole system: with `mock=True`, `ContextRetriever` returns the query unchanged and empty candidates flow through, so you can unit-test orchestration with zero LLM calls and zero model downloads.

**FACT (verified):** `RAGStep.generate` is abstract. `QueryExpansion`, `SelfQuery`, and `Reranker` all override it. `Reranker.generate` is the one that is NOT purely query-in / query-out; it takes `chunks` and `keep_top_k`.

---

### 2. `queries.py` — the query domain model

Retrieval does not pass bare strings around. It passes `Query` objects so metadata travels with the text.

```python
# llm_engineering/domain/queries.py
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

**Why `replace_content` preserves `id`.** Query expansion produces N variants. Each variant must stay associated with the original query's author metadata so the author filter applies to every variant. `replace_content` copies `id`, `author_id`, `author_full_name`, and `metadata`, changing only `content`. If it created a fresh `Query()` instead, the author filter would be lost on all expanded queries except the first.

**Why `Query` subclasses `VectorBaseDocument`.** `VectorBaseDocument` provides `__eq__` and `__hash__` by `id` (`domain/base/vector.py:24-31`). That is what makes `list(set(...))` dedup work in the retriever — chunks with the same `id` collapse to one.

**`EmbeddedQuery` adds `embedding`.** It is what `EmbeddingDispatcher.dispatch()` returns and what `_search_data_category` hands to Qdrant as `query_vector`.

---

### 3. `prompt_templates.py` — the actual templates

```python
# llm_engineering/application/rag/prompt_templates.py
from langchain.prompts import PromptTemplate

from .base import PromptTemplateFactory


class QueryExpansionTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to generate {expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "expand_to_n": expand_to_n,
            },
        )


class SelfQueryTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the user name or user id.
    Your response should consists of only the extracted user name (e.g., John Doe) or id (e.g. 1345256), nothing else.
    If the user question does not contain any user name or id, you should return the following token: none.

    For example:
    QUESTION 1:
    My name is Paul Iusztin and I want a post about...
    RESPONSE 1:
    Paul Iusztin

    QUESTION 2:
    I want to write a post about...
    RESPONSE 2:
    none

    QUESTION 3:
    My user id is 1345256 and I want to write a post about...
    RESPONSE 3:
    1345256

    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])
```

**Key ideas.**

- `{expand_to_n}` and `{separator}` are `partial_variables`: they are fixed when the chain is built and cannot be overridden at invoke time. Only `{question}` is a live `input_variable`. This means one `QueryExpansionTemplate.create_template(expand_to_n-1)` call produces a reusable prompt.
- The separator is a deliberate sentinel string, `#next-question#`, chosen because the LLM is unlikely to emit it by accident. Parsing splits on it.
- The self-query prompt is **few-shot**: three examples teach the "name / none / id" response grammar. The model is told to emit *only* the name or the literal token `none`. That single token is the contract `SelfQuery.generate` checks.
- Note the prompt literally contains the typo `seperated`. It is harmless and it is the shipped text; do not "fix" it in a code path that string-matches, because nothing does.

**Trade-off:** partial variables make the template immutable at runtime but require rebuilding the `PromptTemplate` when `expand_to_n` changes. The retriever rebuilds it once per `search()` call, which is cheap.

---

### 4. `query_expanison.py` — multi-query generation

Note the file name is misspelled `query_expanison.py` in the repo. Keep that exact path.

```python
# llm_engineering/application/rag/query_expanison.py
import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep
from .prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGStep):
    @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
        assert expand_to_n > 0, f"'expand_to_n' should be greater than 0. Got {expand_to_n}."

        if self._mock:
            return [query for _ in range(expand_to_n)]

        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1)
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)

        chain = prompt | model

        response = chain.invoke({"question": query})
        result = response.content

        queries_content = result.strip().split(query_expansion_template.separator)

        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]

        return queries


if __name__ == "__main__":
    query = Query.from_str("Write an article about the best types of advanced RAG methods.")
    query_expander = QueryExpansion()
    expanded_queries = query_expander.generate(query, expand_to_n=3)
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)
```

**Why query expansion exists.** A single query vector covers a small region of embedding space. If that region is slightly off, the right document is never retrieved. Generating N paraphrases lands N probes in different regions, all still relevant to the intent. The book calls this "overcoming the limitations of distance-based similarity search."

**Why `expand_to_n - 1`.** The prompt asks for N-1 *new* questions; the original is prepended explicitly by `queries = [query]`. So `expand_to_n=3` yields exactly 3 queries (1 original + 2 generated), matching the book's example:

```
1. Write an article about the best types of advanced RAG methods.          (original)
2. What are the most effective advanced RAG methods, and how can they be applied?
3. Can you provide an overview of the top advanced retrieval-augmented generation techniques?
```

**Why `temperature=0`.** Deterministic, reproducible expansion. Higher temperature would diversify more, but at the cost of run-to-run variability. The book's design favors reproducibility.

**Why `Query` is interpolated directly into `{"question": query}`.** `Query` is a Pydantic `BaseModel`; LangChain serializes it (its `str` representation includes the content). The model sees the content plus incidental metadata. It works, but it is a subtle coupling worth knowing when debugging prompt logs.

**`mock` returns N references to the same object.** In dummy mode you get N identical queries; dedup later collapses their results. Fine for plumbing tests.

**Latency cost.** N queries mean N vector searches. The retriever parallelizes them, but each still costs an embedding call plus a Qdrant round-trip. The book warns: "Increasing the number of searches can impact your latency... experiment with the number of queries."

---

### 5. `self_query.py` — metadata extraction for filtering

```python
# llm_engineering/application/rag/self_query.py
import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument
from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate


class SelfQuery(RAGStep):
    @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
        if self._mock:
            return query

        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)

        chain = prompt | model

        response = chain.invoke({"question": query})
        user_full_name = response.content.strip("\n ")

        if user_full_name == "none":
            return query

        first_name, last_name = utils.split_user_full_name(user_full_name)
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

        query.author_id = user.id
        query.author_full_name = user.full_name

        return query


if __name__ == "__main__":
    query = Query.from_str("I am Paul Iusztin. Write an article about the best types of advanced RAG methods.")
    self_query = SelfQuery()
    query = self_query.generate(query)
    logger.info(f"Extracted author_id: {query.author_id}")
    logger.info(f"Extracted author_full_name: {query.author_full_name}")
```

**Why self-query instead of trusting the embedding.** You cannot guarantee an author name leaves enough signal in the embedding vector. Embedding a whole query can bury a rare proper noun. Self-query extracts the name as a *structured* field, then uses it as a hard filter, which removes ambiguity that cosine similarity cannot. The book's analogy: searching "Java" may return the island or the language; a metadata filter disambiguates deterministically.

**What `UserDocument.get_or_create` does.** It looks the user up in MongoDB by `first_name`/`last_name`; if absent it *creates* a new `UserDocument`. This is a real gotcha: a hallucinated or new name in a query creates a row in the warehouse. Because the filter then matches a fresh `id` with no documents, subsequent retrieval returns nothing for that author. In production you would want extraction to only match existing users (see Edge Cases).

**Name splitting.** `utils.split_user_full_name` (in `application/utils/split_user_full_name.py`) handles 1-token and multi-token names:

```python
def split_user_full_name(user: str | None) -> tuple[str, str]:
    if user is None:
        raise ImproperlyConfigured("User name is empty")

    name_tokens = user.split(" ")
    if len(name_tokens) == 0:
        raise ImproperlyConfigured("User name is empty")
    elif len(name_tokens) == 1:
        first_name, last_name = name_tokens[0], name_tokens[0]
    else:
        first_name, last_name = " ".join(name_tokens[:-1]), name_tokens[-1]

    return first_name, last_name
```

A single token ("Madonna") becomes first=last="Madonna". A multi-word first name ("Mary Jane Watson") becomes first="Mary Jane", last="Watson".

**Note on the `none` sentinel.** The prompt must return exactly `none`. `generate` strips newlines/spaces and compares to `"none"`. If the model returns `None`, `"None"`, or `"none."`, the guard misses and `split_user_full_name("none.")` produces a bogus user. This is a genuine brittleness — see Common Pitfalls.

---

### 6. `reranking.py` — the post-retrieval cross-encoder

```python
# llm_engineering/application/rag/reranking.py
import opik

from llm_engineering.application.networks import CrossEncoderModelSingleton
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.domain.queries import Query

from .base import RAGStep


class Reranker(RAGStep):
    def __init__(self, mock: bool = False) -> None:
        super().__init__(mock=mock)

        self._model = CrossEncoderModelSingleton()

    @opik.track(name="Reranker.generate")
    def generate(self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        if self._mock:
            return chunks

        query_doc_tuples = [(query.content, chunk.content) for chunk in chunks]
        scores = self._model(query_doc_tuples)

        scored_query_doc_tuples = list(zip(scores, chunks, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents
```

**Why a cross-encoder.** A bi-encoder (the embedding model) encodes the query and each document *independently*. The cross-encoder runs the query and document through one transformer *together*, so attention can align specific query terms with specific passage spans. That interaction is more accurate and far more expensive: cross-encoders cannot precompute document vectors, so they are O(candidates) forward passes per query. Use them only on a small candidate set, which is exactly what the retriever produces.

**Why `zip(scores, chunks, strict=False)`.** `scores` is a `list[float]` and `chunks` a list of `EmbeddedChunk`. Zipping pairs each score with its chunk. `strict=False` tolerates a length mismatch (e.g., the model returned fewer scores) instead of raising. It is defensive, not a correctness guarantee — a silent mismatch would misalign or truncate results.

**Why `sort(key=lambda x: x[0], reverse=True)`.** Sort by the score (tuple index 0) descending. Python's sort is stable, so ties preserve the original candidate order.

**The cross-encoder model.** `CrossEncoderModelSingleton` (in `networks/embeddings.py`) defaults to `cross-encoder/ms-marco-MiniLM-L-4-v2` and `device=settings.RAG_MODEL_DEVICE` (default `"cpu"`). See Session 4.2 for its internals. Note the score is a raw logit-like value, not a probability — the code never needs it bounded, only ordered.

**Concatenation cost review.**

| Stage | Cost |
|-------|------|
| Bi-encoder retrieval | 1 query embedding + ANN search (ms) |
| Cross-encoder rerank | 1 forward pass per (query, chunk) pair |

Reranking 30 candidates ≈ 30 forward passes; on CPU with MiniLM-L-4 this is tens of milliseconds. Reranking 5,000 candidates is not viable. The candidate set size is bounded by `N × K` from the retrieval stage, which is why `_search` uses `limit=k // 3`.

---

### 7. `retriever.py` — the orchestrator (the real code)

```python
# llm_engineering/application/rag/retriever.py
import concurrent.futures

import opik
from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

from llm_engineering.application import utils
from llm_engineering.application.preprocessing.dispatchers import EmbeddingDispatcher
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from llm_engineering.domain.queries import EmbeddedQuery, Query

from .query_expanison import QueryExpansion
from .reranking import Reranker
from .self_query import SelfQuery


class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)

    @opik.track(name="ContextRetriever.search")
    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3,
    ) -> list:
        query_model = Query.from_str(query)

        query_model = self._metadata_extractor.generate(query_model)
        logger.info(
            f"Successfully extracted the author_full_name = {query_model.author_full_name} from the query.",
        )

        n_generated_queries = self._query_expander.generate(query_model, expand_to_n=expand_to_n_queries)
        logger.info(
            f"Successfully generated {len(n_generated_queries)} search queries.",
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in n_generated_queries]

            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(f"{len(n_k_documents)} documents retrieved successfully")

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        return k_documents

    def _search(self, query: Query, k: int = 3) -> list[EmbeddedChunk]:
        assert k >= 3, "k should be >= 3"

        def _search_data_category(
            data_category_odm: type[EmbeddedChunk], embedded_query: EmbeddedQuery
        ) -> list[EmbeddedChunk]:
            if embedded_query.author_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="author_id",
                            match=MatchValue(
                                value=str(embedded_query.author_id),
                            ),
                        )
                    ]
                )
            else:
                query_filter = None

            return data_category_odm.search(
                query_vector=embedded_query.embedding,
                limit=k // 3,
                query_filter=query_filter,
            )

        embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

        post_chunks = _search_data_category(EmbeddedPostChunk, embedded_query)
        articles_chunks = _search_data_category(EmbeddedArticleChunk, embedded_query)
        repositories_chunks = _search_data_category(EmbeddedRepositoryChunk, embedded_query)

        retrieved_chunks = post_chunks + articles_chunks + repositories_chunks

        return retrieved_chunks

    def rerank(self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)

        reranked_documents = self._reranker.generate(query=query, chunks=chunks, keep_top_k=keep_top_k)

        logger.info(f"{len(reranked_documents)} documents reranked successfully.")

        return reranked_documents
```

**Step-by-step, with the "why".**

1. **`Query.from_str(query)`** strips leading/trailing newlines and spaces. Every downstream step operates on the `Query` object, so author metadata can be attached.
2. **`self._metadata_extractor.generate(query_model)`** mutates and returns the query with `author_id`/`author_full_name` set (or leaves them `None`). This is the filter source.
3. **`self._query_expander.generate(query_model, expand_to_n=...)`** returns a list of `Query` variants, each carrying the author metadata via `replace_content`.
4. **Parallel search.** `ThreadPoolExecutor` submits one `_search` task per variant. `as_completed` yields results in completion order. The work is I/O-bound (Qdrant HTTP + embedding), so threads help even under the GIL.
5. **`utils.misc.flatten`** turns `list[list[EmbeddedChunk]]` into `list[EmbeddedChunk]`. It is defined in `application/utils/misc.py` as a flat list comprehension.
6. **`list(set(n_k_documents))`** dedups by object hash (`id`), collapsing chunks that multiple expanded queries retrieved. **Order is lost** — `set` is unordered, so the pre-rerank order is arbitrary. That is fine because step 7 reorders everything by relevance.
7. **`if len(n_k_documents) > 0`** guards the reranker against an empty list. If no documents were retrieved (e.g., author filter matched nothing), the retriever returns `[]` instead of calling the cross-encoder on nothing.
8. **`self.rerank(query, ...)`** passes the **original string** `query`, not `query_model`. `rerank` re-wraps it with `Query.from_str`. The reranker only uses `query.content`, so metadata is irrelevant here — but it is a subtle asymmetry worth noticing.

**`_search` internals.**

- `assert k >= 3`: because each query fans out to three category searches with `limit = k // 3`, a `k < 3` would make `k // 3 == 0`, and Qdrant would return nothing. The assertion prevents that silent failure.
- The closed-over `_search_data_category` builds a Qdrant `Filter(must=[FieldCondition(key="author_id", match=MatchValue(value=str(...)))])` when the query has an author, else `None`.
- `str(embedded_query.author_id)`: the payload stores `author_id` as a string (see `VectorBaseDocument._uuid_to_str`), so the filter value must be the string form.
- **`EmbeddingDispatcher.dispatch(query)`** embeds the query with the same `EmbeddingModelSingleton` used at ingestion. This is the single most important consistency guarantee in the pipeline: same encoder, same vector space.
- The three categories are queried separately so that a single category cannot monopolize the `k` budget. `limit=k // 3` per category caps the per-query yield at roughly `k`; because three categories × `k//3` ≤ `k`, one query returns ≤ `k` chunks.

**Category collections.** Each `EmbeddedChunk` subclass maps to a Qdrant collection via its `Config.name` (`embedded_posts`, `embedded_articles`, `embedded_repositories`) and a `DataCategory` (`domain/embedded_chunks.py`). Articles and repositories carry extra fields (`link`, `name`); all three carry `author_id` and `author_full_name` in the payload.

---

## 🔬 Deep Dive: Candidate Arithmetic and Latency

### How many candidates reach the reranker?

For `k` and `N` expanded queries, with three categories each searched at `limit = k // 3`:

```
max per query = 3 * (k // 3)   <= k
max before dedup = N * (3 * (k // 3))   <= N * k
after dedup <= N * k
```

Worked example, `k=3`, `N=3` (`expand_to_n=3`):

```
per category limit = 3 // 3 = 1
per query = 1 post + 1 article + 1 repo = 3
N=3 queries -> up to 9 candidates
reranker scores up to 9 pairs, keeps top 3
```

Worked example, `k=6`, `N=3`:

```
per category limit = 6 // 3 = 2
per query = 2 + 2 + 2 = 6
N=3 queries -> up to 18 candidates
reranker scores up to 18 pairs, keeps top 6
```

`k=4`, `N=3`:

```
per category limit = 4 // 3 = 1   (integer division floors)
per query = 3
up to 9 candidates -> keep top 4
```

Integer division matters: `k=4` and `k=5` both yield limit 1. Choose `k` as a multiple of 3 to get the most out of the budget.

### Why dedup is not optional

Different expanded queries often retrieve the same chunk (e.g., a strong article ranks top-1 for every paraphrase). Without dedup the reranker would score the same text repeatedly and the final top-k could contain duplicates, wasting context window on redundant tokens. `set()` by `id` removes them.

### Latency model

```
T_search ≈ T_embed_query + max_over_threads(T_qdrant)     # parallel
T_rerank ≈ C * T_cross_forward                              # C candidates
T_total  ≈ T_selfquery + T_expansion + T_search + T_rerank
```

`T_selfquery` and `T_expansion` are LLM calls (the slowest part at network latency), `T_search` is parallelized, `T_rerank` is CPU/GPU bound. If latency matters more than recall, reduce `expand_to_n_queries` and `k` first — the two LLM calls dominate.

| Knob | Effect on latency | Effect on recall |
|------|-------------------|------------------|
| `expand_to_n_queries` ↑ | +N LLM? no, 1 LLM call; +N searches | higher |
| `k` ↑ | more candidates to rerank | higher |
| Reranker device CPU→GPU | lower rerank time, uses VRAM | none |
| `author_id` filter present | smaller search space | lower if over-filtered |

---

## 🛠️ Hands-On: Exercise the Pipeline

> These snippets use the real classes. Run them from the repo root with the project environment (Qdrant + MongoDB up via `docker compose`, `.env` with `OPENAI_API_KEY`). If you only want plumbing, use `mock=True`.

### Step 1: Mock mode (no API, no models)

```python
from llm_engineering.application.rag.retriever import ContextRetriever

retriever = ContextRetriever(mock=True)
docs = retriever.search("My name is Paul Iusztin. How does RAG work?", k=3, expand_to_n_queries=3)
print(len(docs))   # 0 in mock mode: _search still hits Qdrant, reranker returns chunks unchanged
print(docs)
```

With `mock=True`, `SelfQuery` returns the query untouched (no author), `QueryExpansion` returns three identical queries, and `Reranker.generate` returns candidates unchanged. The Qdrant search still runs, so this is a good test of the vector DB wiring without OpenAI.

### Step 2: Query expansion only

```bash
python -m llm_engineering.application.rag.query_expanison
```

Expected (book, GPT-4o-mini):

```
... | INFO - Write an article about the best types of advanced RAG methods.
... | INFO - What are the most effective advanced RAG methods, and how can they be applied?
... | INFO - Can you provide an overview of the top advanced retrieval-augmented generation techniques?
```

### Step 3: Self-query only

```bash
python -m llm_engineering.application.rag.self_query
```

Expected:

```
... | INFO - Extracted author_id: <user-uuid>
... | INFO - Extracted author_full_name: Paul Iusztin
```

The `author_id` is the MongoDB `_id` of the `UserDocument` for `Paul Iusztin`. If the user does not exist yet, `get_or_create` creates it.

### Step 4: Full retrieval

```python
from loguru import logger
from llm_engineering.application.rag.retriever import ContextRetriever

query = """
        My name is Paul Iusztin.

        Could you draft a LinkedIn post discussing RAG systems?
        I'm particularly interested in:
            - how RAG works
            - how it is integrated with vector DBs and large language models (LLMs).
        """

retriever = ContextRetriever(mock=False)
documents = retriever.search(query, k=3)
for rank, document in enumerate(documents):
    logger.info(f"{rank + 1}: {document}")
```

Expected: three `EmbeddedArticleChunk`-style objects with `author_full_name='Paul Iusztin'` and `metadata={'embedding_model_id': 'sentence-transformers/all-MiniLM-L6-v2', 'embedding_size': 384, 'max_input_length': 256}` (book output).

### Step 5: Inspect a chunk's metadata

```python
doc = documents[0]
print(doc.author_full_name)
print(doc.metadata)        # embedding_model_id, embedding_size, max_input_length
print(type(doc).__name__)  # EmbeddedArticleChunk / EmbeddedPostChunk / EmbeddedRepositoryChunk
print(doc.platform)
```

The metadata is what lets you build a references list for the user (links, platform), which the book calls out as a trust booster.

---

## 📝 Exercise: Add a Category Router

### Task

Today `_search` always queries all three categories. Add a router that predicts which categories are relevant and queries only those, cutting Qdrant calls from three to one or two.

### Why this is worth doing

The book's second suggested improvement: "add a router between the query and the search... a multi-category classifier that predicts the data categories we must retrieve." For a theoretical paragraph you likely only need articles; for a code illustration you likely need articles *and* repositories.

### Template

```python
# llm_engineering/application/rag/router.py  (new file)
from llm_engineering.domain.queries import Query


class QueryRouter:
    """Predict which EmbeddedChunk categories to search for a query."""

    def route(self, query: Query) -> list[str]:
        # return a subset of {"posts", "articles", "repositories"}
        raise NotImplementedError
```

Then in a `ContextRetriever` subclass, replace the fixed three calls in `_search` with a loop over the routed categories, preserving `limit = k // 3` per category (or rebalancing the budget across the chosen categories).

### Acceptance criteria

1. `mock=True` returns the same as before (all categories).
2. A query mentioning "article" and "paragraph" routes to `articles` only.
3. A query mentioning "code snippet" routes to `articles` + `repositories`.
4. Retrieval still returns `<= k` documents and reranking still keeps top `k`.
5. You recorded latency before/after on a fixed query set.

**Hint:** a zero-shot `ChatOpenAI` prompt that returns a JSON list, or an embedding-similarity classifier over category descriptions. Reuse `RAGStep` and `PromptTemplateFactory` so it plugs into the existing abstractions.

---

## 🐛 Common Pitfalls

- **`k` must be ≥ 3.** `_search` asserts it. `k < 3` makes `limit = k // 3 == 0`, and Qdrant returns nothing. If you want small `k`, change the per-category budget, not `k`.
- **`k=4` and `k=5` waste budget.** Integer division means they retrieve the same as `k=3`. Prefer multiples of 3.
- **`set()` dedup loses order.** Do not rely on pre-rerank ordering. If you need stable order before reranking, dedup with a dict keyed by `id` (as the ingestion side does) instead of `set`.
- **Self-query can create spurious users.** `UserDocument.get_or_create` inserts a row for any name the LLM extracts. A hallucinated name yields an `author_id` with zero documents, so retrieval returns `[]`. In production, look up existing users only, or gate the filter behind a match.
- **The `none` sentinel is exact-match fragile.** The guard is `if user_full_name == "none"`. `"None"`, `"none."`, or a blank string slips through into `split_user_full_name`. Harden with `.lower().strip(" .")` and an allowlist.
- **Author filter is a string.** The Qdrant filter uses `value=str(embedded_query.author_id)`. Passing the raw `UUID` will not match the string payload.
- **`mock=True` still queries Qdrant.** Mock mode skips LLM calls and reranking, not the vector search. Set up the DB or expect logged search failures.
- **Cross-encoder on CPU by default.** `RAG_MODEL_DEVICE="cpu"`. Reranking a large candidate set is slow on CPU; keep the candidate set small or move to GPU deliberately (and watch VRAM while training runs).
- **The two LLM calls dominate latency.** Self-query and expansion are sequential network calls before any search. Cache or batch them if latency is critical.
- **`opik.track` decorators execute on import of the class.** They are monitoring hooks (Session 7.2); if Opik is misconfigured, tracing may warn but should not break retrieval. Verify when you change monitoring.
- **Same embedding model at both ends.** If `TEXT_EMBEDDING_MODEL_ID` changes without re-indexing, query vectors and stored vectors live in different spaces and every result is garbage. See Session 4.2.

---

## 🎓 Knowledge Check

1. **Why does the retrieval module contain most of the RAG engineering?**
   Because generation is a single LLM call, while retrieval must wrangle data to make the context relevant. The book: "At the retrieval step... you write most of the RAG inference code."

2. **What problem does query expansion solve?**
   A single query vector covers a small region of embedding space; paraphrases probe multiple regions and raise the chance of finding relevant documents.

3. **Why is `expand_to_n - 1` passed to the template?**
   The original query is prepended explicitly, so the LLM only needs to generate the remaining N-1 variants. `expand_to_n=3` yields 3 queries total.

4. **What is self-querying and why filter on `author_id`?**
   It extracts structured metadata (the author name/ID) from the query so retrieval can hard-filter on it, which embeddings alone cannot guarantee.

5. **What does `Query.replace_content` preserve, and why?**
   It preserves `id`, `author_id`, `author_full_name`, and `metadata`, so every expanded query still carries the author filter.

6. **Why does `_search` assert `k >= 3`?**
   Each query fans out to three categories at `limit=k // 3`; `k < 3` gives limit 0 and empty results.

7. **How many candidates can reach the reranker for `k=6, expand_to_n=3`?**
   At most 18 before dedup (3 queries × 3 categories × 2), fewer after dedup.

8. **Why is the search parallelized with `ThreadPoolExecutor`?**
   Each search is I/O-bound (embedding + Qdrant HTTP), so concurrent tasks reduce wall-clock latency.

9. **Why does `list(set(...))` dedup work on chunks?**
   `VectorBaseDocument` defines `__eq__` and `__hash__` by `id`, so equal-id chunks collapse.

10. **What does the cross-encoder do that the bi-encoder cannot?**
    It runs the query and document through one transformer together, so attention aligns query terms with document spans. Bi-encoders encode them independently.

11. **Why can't you rerank the whole corpus with a cross-encoder?**
    It needs one forward pass per (query, document) pair; cost grows linearly with candidates and cannot be precomputed.

12. **What does `Reranker.generate` return in mock mode?**
    The input chunks unchanged (truncation is not applied).

13. **What does `EmbeddingDispatcher.dispatch(query)` guarantee?**
    The query is embedded with the same `EmbeddingModelSingleton` used at ingestion, keeping both in one vector space.

14. **Why is the code's `set` dedup acceptable despite losing order?**
    Because the reranker reorders all candidates by relevance immediately afterward.

15. **Name one proposed improvement to the retrieval step from the book.**
    A category router (query only the relevant collections), hybrid search with BM25, or multi-index embeddings.

---

## 📖 Glossary

- **Advanced RAG** — optimizations to vanilla RAG at the pre-retrieval, retrieval, and post-retrieval stages.
- **Bi-encoder** — a model that encodes texts independently into vectors; used for fast first-pass retrieval.
- **Cross-encoder** — a model that scores a (query, document) pair jointly; accurate but slow; used for reranking.
- **Query expansion / multi-query** — generating N paraphrases of a query to probe more embedding space.
- **Self-query** — extracting structured metadata from a query (here, author) to use as a vector-search filter.
- **Filtered vector search** — ANN search constrained by a payload filter (e.g., `author_id`).
- **Reranking** — reordering retrieved candidates by a cross-encoder relevance score and keeping the top K.
- **Candidate set** — the N × K chunks gathered before reranking.
- **ANN** — approximate nearest neighbor search; trades exactness for speed in vector DBs.
- **`RAGStep`** — the ABC every advanced retrieval step implements (`generate`, `mock`).
- **`PromptTemplateFactory`** — the ABC that standardizes `create_template()` returning a LangChain `PromptTemplate`.
- **`EmbeddingDispatcher`** — the dispatcher that routes a domain object to the correct embedding handler and encoder.
- **Opik** — experiment/monitoring tool used by the `@opik.track` decorators (Session 7.2).

---

## 🔗 Next Session

**Session [4.2](session_4.2_embedding_models.md): Embedding Models & Cross-Encoders**

We open up `EmbeddingModelSingleton` and `CrossEncoderModelSingleton`: how `SentenceTransformer.encode` works (mean pooling + normalization), why the embedding dimension drives Qdrant collection sizing, and how to benchmark and swap models.

Related sessions:

- [Session 2.3](session_2.3_feature_engineering.md) — the RAG ingestion pipeline that populates the vector DB.
- [Session 6.2](session_6.2_rag_inference_flow.md) — prompt building (`EmbeddedChunk.to_context`) and the `rag()` function.
- [Session 7.4](session_7.4_rag_evaluation.md) — evaluating retrieval quality.

---

## 📚 Additional Resources

- [LangChain MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)
- [LangChain self-querying retrieval](https://python.langchain.com/docs/how_to/self_query/)
- [Qdrant filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [SentenceTransformers cross-encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Superlinked multi-indexing](https://superlinked.com/vectorhub/articles/real-time-retrieval-system-social-media-data)
- Source: `llm_engineering/application/rag/retriever.py`, `query_expanison.py`, `self_query.py`, `reranking.py`, `prompt_templates.py`, `base.py`.

---

**Estimated Time**: 4-5 hours

**Prerequisites**: [Sessions 1.1-1.3](../README.md), [2.1](session_2.1_web_crawling.md)-[2.3](session_2.3_feature_engineering.md)

**Outcome**: You can read and modify the entire retrieval module, explain every stage and its trade-offs, size the candidate set for a given `k` and `expand_to_n`, and extend the pipeline with a router or hybrid search.
