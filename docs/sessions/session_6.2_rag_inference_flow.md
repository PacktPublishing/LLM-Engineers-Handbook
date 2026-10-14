# Session 6.2: RAG Inference Flow

## 🎯 Learning Objectives

By the end of this session, you will:
- Trace a user query through the complete RAG inference path
- Understand how retrieved chunks become LLM context via `to_context`
- Read the content-creator prompt used for generation
- Inspect the token accounting and Opik trace metadata
- Run retrieval-only and full-generation harnesses
- Reason about the `N x K` candidate budget and why the retriever divides `k` by three
- Explain the tradeoffs of query expansion, self-querying, and reranking
- Diagnose the most common RAG failure modes from trace evidence
- Extend the flow with citations, a router, and hybrid search

---

## ✅ Prerequisites

- **Session 4.1 (Advanced RAG)** — the theory of query expansion, self-querying, filtered
  vector search, and reranking.
- **Session 4.2 (Embedding Models)** — the embedding model and vector dimension.
- **Session 6.1 (FastAPI API)** — the business microservice that exposes `/rag`.
- You should have run the **feature pipeline** at least once so the Qdrant collections
  (`embedded_posts`, `embedded_articles`, `embedded_repositories`) are populated.
- MongoDB running with a `users` collection (self-query creates a `UserDocument`).
- `.env` with `OPENAI_API_KEY` (query expansion and self-query both call OpenAI) and
  `COMET_API_KEY` (optional, for tracing).

---

## 🏗️ Architecture Overview

The RAG inference pipeline has three responsibilities: **retrieve**, **augment**, and
**generate**. The book splits them into two modules: a retrieval module
(`ContextRetriever`) and an inference service (`rag` + `call_llm_service`). Augmenting
the prompt is deliberately *not* a whole module — that would be overengineering.

```
User query: "My name is Paul Iusztin. Could you draft a LinkedIn post about RAG?"
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ContextRetriever.search(query, k=3)                                   │
│                                                                       │
│  1. SelfQuery        → author_id = Paul's user id                     │
│  2. QueryExpansion   → 3 query variants (temperature=0)               │
│  3. ThreadPool search → per variant, per collection (k//3 each)       │
│  4. dedupe (hash by id)                                               │
│  5. Reranker (cross-encoder) → keep_top_k = k                         │
└──────────────────────────────────────────────────────────────────────┘
        │  list[EmbeddedChunk]
        ▼
EmbeddedChunk.to_context(documents)
        │  a single formatted string
        ▼
call_llm_service(query, context)
        │  InferenceExecutor builds the content-creator prompt
        ▼
LLMInferenceSagemakerEndpoint.inference()
        │
        ▼
answer  →  Response + Opik trace metadata
```

### The eight steps in the book's flow

The book (Chapter 9) describes the pipeline as eight steps. Mapping each to the code:

| # | Book step | Code location | Notes |
|---|-----------|---------------|-------|
| 1 | User query | `rag_endpoint` → `rag(query)` | Request body is `{"query": "..."}` |
| 2 | Query expansion | `QueryExpansion.generate` | `expand_to_n - 1` new queries + original |
| 3 | Self-querying | `SelfQuery.generate` | Extracts `author_id` / `author_full_name` |
| 4 | Filtered vector search | `ContextRetriever._search` | Qdrant `Filter(must=[FieldCondition(author_id)])` |
| 5 | Collecting results | `_search` returns 3 lists, concatenated | `k // 3` per collection |
| 6 | Reranking | `ContextRetriever.rerank` → `Reranker.generate` | Cross-encoder scores, keeps top `k` |
| 7 | Build prompt + call LLM | `to_context` → `InferenceExecutor.execute` | Content-creator prompt |
| 8 | Answer | `rag` returns string, endpoint wraps it | `QueryResponse(answer=...)` |

### Module boundaries (why two microservices)

The book makes a point that separates **LLM microservice** from **business microservice**.
The LLM microservice is narrow: it takes a fully-formed prompt and returns text. The
business microservice owns retrieval, prompt building, and monitoring. That split is why
`call_llm_service` exists as its own function: it is the seam between the two services and
the natural place to trace the exact prompt sent to the model.

```
┌────────────────────────────┐        HTTP         ┌──────────────────────────┐
│  Business microservice      │ ─────────────────▶  │  LLM microservice         │
│  (FastAPI, this session)    │   prompt in body    │  (SageMaker endpoint)     │
│  - retrieval                │ ◀─────────────────  │  - generate text          │
│  - prompt augmentation      │   generated_text    │  - no retrieval knowledge │
│  - Opik tracing             │                     │  - no context awareness   │
└────────────────────────────┘                     └──────────────────────────┘
```

---

## 📁 Key Files Explained

### 1. `retriever.py` - Multi-Stage Retrieval

```python
# llm_engineering/application/rag/retriever.py
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
```

**Key Concepts**:
- **`Query.from_str`** normalizes the raw string into a `Query` value object (strips
  leading/trailing whitespace and newlines).
- **`SelfQuery`** optionally extracts an author and attaches `author_id`/`author_full_name`
  for metadata filtering. If no name is present it returns the query unchanged.
- **`QueryExpansion` generates `expand_to_n_queries` variants** (including the original),
  increasing recall across the embedding space.
- **`ThreadPoolExecutor`** runs each variant's search concurrently, then `flatten` + `set`
  de-duplicate by identity (`VectorBaseDocument.__hash__` is `hash(self.id)`).
- **Reranking only happens when candidates exist**, and `keep_top_k=k` returns exactly the
  final context size (or fewer if fewer candidates were retrieved).

#### Why the thread pool matters

Query expansion multiplies the number of vector searches by `N`. Without parallelism,
latency scales linearly with `N`; with `ThreadPoolExecutor`, the searches run concurrently
and latency is bounded by the slowest single search (plus overhead). Qdrant I/O is
network/disk-bound, so threads help even though Python has a GIL. The book explicitly calls
this out: expanding to more queries increases latency, and parallelizing "drastically
reduc[es]" it.

#### Why `k // 3` per collection

There are three data categories: posts, articles, repositories. Each `_search` call
queries all three collections and requests `k // 3` from each. The sum is `<= k` per variant
(not equal, because a filter can yield fewer, or a collection can be empty). This is the
book's `K` distributed evenly so no single source dominates the candidate pool. It is also
why `_search` asserts `k >= 3`: with `k < 3`, `k // 3 == 0` and no documents would be
requested at all.

### The per-variant search

```python
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
```

**Key Concepts**:
- **Queries are embedded on the fly** with the same `EmbeddingDispatcher` used for indexing,
  producing an `EmbeddedQuery` (never persisted). Using the same dispatcher guarantees the
  same embedding model at ingestion and query time — a correctness requirement for retrieval.
- **`k // 3` per collection** distributes the candidate budget evenly across posts, articles,
  and repositories, so no single source dominates.
- **Metadata filtering** builds a Qdrant `Filter` on `author_id` when SelfQuery found a user,
  restricting results to that person's content. This is the self-query feature in action.
- **`EmbeddingDispatcher.dispatch(query)`** returns an `EmbeddedQuery`, which is a `Query`
  plus an `embedding` field. The filter and the vector are combined in a single Qdrant call.

#### The `_search_data_category` closure

`_search_data_category` is a nested function (a closure) so it can capture `k` without an
extra argument. It is generic over the ODM class: the same function body queries
`EmbeddedPostChunk`, `EmbeddedArticleChunk`, or `EmbeddedRepositoryChunk`. Each class defines
`Config.name` and `Config.category`, so `search` can be called polymorphically.

### Candidate budget worked example

```
k = 3,  expand_to_n_queries = 3

Per variant, per collection:   k // 3 = 1
Per variant (3 collections):   1 + 1 + 1 = 3   (<= k)
Across N = 3 variants:         3 * 3 = 9        (<= N x K)
After dedupe:                  <= 9 unique chunks
After rerank(keep_top_k=3):    exactly 3 (if >= 3 candidates exist)
```

Larger `k` raises recall but costs tokens. Larger `expand_to_n_queries` widens the search
but costs OpenAI calls and latency.

---

### 2. `EmbeddedChunk.to_context` - Context Rendering

```python
# llm_engineering/domain/embedded_chunks.py
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
```

**Key Concepts**:
- Each chunk is rendered with its **type, platform, author, and content**. Grouping content
  blocks like this makes it easy for the model to attribute a fact to a source.
- The **1-indexed numbering** appears in traces, so you can see which chunk informed the
  answer.
- `chunk.__class__.__name__` (for example `EmbeddedArticleChunk`) tells the model whether the
  source was a post, article, or repository. Note it is the *Python class name*, not the
  Qdrant collection name.
- The entity exposes far more metadata (`document_id`, `author_id`, `metadata`, and on
  subclasses `link`/`name`) than `to_context` renders. That extra metadata is exactly what the
  citations exercise below surfaces.

#### Rendering worked example

For two chunks, `to_context` returns (whitespace abbreviated):

```
            Chunk 1:
            Type: EmbeddedArticleChunk
            Platform: decodingml.substack.com
            Author: Paul Iusztin
            Content: 4 Advanced RAG Algorithms You Must Know...

            Chunk 2:
            Type: EmbeddedPostChunk
            Platform: linkedin.com
            Author: Paul Iusztin
            Content: ...
```

The model sees a flat list of labeled blocks. Because the blocks are separated by blank
lines and labeled, the generator can cite "Chunk 1" if you ask it to.

---

### 3. `InferenceExecutor` - The Generation Prompt

```python
# llm_engineering/model/inference/run.py
class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        context: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}
            """
        else:
            self.prompt = prompt

    def execute(self) -> str:
        self.llm.set_payload(
            inputs=self.prompt.format(query=self.query, context=self.context),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
            },
        )
        answer = self.llm.inference()[0]["generated_text"]

        return answer
```

**Key Concepts**:
- **"Use the provided context as the primary source of information"** instructs grounding.
  This is what makes the output a RAG result rather than free-form generation. The phrase
  "primary source" (not "the only source") leaves the model room to use parametric knowledge
  as a fallback — a deliberate, debatable choice you should be aware of.
- **The exact `{query}` / `{context}` placeholders** are resolved with `str.format`. Curly
  braces in retrieved content would break `format`; the current pipeline does not sanitize
  them, which is a known fragility to watch.
- **Generation parameters**: `max_new_tokens` (150 by default) bounds cost and latency;
  `repetition_penalty=1.1` reduces loops; `temperature=0.01` keeps it faithful. Note
  `TOP_P_INFERENCE` (0.9) lives in the endpoint's default payload but is *not* overridden here
  — it survives because `set_payload` only updates the keys it is given.
- **`self.llm.inference()[0]["generated_text"]`** assumes the SageMaker HF response is a
  single-element list of dicts. This is the HF text-generation response shape.

#### The empty-context case

`context` defaults to `""` when `None` or empty. So calling `InferenceExecutor(llm, query)`
with no context yields a prompt with an empty `Context:` section — a pure LLM call. This is
the "use the LLM without RAG" path the book mentions, and it is useful for A/B comparison.

---

### 4. The Full RAG Function

```python
# llm_engineering/infrastructure/inference_pipeline_api.py
@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()

    return answer


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Key Concepts**:
- **The flow is three lines**: retrieve, render, generate. All the complexity is hidden
  behind the retriever and executor.
- **`k=3`** is the final context size after reranking. Larger `k` improves recall but costs
  tokens and can add noise.
- **`opik_context.update_current_trace`** attaches the model ids, temperature, and **token
  counts for query, context, and answer**. This is the observability payload you inspect in
  Session 7.2.
- **`misc.compute_num_tokens`** tokenizes with the `HF_MODEL_ID` tokenizer
  (`AutoTokenizer.from_pretrained(settings.HF_MODEL_ID)`), so the counts reflect the
  generator's actual tokenizer — not OpenAI's and not the embedding model's.
- **Errors become HTTP 500** with the exception string in `detail`. That leaks internal error
  text to callers; acceptable for an internal service, a concern for a public one.

> **Book vs repo note.** The book's Chapter 9 listing shows `retriever.search(query, k=3 * 3)`
> (k=9) and `get_current_trace()`. The repo is newer and is the source of truth: it uses
> `k=3` and `opik_context.update_current_trace`. Always trust the repo for code.

---

### 5. `tools/rag.py` - Retrieval-Only Harness

```python
# tools/rag.py
from langchain.globals import set_verbose
from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.infrastructure.opik_utils import configure_opik

if __name__ == "__main__":
    configure_opik()
    set_verbose(True)

    query = """
        My name is Paul Iusztin.

        Could you draft a LinkedIn post discussing RAG systems?
        I'm particularly interested in:
            - how RAG works
            - how it is integrated with vector DBs and large language models (LLMs).
        """

    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=9)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document}")
```

**Key Concepts**:
- **No LLM call**: this harness exercises only retrieval, which is ideal for debugging
  retrieval quality without incurring generation cost.
- **`set_verbose(True)`** makes LangChain log its chains, useful while debugging prompt
  construction (query expansion and self-query are LangChain `prompt | model` chains).
- **`k=9`** retrieves a wider candidate set for manual inspection. Recall that `_search` will
  request `9 // 3 = 3` per collection per variant.
- **`configure_opik()`** is called explicitly here because this script does not import the
  FastAPI app (which configures Opik at import time).

---

### 6. The Advanced-RAG building blocks (supporting files)

These are the pieces `ContextRetriever` composes. You met them in Session 4.1; here is the
exact code this flow relies on.

```python
# llm_engineering/application/rag/query_expanison.py   (note the repo's filename spelling)
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
```

```python
# llm_engineering/application/rag/self_query.py
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
```

```python
# llm_engineering/application/rag/reranking.py
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

**Key Concepts**:
- **`RAGStep`** is a tiny ABC: a constructor storing `self._mock` and an abstract `generate`.
  The mock flag lets you run the whole pipeline offline (no OpenAI, no cross-encoder) — it
  substitutes copies of the original query and returns chunks unsorted.
- **Query expansion** asks OpenAI for `expand_to_n - 1` alternatives, uses a sentinel
  separator `"#next-question#"`, and always prepends the original query. The original keeps
  its `id`; variants reuse it via `replace_content`. Curly braces in a query reaching the
  expansion template would break `PromptTemplate`, but LangChain escapes the input value —
  unlike the `str.format` in `InferenceExecutor`.
- **Self-query** writes a `UserDocument` to MongoDB if the extracted name is new
  (`get_or_create`). This is a **write during a read path** — a subtle side effect worth
  knowing before you put the endpoint under load.
- **Reranker** scores `(query, chunk.content)` pairs with a cross-encoder and sorts
  descending. It deliberately uses only `content` (ignoring platform/author) for scoring; the
  filter already narrowed by author.

#### The query-expansion template

```python
# llm_engineering/application/rag/prompt_templates.py
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
```

`expand_to_n` and `separator` are `partial_variables` (fixed when the template is built).
Only `{question}` varies per call. If the model returns a separator inside a sentence, the
split will still work; if it returns *fewer* alternatives than requested, you get fewer
queries (the code tolerates this).

---

## 🔬 Deep Dive: Token Budget Management

```
query tokens  ─┐
context tokens ─┼──► must fit inside  MAX_INPUT_LENGTH  (settings)
prompt tokens  ─┘                      + MAX_NEW_TOKENS_INFERENCE <= MAX_TOTAL_TOKENS
```

| Setting | Default | Role |
|---------|---------|------|
| `MAX_INPUT_LENGTH` | 2048 | Max prompt length the endpoint accepts |
| `MAX_TOTAL_TOKENS` | 4096 | Input + output ceiling |
| `MAX_NEW_TOKENS_INFERENCE` | 150 | Generated tokens per answer |
| `TEMPERATURE_INFERENCE` | 0.01 | Near-deterministic generation |
| `TOP_P_INFERENCE` | 0.9 | Nucleus sampling, from the endpoint default payload |

If `k` is large and chunks are long, the context can exceed `MAX_INPUT_LENGTH`, and the
endpoint truncates from the front (or errors). The Opik trace's `context_tokens` is the
number to watch. Raise `MAX_INPUT_LENGTH` or reduce `k` when you approach the limit.

The arithmetic: `prompt_tokens + MAX_NEW_TOKENS_INFERENCE <= MAX_TOTAL_TOKENS`. With the
defaults, the prompt must stay under `4096 - 150 = 3946` tokens, comfortably above the
2048 `MAX_INPUT_LENGTH` guard. `MAX_INPUT_LENGTH` is the binding constraint in practice.

---

## 📊 The Retrieval Math, End to End

Let `N = expand_to_n_queries`, `K = k`.

| Quantity | Formula | Example (N=3, K=3) |
|----------|---------|--------------------|
| Collections queried per variant | 3 | 3 |
| Per-collection limit | `K // 3` | 1 |
| Candidates per variant | `<= K` | <= 3 |
| Raw candidates across variants | `<= N * K` | <= 9 |
| After dedupe | `<= N * K` | <= 9 |
| Final context | `min(keep_top_k, candidates)` | 3 |

Deduplication is exact by `id`: `list(set(n_k_documents))` relies on
`VectorBaseDocument.__hash__ = hash(self.id)` and `__eq__` comparing ids. Two chunks with
identical text but different ids are *not* deduplicated — only the same stored point is.

---

## 🛠️ Hands-On: Trace One Query End to End

### Step 1: Retrieval only

```bash
python -m tools.rag
```

Check the top-ranked documents. Are they on topic and (if you named an author) filtered to
that author? The log prints `author_full_name`, the number of generated queries, the number
of documents retrieved, and the rerank count.

### Step 2: Full RAG via the API

```bash
python -m tools.ml_service
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" \
  -d "{\"query\": \"My name is Paul Iusztin. Draft a LinkedIn post about RAG.\"}"
```

### Step 3: Inspect the Opik trace

Open the Opik project (`COMET_PROJECT`) and find the `rag` trace. Verify:
- A `SelfQuery.generate` span with the extracted author.
- A `QueryExpansion.generate` span with the variants.
- A `ContextRetriever.search` span with the candidate count.
- A `Reranker.generate` span.
- The `rag` trace metadata with `query_tokens`, `context_tokens`, `answer_tokens`.

### Step 4: Compare `k` values

Run with `k=1`, `k=3`, and `k=9` and compare answer quality and `context_tokens`. Note:
`k=1` and `k=2` trip the `assert k >= 3` in `_search` when called through `search`
(the `k // 3` per-collection rule), so compare `k=3`, `k=6`, and `k=9` instead.

### Step 5: Toggle author filtering

Run the same query with and without "My name is Paul Iusztin". With a name, `_search`
applies a Qdrant `author_id` filter; without it, `query_filter=None`. Compare candidate
counts in the trace — filtering should reduce them.

---

## 📝 Exercise: Add Source Citations

### Task

Return the source categories alongside the answer.

1. Change `rag()` to also return the list of distinct `platform` values from the retrieved chunks.
2. Extend `QueryResponse` with `sources: list[str]`.
3. Verify the API returns citations.

```python
def rag(query: str) -> tuple[str, list[str]]:
    ...
    sources = sorted({doc.platform for doc in documents})
    return answer, sources
```

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer, sources = rag(query=request.query)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Goal**: Turn retrieval provenance into a user-visible feature. This is the first step
toward trustworthy, auditable RAG.

### Stretch: richer provenance

Instead of only `platform`, return structured citations built from the chunk metadata:
`{"type": doc.__class__.__name__, "platform": doc.platform, "link": getattr(doc, "link", None)}`.
Sort by chunk order (not alphabetically) so the citation order matches `to_context`'s
numbering.

---

## 🧪 Second Exercise: Add a Retrieval Router

The book proposes a router that predicts which collections to query, so a theoretical query
hits only articles instead of all three.

### Task

1. Add a `router(query) -> set[type[EmbeddedChunk]]` classifier. Start with a mock keyword
   heuristic, then an LLM variant.
2. Modify `_search` to query only the selected categories.
3. When the router selects one category, use the full `k` for it (not `k // 3`), otherwise
   the limit stays `k // 3` per selected category.

```python
def _search(self, query, k=3):
    assert k >= 3
    embedded_query = EmbeddingDispatcher.dispatch(query)
    categories = self._route(query)  # returns a subset of the three classes
    per_collection = k if len(categories) == 1 else k // 3
    chunks = []
    for category in categories:
        chunks += _search_one(category, embedded_query, per_collection)
    return chunks
```

**Verification**: trace both a "write code" query (predicts articles + repositories) and a
"write theory" query (predicts articles only) and confirm the number of Qdrant searches drops.
Compare `context_tokens` and latency in Opik.

**Goal**: Learn to trade recall breadth for precision and latency, the central retrieval
tradeoff.

---

## 🐛 Common Pitfalls

- **Empty retrieval**: if collections are empty or the author filter matches nothing,
  `to_context([])` returns `""` and the model answers from parametric memory. Check that
  feature engineering ran and the author name is correct. `context_tokens == 0` in the trace
  is the tell.
- **`str.format` on braces**: retrieved content containing `{` or `}` will raise in
  `prompt.format(...)`. Escape retrieval text or switch to a safer template. (Query expansion
  is safe because LangChain escapes values; the executor is not.)
- **Context overflow**: with large `k`, watch `context_tokens` against `MAX_INPUT_LENGTH`.
- **Author filter too strict**: SelfQuery filters on `author_id`; a wrong extraction yields
  zero results. The SelfQuery prompt returns `none` when no name is present, which correctly
  disables filtering. But a *partial* name ("Paul") creates a `UserDocument(first_name="Paul",
  last_name="Paul")` via `split_user_full_name` and filters to a user that may not exist.
- **Reranker cold-start**: the first call downloads the cross-encoder, adding latency to the
  first request. `CrossEncoderModelSingleton` caches it for later calls.
- **`assert` stripped under `-O`**: `_search`'s `assert k >= 3` disappears if Python runs
  with `-O`. Do not rely on it for validation in production.
- **Self-query writes to Mongo on the read path**: heavy concurrent traffic can create users
  and contend on the `users` collection. Consider caching or pre-creating users.
- **`answer` assumed non-empty**: `inference()[0]["generated_text"]` raises `IndexError` /
  `KeyError` if the endpoint returns an unexpected shape (for example an error payload with a
  200 status).
- **Thread saturation**: `ThreadPoolExecutor()` with no `max_workers` defaults once an
  unbounded number of variants is requested. With large `N`, bound the pool.

---

## 🧭 Edge Cases and Failure Modes

| Scenario | What happens | Signal to watch |
|----------|--------------|-----------------|
| Query has a name, no such user's chunks | `Filter` matches nothing | `context_tokens` low/0, candidate count 0 |
| Query has "none" name | No filter, searches all authors | Full candidate pool |
| All variants identical (mock mode) | Same chunks returned, deduped to one set | Candidate count == one variant's |
| One collection empty | That category contributes 0 | Total `< N * K` |
| Retrieved chunk contains `{` | `str.format` raises | HTTP 500 with `KeyError`/`IndexError` |
| Context exceeds `MAX_INPUT_LENGTH` | Endpoint truncates/errors | `context_tokens` > 2048 |
| Cross-encoder returns ties | `sort` is stable, original order preserved | Ordering of equal scores |
| Duplicate text, different ids | Both survive dedupe | Extra near-identical chunks |
| Endpoint returns error JSON | `inference()[0]` raises | HTTP 500 |

---

## 🎓 Knowledge Check

1. **What three stages make up `ContextRetriever.search`?**
   - Answer: self-query metadata extraction, query expansion, and parallel multi-collection
     search followed by reranking.

2. **How many candidates does each collection return per query variant?**
   - Answer: `k // 3`.

3. **How are duplicates removed across variants?**
   - Answer: By converting to a `set`, since documents hash and compare by `id`.

4. **What does `to_context` include per chunk?**
   - Answer: index, type (class name), platform, author, and content.

5. **Which prompt makes the generation grounded?**
   - Answer: The content-creator prompt instructing the model to use the context as the
     primary source.

6. **Which metadata does the `rag` trace record?**
   - Answer: model id, embedding model id, temperature, and query/context/answer token counts.

7. **Why does `_search` assert `k >= 3`?**
   - Answer: Because each `_search` call divides `k` across three collections; with `k < 3`,
     `k // 3 == 0` and nothing is requested.

8. **Why must the same `EmbeddingDispatcher` be used at ingestion and query time?**
   - Answer: To guarantee the same embedding model and vector space, or similarity search is
     meaningless.

9. **What is the difference between the query-expansion template and the executor prompt on
   the subject of curly braces?**
   - Answer: LangChain escapes values passed to `PromptTemplate`, but `InferenceExecutor`
     uses raw `str.format`, which raises on unescaped braces in retrieved content.

10. **What side effect does SelfQuery have beyond reading?**
    - Answer: It calls `UserDocument.get_or_create`, which may write a new user to MongoDB on
      the read path.

11. **How is the candidate budget computed for N variants and K final chunks?**
    - Answer: up to `N * K` raw candidates, deduped, then reranked down to `K`.

12. **Which token count tells you retrieval returned nothing useful?**
    - Answer: `context_tokens == 0`.

13. **Why is the reranker's first call slow?**
    - Answer: The cross-encoder model is downloaded and loaded on first use, then cached by
      the singleton.

14. **What happens if a retrieved chunk has the same text but a different id as another?**
    - Answer: Both are retained; dedupe is by id, not content.

15. **Where in the request lifecycle would you bound concurrency for many variants?**
    - Answer: In the `ThreadPoolExecutor(max_workers=...)` inside `search`, before the
      per-variant searches are submitted.

---

## 📖 Glossary

- **RAG** — Retrieval-Augmented Generation: retrieve context, put it in the prompt, generate.
- **Query expansion** — generating several paraphrases of the user query to widen recall.
- **Self-query** — extracting metadata (here, author) from the query to use as a filter.
- **Filtered vector search** — a similarity search constrained by metadata filters.
- **Reranking** — re-scoring candidates with a stronger model (cross-encoder) before selecting.
- **Cross-encoder** — a model that scores a query/document pair jointly, more accurate than
  comparing independent embeddings.
- **EmbeddedChunk** — the domain entity stored in Qdrant: content + embedding + metadata.
- **EmbeddedQuery** — a `Query` plus an embedding, built at request time and not persisted.
- **ODM** — Object-Document Mapper: maps domain classes to collections
  (`VectorBaseDocument`, `NoSQLBaseDocument`).
- **N x K** — the candidate pool before reranking: N expanded queries times K chunks.
- **Grounding** — constraining generation to retrieved evidence.
- **Parametric memory** — knowledge stored in the model weights; the fallback when context is
  empty.

---

## 🔗 Next Session

**Session 7.1**: Experiment Tracking with Comet ML

We connect training metrics and RAG traces into dashboards.

Related reading in this repo:
- [Session 4.1: Advanced RAG](session_4.1_advanced_rag.md)
- [Session 4.2: Embedding Models](session_4.2_embedding_models.md)
- [Session 6.1: FastAPI API](session_6.1_fastapi_api.md)
- [Session 7.2: Opik Monitoring](session_7.2_opik_monitoring.md)

---

## 📚 Additional Resources

- [Qdrant Filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [LangChain MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)
- [LangChain self-querying retrieval](https://python.langchain.com/docs/how_to/self_query/)
- [sentence-transformers CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html)
- [Opik Tracing](https://www.comet.com/docs/opik/tracing)

---

## 📑 References

- Iusztin, P. & Labonne, M. *LLM Engineer's Handbook.* Packt, 2024. **Chapter 9, "RAG
  Inference Pipeline"** (pp. 346-380): RAG architecture, query expansion, self-querying,
  filtered vector search, reranking, `ContextRetriever`, prompt augmentation.
- Repo source of truth: `llm_engineering/application/rag/retriever.py`,
  `llm_engineering/model/inference/run.py`,
  `llm_engineering/infrastructure/inference_pipeline_api.py`,
  `llm_engineering/domain/embedded_chunks.py`, `llm_engineering/settings.py`.
- Superlinked, "A real-time retrieval system for social media data" — multi-index motivation
  cited in the chapter.

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 4.1, 4.2, 6.1

**Outcome**: You can trace and debug a full RAG query, reason about the token budget and
retrieval quality, and extend the pipeline with citations and a retrieval router.
