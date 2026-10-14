# Session 2.2: Text Preprocessing Pipeline (Cleaning and Chunking)

> Book reference: Chapter 4, *RAG Feature Pipeline* (pages 168-203).
> Repo: `llm_engineering/application/preprocessing/*.py`, `pipelines/feature_engineering.py`, `steps/feature_engineering/clean.py`, `steps/feature_engineering/rag.py`.

## 🎯 Learning Objectives

By the end of this session, you will:
- Explain the **factory + dispatcher + handler** pattern the repo repeats for cleaning, chunking, and embedding.
- Read the two text operations: `clean_text` (regex normalization) and the two chunking functions `chunk_text` / `chunk_article`.
- Understand why chunking is **category-specific** and how each handler picks its parameters.
- Trace how a chunk ID becomes a deterministic function of its text via `hashlib.md5`.
- Distinguish the two category-inference paths: collection name (cleaning) vs `Config.category` (chunking).
- Prove idempotency by re-running cleaning and chunking and comparing IDs.

---

## 🏗️ Architecture Overview

### Where preprocessing sits

Preprocessing is the bridge between the ETL stage (raw documents in MongoDB) and feature engineering (embedded vectors in Qdrant). Chapter 4 lists five RAG feature steps — extract, clean, chunk, embed, load — and this session owns the middle two.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     Preprocessing Pipeline                               │
│                                                                          │
│  Raw docs (MongoDB): Post / Article / Repository                         │
│        │  content: dict                                                  │
│        ▼                                                                 │
│  ┌────────────────────────────────┐                                      │
│  │ CleaningDispatcher.dispatch     │  category = DataCategory(            │
│  │    └─ CleaningHandlerFactory    │      data_model.get_collection_name()│
│  │        ├─ PostCleaningHandler   │  )                                   │
│  │        ├─ ArticleCleaningHandler│                                      │
│  │        └─ RepositoryCleaningHandler                                     │
│  └────────────────────────────────┘                                      │
│        │  CleanedDocument: content is now a single str                   │
│        ▼                                                                 │
│  ┌────────────────────────────────┐                                      │
│  │ ChunkingDispatcher.dispatch     │  category = data_model.get_category()│
│  │    └─ ChunkingHandlerFactory    │                                      │
│  │        ├─ PostChunkingHandler   │  250 / 25   (token)                  │
│  │        ├─ ArticleChunkingHandler│  1000..2000 (sentence)               │
│  │        └─ RepositoryChunkingHandler 1500/100 (token)                  │
│  └────────────────────────────────┘                                      │
│        │  Chunk objects with deterministic md5->UUID ids                 │
│        ▼                                                                 │
│  EmbeddingDispatcher (Session 2.3)                                       │
└────────────────────────────────────────────────────────────────────────┘
```

### The three-level pattern, repeated three times

Every transform has the same shape:

```
Dispatcher  ──reads category──►  Factory  ──builds──►  Handler
  (entry point)                  (selection)          (logic)
```

| Level | Cleaning | Chunking | Embedding |
|-------|----------|----------|-----------|
| Dispatcher | `CleaningDispatcher` | `ChunkingDispatcher` | `EmbeddingDispatcher` |
| Factory | `CleaningHandlerFactory` | `ChunkingHandlerFactory` | `EmbeddingHandlerFactory` |
| Base handler | `CleaningDataHandler` | `ChunkingDataHandler` | `EmbeddingDataHandler` |

This is a **simple factory**, not the builder pattern of the crawler dispatcher. Each factory is one `if/elif` chain returning a concrete handler; each dispatcher infers the category and calls `handler.<verb>()`.

### The category enum

```python
# llm_engineering/domain/types.py
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

Because it is a `StrEnum`, `DataCategory.POSTS == "posts"` is `True`, and `DataCategory("posts")` reconstructs the member. That constructor call is exactly what cleaning uses to go from a collection name to a category.

---

## 📁 Key Files Explained

### 1. `llm_engineering/application/preprocessing/operations/cleaning.py` — the only cleaning primitive

```python
# llm_engineering/application/preprocessing/operations/cleaning.py
import re


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
```

**What it does, line by line**:

- `re.sub(r"[^\w\s.,!?]", " ", text)` — replaces any character that is **not** a word character (`\w` = `[A-Za-z0-9_]`), whitespace, or one of `.,!?` with a space. This removes Markdown (`#`, `*`, `` ` ``), braces, HTML remnants, and most punctuation, without deleting the sentence-ending marks that chunking relies on.
- `re.sub(r"\s+", " ", text)` — collapses any run of whitespace (including newlines and tabs) into one space.
- `.strip()` — trims leading/trailing whitespace.

**Why so simple**: the book calls cleaning "more art than science". The project keeps it minimal and deterministic. Semantic quality comes from chunking and prompt design, not from aggressive normalization. A regex-only cleaner is auditable and reproducible.

**What it does NOT do** (despite the book's prose mentioning emoji/URL handling in general): the repo's actual `clean_text` does not strip emojis or replace URLs with placeholders. Treat the book's Chapter 4 list as the *conceptual* cleaning checklist and the repo function as the *concrete implementation* in this codebase.

**Worked examples**:

| Input | Output |
|-------|--------|
| `"# Title\n\nHello   world!"` | `"Title Hello world!"` |
| `` "`code` and *bold*" `` | `"code and bold"` |
| `"a\n\n\nb   c"` | `"a b c"` |
| `"§°±"` | `""` (everything replaced, then stripped) |

### 2. `llm_engineering/application/preprocessing/operations/chunking.py` — two strategies

```python
# llm_engineering/application/preprocessing/operations/chunking.py
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from llm_engineering.application.networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
    text_split_by_characters = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )
    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens


def chunk_document(text: str, min_length: int, max_length: int) -> list[str]:
    """Alias for chunk_article()."""

    return chunk_article(text, min_length, max_length)


def chunk_article(text: str, min_length: int, max_length: int) -> list[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    extracts = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())
            current_chunk = sentence + " "

    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())

    return extracts
```

**`chunk_text` is a two-pass split**:

1. **Pass 1 — paragraphs by characters.** `RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)` splits on blank lines. This keeps whole paragraphs together so ideas are not cut mid-thought. Note `chunk_overlap=0` here.
2. **Pass 2 — tokens by the embedding tokenizer.** `SentenceTransformersTokenTextSplitter` re-splits each paragraph using `tokens_per_chunk=embedding_model.max_input_length` and `model_name=embedding_model.model_id`, with the caller's `chunk_overlap`.

**Why two passes**: the embedding model has a hard token limit. Paragraph splitting avoids cutting mid-idea; the token pass guarantees every returned chunk fits the model. `tokens_per_chunk` is bound to the *loaded model*, so swapping `TEXT_EMBEDDING_MODEL_ID` automatically changes chunk size — no hard-coded numbers to drift.

> The `embedding_model` is a module-level singleton created at import. Importing `operations.chunking` therefore loads the embedding model into memory (onto `RAG_MODEL_DEVICE`, default `cpu`).

**`chunk_article` is sentence-aware**:

- The regex `r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"` splits on whitespace that follows `.`, `?`, or `!`, with two negative lookbehinds guarding against abbreviations like `e.g.` (a word char, dot, word char) and initials like `A.` or `Dr.`-style `[A-Z][a-z].`.
- It greedily appends sentences while the running length stays `<= max_length`.
- On overflow, it emits the current chunk **only if it is at least `min_length`**, then starts a new chunk with the overflowing sentence. This prevents tiny, low-signal tail chunks.
- After the loop, the final buffer is emitted if it meets `min_length`.

**Tradeoff table**:

| Function | Split unit | Bound | Best for | Risk |
|----------|-----------|-------|----------|------|
| `chunk_text` | paragraph (chars) then token | model `max_input_length` | posts, code | token split can cut a sentence mid-word at the boundary |
| `chunk_article` | sentence (regex) | `min_length`/`max_length` chars | prose articles | a single `max_length`-exceeding sentence is emitted as its own chunk |

Note `chunk_document` is now just a **backward-compatible alias** for `chunk_article`.

### 3. `llm_engineering/application/preprocessing/cleaning_data_handlers.py` — raw to cleaned

```python
# llm_engineering/application/preprocessing/cleaning_data_handlers.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)
from llm_engineering.domain.documents import (
    ArticleDocument,
    Document,
    PostDocument,
    RepositoryDocument,
)

from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass


class PostCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PostDocument) -> CleanedPostDocument:
        return CleanedPostDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            image=data_model.image if data_model.image else None,
        )


class ArticleCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ArticleDocument) -> CleanedArticleDocument:
        valid_content = [content for content in data_model.content.values() if content]

        return CleanedArticleDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(valid_content)),
            platform=data_model.platform,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> CleanedRepositoryDocument:
        return CleanedRepositoryDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )
```

**Key concepts**:

- **The transform is dict → str.** A raw document's `content` is a dict; the cleaned document's `content` is a single string. `" #### ".join(content.values())` flattens it. The `#` characters are later stripped by `clean_text`, so the separator is effectively whitespace — the intent is a visual marker in the raw string, not a surviving delimiter.
- **`id` is preserved.** `id=data_model.id` means cleaning does not mint a new identity. Downstream chunks carry `document_id` pointing to this same id.
- **Article filters empties.** `valid_content` drops `None`/empty fields (e.g., a missing subtitle) before joining, so you do not get `"Title ####  #### Body"`.
- **`Generic[DocumentT, CleanedDocumentT]`** makes the base handler type-parametric; `PostCleaningHandler` narrows both to the concrete types.
- **`Platform`/author fields are copied verbatim** so provenance travels with the data.

### 4. `llm_engineering/application/preprocessing/chunking_data_handlers.py` — per-category chunking

```python
# llm_engineering/application/preprocessing/chunking_data_handlers.py (excerpt)
class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass


class PostChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 250,
            "chunk_overlap": 25,
        }

    def chunk(self, data_model: CleanedPostDocument) -> list[PostChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(
            cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"]
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = PostChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                image=data_model.image if data_model.image else None,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


class ArticleChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 1000,
            "max_length": 2000,
        }

    def chunk(self, data_model: CleanedArticleDocument) -> list[ArticleChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_article(
            cleaned_content, min_length=self.metadata["min_length"], max_length=self.metadata["max_length"]
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ArticleChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


class RepositoryChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 1500,
            "chunk_overlap": 100,
        }
    # chunk() mirrors PostChunkingHandler but builds RepositoryChunk with name/link
```

**Chunk-size rationale**:

| Category | Strategy | Parameters | Why |
|----------|----------|-----------|-----|
| Posts | token (`chunk_text`) | `chunk_size=250`, `chunk_overlap=25` | short-form; tight chunks improve retrieval precision |
| Articles | sentence (`chunk_article`) | `min_length=1000`, `max_length=2000` | long prose; keep sentences intact in larger windows |
| Repositories | token (`chunk_text`) | `chunk_size=1500`, `chunk_overlap=100` | code needs larger windows for context |

> **Important correction**: the article handler's metadata is `min_length=1000, max_length=2000` (character counts for the sentence packer). Older versions of this doc listed "1000..2000 sentence chunks" imprecisely — these are *characters*, and the unit is *sentences*, not tokens.

**Deterministic IDs**: `hashlib.md5(chunk.encode()).hexdigest()` produces a 32-hex-char digest, then `UUID(chunk_id, version=4)` wraps it as a UUID4-shaped string. The id is a **pure function of the chunk text**. Re-running the pipeline yields identical ids, so a Qdrant upsert overwrites the same point instead of duplicating it. This is the idempotency guarantee.

**Why `metadata` lives on the handler**: `self.metadata` is copied onto every chunk's `metadata` field, so each stored vector records the exact parameters that produced it. This is observability baked into the payload.

### 5. `llm_engineering/application/preprocessing/dispatchers.py` — the three facades

```python
# llm_engineering/application/preprocessing/dispatchers.py (excerpt)
class CleaningDispatcher:
    factory = CleaningHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        data_category = DataCategory(data_model.get_collection_name())
        handler = cls.factory.create_handler(data_category)
        clean_model = handler.clean(data_model)

        logger.info(
            "Document cleaned successfully.",
            data_category=data_category,
            cleaned_content_len=len(clean_model.content),
        )

        return clean_model


class ChunkingDispatcher:
    factory = ChunkingHandlerFactory

    @classmethod
    def dispatch(cls, data_model: VectorBaseDocument) -> list[VectorBaseDocument]:
        data_category = data_model.get_category()
        handler = cls.factory.create_handler(data_category)
        chunk_models = handler.chunk(data_model)

        logger.info(
            "Document chunked successfully.",
            num=len(chunk_models),
            data_category=data_category,
        )

        return chunk_models
```

**Two category-inference paths** — this is the subtle part:

- **Cleaning** takes a *raw* document whose class only has a Mongo `Settings.name` (e.g. `"articles"`). It converts that string: `DataCategory(data_model.get_collection_name())`.
- **Chunking** takes a *vector* document whose class has `Config.category` (a `DataCategory` already). It calls `data_model.get_category()`.

Both end at the same enum, but they read from different class configuration systems (`Settings.name` for Mongo classes vs `Config.category` for vector classes). Confusing the two is a common source of bugs when adding a category.

**Extension cost**: adding a category means one enum value, one document class, one cleaned class, one chunk class, one cleaning handler, one chunking handler, and two factory branches. Nothing else changes — O(1) per call site.

---

## 🔧 Pipeline Integration

### `steps/feature_engineering/clean.py`

```python
# steps/feature_engineering/clean.py
@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    cleaned_documents = []
    for document in documents:
        cleaned_document = CleaningDispatcher.dispatch(document)
        cleaned_documents.append(cleaned_document)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))

    return cleaned_documents


def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    metadata = {"num_documents": len(cleaned_documents)}
    for document in cleaned_documents:
        category = document.get_category()
        if category not in metadata:
            metadata[category] = {}
        if "authors" not in metadata[category]:
            metadata[category]["authors"] = list()

        metadata[category]["num_documents"] = metadata[category].get("num_documents", 0) + 1
        metadata[category]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
```

The `_get_metadata` helper produces per-category document counts and de-duplicated author lists, attached to the ZenML output artifact.

### `steps/feature_engineering/rag.py` — `chunk_and_embed`

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

> **File naming**: the `chunk_and_embed` step lives in `steps/feature_engineering/rag.py` and is re-exported from `steps/feature_engineering/__init__.py`. There is no `chunk_and_embed.py`; the step function is named `chunk_and_embed`.

The batch size `10` is the memory/speed dial (detailed in Session 2.3).

---

## 🛠️ Hands-On

### Step 1: Fetch a raw document

```python
from llm_engineering.domain.documents import ArticleDocument

doc = ArticleDocument.bulk_find(author_id="<user-uuid>")[0]
print(type(doc.content), list(doc.content.keys()))
# <class 'dict'> ['Title', 'Subtitle', 'Content', 'language']
```

### Step 2: Clean it

```python
from llm_engineering.application.preprocessing import CleaningDispatcher

cleaned = CleaningDispatcher.dispatch(doc)
print(cleaned.get_collection_name())   # cleaned_articles
print(len(cleaned.content), cleaned.content[:200])
```

### Step 3: Chunk it

```python
from llm_engineering.application.preprocessing import ChunkingDispatcher

chunks = ChunkingDispatcher.dispatch(cleaned)
print(f"{len(chunks)} chunks")
for c in chunks[:3]:
    print(c.id, "|", c.content[:80])
```

### Step 4: Prove idempotency

```python
chunks_again = ChunkingDispatcher.dispatch(cleaned)
assert [c.id for c in chunks] == [c.id for c in chunks_again]
print("Deterministic chunk IDs confirmed")
```

### Step 5: Inspect the operation directly

```python
from llm_engineering.application.preprocessing.operations import clean_text, chunk_text

print(repr(clean_text("# Hello   world\n\n- item 1")))
print(chunk_text("para one.\n\npara two is a bit longer.", chunk_size=20, chunk_overlap=0))
```

---

## 📝 Exercise 1: Add a short "comment" category

**Task**: support short comment documents with their own chunking rule.

1. Add `COMMENTS = "comments"` to `DataCategory`.
2. Add `CommentDocument` (Mongo, `Settings.name = DataCategory.COMMENTS`) and `CleanedCommentDocument` (vector, `Config.name = "cleaned_comments"`, `Config.category = DataCategory.COMMENTS`, `use_vector_index = False`).
3. Add `CommentChunk` (`Config.category = DataCategory.COMMENTS`).
4. Add `CommentCleaningHandler` and `CommentChunkingHandler` (try `chunk_size=120, chunk_overlap=10`).
5. Register both in their factories.
6. Round-trip a sample document through cleaning and chunking.

**Goal**: internalize the three-level pattern and where each addition belongs.

---

## 📝 Exercise 2: Tune article chunking against retrieval

**Task**: measure how `min_length`/`max_length` change chunk count and retrieval.

1. Take one long `CleanedArticleDocument`.
2. Chunk it with `(1000, 2000)`, `(500, 1000)`, and `(2000, 4000)` by passing parameters directly to `chunk_article`.
3. Record chunk count and average chunk length for each setting.
4. Embed each variant (Session 2.3) and search a fixed query; compare the top-3 chunks by eye.
5. Write down which setting gives the most self-contained chunks.

**Goal**: connect a chunking hyperparameter to RAG quality, not just chunk count.

---

## 🐛 Common Pitfalls

- **Empty content**: `clean_text` returns `""` for all-symbol input. `SentenceTransformersTokenTextSplitter` on an empty string yields no chunks, which is safe but silent — no warning is logged. Guard upstream if this matters.
- **`max_input_length` mismatch**: `chunk_text` always uses `embedding_model.max_input_length`. Never hard-code a token size, or chunks may be silently truncated at embed time.
- **Factory coverage**: unknown categories raise `ValueError("Unsupported data type")`. New categories must be registered in **every** factory you route through.
- **Wrong category source**: cleaning reads `Settings.name`, chunking reads `Config.category`. Mixing them raises or misroutes.
- **`" #### "` separator surprises**: it is a join token only; `clean_text` strips the `#`. Do not depend on it surviving.
- **Article metadata is characters, not tokens**: `min_length`/`max_length` bound the sentence packer in characters.
- **`chunk_article` long-sentence behavior**: a single sentence longer than `max_length` is emitted alone (it cannot be split), so it can exceed the bound.

---

## 🎓 Knowledge Check

1. **Why two splitting passes in `chunk_text`?**
   - Answer: paragraph splitting keeps ideas intact; the token pass enforces the embedding model's token limit.

2. **Why is a chunk ID an md5 of its content?**
   - Answer: so re-processing yields identical IDs and the Qdrant upsert is idempotent (no duplicates).

3. **How does `CleaningDispatcher` decide its handler?**
   - Answer: `DataCategory(data_model.get_collection_name())` maps the Mongo collection name to a category; the factory selects the handler.

4. **How does `ChunkingDispatcher` decide its handler?**
   - Answer: `data_model.get_category()`, which reads the vector class's `Config.category`.

5. **What stays the same between a raw document and its cleaned form?**
   - Answer: `id`, `platform`, `author_id`, `author_full_name` (plus `link`/`name`/`image` where present).

6. **Why does `chunk_article` need both `min_length` and `max_length`?**
   - Answer: `max_length` caps chunk size; `min_length` prevents emitting tiny, low-signal chunks.

7. **Where is per-category chunk configuration stored?**
   - Answer: in each handler's `metadata` property, copied onto every chunk's `metadata` field.

8. **What does `clean_text` remove, exactly?**
   - Answer: every character that is not a word char, whitespace, or `. , ! ?` — then it collapses whitespace and strips.

9. **What are the article chunking parameters in the repo?**
   - Answer: `min_length=1000`, `max_length=2000` characters.

10. **What is the relationship between `chunk_document` and `chunk_article`?**
    - Answer: `chunk_document` is a backward-compatible alias that calls `chunk_article`.

11. **Why does the article cleaning handler filter empty values?**
    - Answer: to avoid joining `None`/empty subtitle fields into the string.

12. **What is the return type of each `ChunkingDataHandler.chunk`?**
    - Answer: a `list` of category-specific `Chunk` subclasses (`PostChunk`, `ArticleChunk`, `RepositoryChunk`).

13. **What does `CleanedDocument.use_vector_index = False` mean?**
    - Answer: cleaned docs are stored in Qdrant without vectors; the metadata index acts like a NoSQL store (for fine-tuning), not a search index.

14. **What batching size does `chunk_and_embed` use, and why?**
    - Answer: 10; it balances GPU throughput against VRAM.

15. **Which file contains the `chunk_and_embed` step?**
    - Answer: `steps/feature_engineering/rag.py`.

---

## 📖 Glossary

- **Dispatcher**: the entry point that infers a category and delegates to a factory-built handler.
- **Factory (simple)**: an `if/elif` selector returning a concrete handler for a category.
- **Handler**: the object containing the actual transform logic.
- **Deterministic ID**: an ID derived purely from content (md5), so identical content maps to identical IDs.
- **Idempotent upsert**: writing the same deterministic ID again overwrites rather than duplicates.
- **Token splitter**: splits text by model tokens; enforces the embedding window.
- **Sentence splitter**: splits text on sentence boundaries; preserves meaning for prose.
- **Character splitter**: splits by raw character count; used here only for paragraph boundaries.
- **`max_input_length`**: the embedding model's `max_seq_length`, the token bound on a chunk.

---

## 🔗 Next Session

**Session 2.3: Feature Engineering Pipeline**

We embed the chunks with `EmbeddingModelSingleton`, batch them, and load them into Qdrant.

---

## 📚 Additional Resources

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [SentenceTransformers Token Splitter](https://python.langchain.com/docs/integrations/document_transformers/sentence_transformers_token/)
- [Factory pattern](https://refactoring.guru/design-patterns/factory-method)
- [Python `re` module](https://docs.python.org/3/library/re.html)
- [Python `hashlib`](https://docs.python.org/3/library/hashlib.html)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 1.2, 1.3, 2.1

**Outcome**: You can route any document category through cleaning and category-specific chunking, explain why chunks get deterministic IDs, and tune chunk parameters with reason.
