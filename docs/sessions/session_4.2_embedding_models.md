# Session 4.2: Embedding Models & Cross-Encoders

> Book: Chapter 4, "RAG Feature Pipeline" (pages 136-155) for embedding theory.
> Repo: `llm_engineering/application/networks/embeddings.py`, `application/networks/base.py`.
> Companion: [Session 4.1](session_4.1_advanced_rag.md) uses both models through the retrieval module.

## 🎯 Learning Objectives

By the end of this session, you will:

- Explain what an embedding is and why a dense vector beats one-hot or hashing.
- Read `EmbeddingModelSingleton` and `CrossEncoderModelSingleton` line by line.
- Trace how `embedding_size` sizes a Qdrant collection and how a model swap ripples through the system.
- Compute cosine similarity by hand and interpret the book's worked example.
- Benchmark embedding throughput and reranker score spread on the RTX 5000 (Turing, sm_75, 16 GB).
- Choose a device (CPU vs GPU) and reason about VRAM contention with training.
- Swap embedding models end to end without breaking the vector index.

---

## 🏗️ Architecture Overview

### What is an embedding?

An embedding is a dense numerical representation of an object (text, image, audio) as a vector in a continuous space, where semantically similar objects land close together. In NLP, an encoder transformer maps a token sequence to a fixed-length vector. Sentence-level embedding models (SentenceTransformers) add a pooling step over token vectors so an entire sentence becomes one vector.

```
"The dog is swimming."  ──► encoder ──► mean-pool ──► [0.21, -0.04, ..., 0.77]  (384 floats)
"I am going swimming."  ──► encoder ──► mean-pool ──► [0.03,  0.11, ..., -0.02] (384 floats)
```

The two vectors are close if the sentences are semantically related. Distance is measured with cosine similarity (the project's Qdrant collections use `Distance.COSINE`).

### Why not one-hot or hashing?

| Method | Dimension | Preserves semantics | Curse of dimensionality |
|--------|-----------|---------------------|-------------------------|
| One-hot | = vocabulary size (10,000+) | No | Yes (N tokens × 10,000 dims) |
| Feature hashing | fixed buckets | Partially, loses relationships | No |
| Embeddings | 64-2048, controllable | Yes | No |

The book's example: a 10,000-token vocabulary with one-hot gives each token a length-10,000 vector; an N-token input becomes N × 10,000 parameters, unusable when N ≥ 100. Hashing reduces dimension but loses semantic relationships and risks collisions. Embeddings condense meaning into a low, fixed dimension while keeping geometry.

### Two models, two jobs

```
┌────────────────────────────────────────────────────────────────────┐
│  Retrieval stage 1: BI-ENCODER (fast, approximate)                  │
│                                                                      │
│   query ──► [encoder] ──► q_vec ─┐                                   │
│                                  ├──► cosine(q_vec, d_vec) → top-N   │
│   doc   ──► [encoder] ──► d_vec ─┘        (doc vectors precomputed)  │
│                                                                      │
│   Encodes query and document INDEPENDENTLY. All doc vectors can be   │
│   indexed and compared with a matrix multiply. Scales to millions.   │
├────────────────────────────────────────────────────────────────────┤
│  Retrieval stage 2: CROSS-ENCODER (slow, accurate)                  │
│                                                                      │
│   [query] [SEP] [doc] ──► [transformer] ──► relevance score          │
│                                                                      │
│   Scores one (query, doc) pair at a time. No precomputation, so it   │
│   is used only to rerank a small candidate set (top-N → top-k).      │
└────────────────────────────────────────────────────────────────────┘
```

| Property | Bi-Encoder (`EmbeddingModelSingleton`) | Cross-Encoder (`CrossEncoderModelSingleton`) |
|----------|----------------------------------------|----------------------------------------------|
| Input | one text | a `(query, doc)` pair |
| Output | a vector | a scalar score |
| Precompute docs | Yes | No |
| Scale | Millions | Tens to hundreds per query |
| Used in project | Ingestion + first-pass search | `Reranker.generate` (Session 4.1) |
| Speed | Fast | Slow (linear in candidate count) |
| Default model | `sentence-transformers/all-MiniLM-L6-v2` | `cross-encoder/ms-marco-MiniLM-L-4-v2` |
| Default device | `cpu` (`RAG_MODEL_DEVICE`) | `cpu` (`RAG_MODEL_DEVICE`) |

### Where each model is wired in

```
EmbeddingModelSingleton ──► embedding_data_handlers.py   (embed chunks; tag metadata)
EmbeddingModelSingleton ──► domain/base/vector.py        (_create_collection sizes vectors)
EmbeddingModelSingleton ──► EmbeddingDispatcher          (embed the query at inference, 4.1)
CrossEncoderModelSingleton ─► reranking.py               (rerank candidates, 4.1)
```

The key invariant: **the same `EmbeddingModelSingleton` instance** (a singleton) embeds documents at ingestion and queries at inference, so both live in one vector space.

---

## 📁 Key Files Explained

### 1. `base.py` — the thread-safe singleton

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

**Why a meta-class.** `EmbeddingModelSingleton` and `CrossEncoderModelSingleton` are declared `(metaclass=SingletonMeta)`. The first construction loads the transformer; every later call returns the cached instance. Without this, each `ContextRetriever` and each worker thread would reload hundreds of MB of weights.

**Why the lock.** `ContextRetriever.search` fans out with `ThreadPoolExecutor` (Session 4.1). If several threads reach `__init__` at once, the lock ensures only the first builds the instance; the rest wait and then get the cached one. This is a genuine race guard, not decoration.

**Gotcha.** The instance is keyed by class, not by constructor arguments. `EmbeddingModelSingleton(model_id="...")` returns the *already-built* model, ignoring the new `model_id`. To change models you must change the setting *before* the first call (or run a fresh process). See Common Pitfalls.

---

### 2. `EmbeddingModelSingleton` — the bi-encoder

Full source (all properties verified):

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
    """
    A singleton class that provides a pre-trained transformer model for generating embeddings of input text.
    """

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
```

**The `SentenceTransformer.encode` pipeline.** tokenize → transformer forward pass → **mean pooling** over token embeddings → optional normalization. `all-MiniLM-L6-v2` produces **384-dim** vectors. The book's example confirms the shape:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([
    "The dog sits outside waiting for a treat.",
    "I am going swimming.",
    "The dog is swimming.",
])
print(embeddings.shape)   # [3, 384]
```

**Why `embedding_size` probes with `""`.** A single cheap `encode("")` returns a vector whose length is the true output dimension. This is more robust than hard-coding 384: swap the model and the size follows automatically. `_create_collection` reads it:

```python
# llm_engineering/domain/base/vector.py
@classmethod
def _create_collection(cls, collection_name: str, use_vector_index: bool = True) -> bool:
    if use_vector_index is True:
        vectors_config = VectorParams(size=EmbeddingModelSingleton().embedding_size, distance=Distance.COSINE)
    else:
        vectors_config = {}

    return connection.create_collection(collection_name=collection_name, vectors_config=vectors_config)
```

So the vector dimension and the collection size **can never disagree at creation time** — both come from the same model. This is the design's best safety property.

**Why `cached_property`.** The probe runs once. Every later access reads the cached int. Re-probing on every search would waste a forward pass.

**Why `max_input_length` = `self._model.max_seq_length`.** For MiniLM this is 256 tokens. Anything longer is silently truncated by the encoder. The ingestion side must therefore chunk below this budget. In this repo the chunking handlers use **character** windows, not token windows (`ChunkingDataHandler.metadata`): posts 250 chars with 25 overlap, articles 1000-2000 chars, repositories 1500 chars with 100 overlap (`application/preprocessing/chunking_data_handlers.py`). Because those windows are character-based, verify at ingestion that a chunk does not exceed `max_input_length` tokens; the encoder will not raise, it will just cut text.

**Why CPU by default.** `RAG_MODEL_DEVICE="cpu"` (`settings.py:65`). MiniLM is tiny; CPU is enough for indexing, and it leaves the 16 GB of VRAM free for fine-tuning (Session 5.1). Move to `cuda` only when you deliberately need lower query latency and are not training.

**Why the try/except returns empty.** One malformed input logs and returns `[]` (or an empty ndarray) instead of raising, so a single bad chunk cannot kill a batch. The caller (`embed_batch`) zips inputs with embeddings via `zip(..., strict=False)`, so a short result list truncates rather than misaligns — but it also silently drops the tail. Watch for empty results in logs.

**Dimension cheat-sheet (typical values; verify with the probe):**

| Model | Dim | Max tokens | Notes |
|-------|-----|-----------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` (default) | 384 | 256 | Fast, small, good baseline |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 384 | Higher quality, ~2× cost |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | Strong retrieval, heavier |
| `intfloat/e5-large-v2` | 1024 | 512 | Needs `query:` / `passage:` prefixes |
| `clip-ViT-B-32` (multimodal) | 512 | 77 | Same space for text and images |

> Changing `TEXT_EMBEDDING_MODEL_ID` changes the vector dimension. Existing Qdrant collections sized for 384 dims must be recreated; the stored `metadata.embedding_size` reveals the mismatch.

**How metadata tags the encoder.** Every embedding handler stamps three fields:

```python
# llm_engineering/application/preprocessing/embedding_data_handlers.py
metadata={
    "embedding_model_id": embedding_model.model_id,
    "embedding_size": embedding_model.embedding_size,
    "max_input_length": embedding_model.max_input_length,
}
```

These travel into the Qdrant payload. After a model swap, query the payload to see which index was built with which model.

---

### 3. `CrossEncoderModelSingleton` — the reranker

```python
class CrossEncoderModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = settings.RERANKING_CROSS_ENCODER_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
    ) -> None:
        """
        A singleton class that provides a pre-trained cross-encoder model for scoring pairs of input text.
        """

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

**Why `self._model.model.eval()`.** `CrossEncoder` wraps a HuggingFace model under the `.model` attribute. The double `.model` sets the inner transformer to eval mode (drops dropout), ensuring deterministic scores. The bi-encoder used `self._model.eval()` because `SentenceTransformer` exposes `eval` directly.

**Why `predict` takes a list of pairs.** Batching still applies: the `Reranker` sends all candidates in one call, so the "N calls for N pairs" naive version becomes one batched forward pass. That is the batch-reranking speedup noted in Session 4.1.

**Why a wrapper at all.** The book gives two reasons: (1) the singleton avoids reloading the model on every rerank, and (2) the wrapper defines *our interface* for a reranker. If you later swap in an API-based reranker, you write a new class with the same `__call__` signature and change one line in `Reranker`.

**The default model** `cross-encoder/ms-marco-MiniLM-L-4-v2` is trained on MS MARCO for passage reranking: it scores how well a passage answers a query. The score is not bounded to [0, 1] in general; the code only sorts by it.

---

### 4. `settings.py` — the knobs

```python
# llm_engineering/settings.py (RAG block)
TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
RAG_MODEL_DEVICE: str = "cpu"
```

`Settings` loads from `.env` via Pydantic `BaseSettings`. Change a model by editing `.env` (or `TEXT_EMBEDDING_MODEL_ID`), not by editing code. `OPENAI_MODEL_ID` defaults to `gpt-4o-mini` and drives the query-expansion / self-query LLM (Session 4.1).

---

## 🔬 Deep Dive: Similarity Mathematics

### Cosine similarity

```
cos(q, d) = (q · d) / (||q|| · ||d||)
```

where `q · d = Σᵢ qᵢ dᵢ`, `||q|| = sqrt(Σᵢ qᵢ²)`. Range is [-1, 1]; 1 means identical direction. Qdrant collections are created with `Distance.COSINE`, so ranking by cosine is exactly what the vector search does.

### Worked example (book, page 141)

For the three sentences above, `model.similarity(embeddings, embeddings)` yields:

```
        sent0    sent1    sent2
sent0   1.0000  -0.0389   0.2692
sent1  -0.0389   1.0000   0.3837
sent2   0.2692   0.3837   1.0000
```

Readings:

- `[0,0] = 1.0`: a sentence with itself is always 1 (cos θ = 1).
- `[0,1] = -0.0389`: "The dog sits outside waiting for a treat." vs "I am going swimming." Nearly orthogonal, almost no shared meaning.
- `[0,2] = 0.2692`: the dog sentence vs "The dog is swimming." Small but positive: they share "dog".
- `[1,2] = 0.3837`: "I am going swimming." vs "The dog is swimming." Higher, because both are about swimming.

The matrix is symmetric (`[i,j] = [j,i]`) and has 1s on the diagonal. This is the geometry the retriever exploits: the query vector is compared to every stored chunk vector.

### Normalized embeddings make cosine a dot product

`all-MiniLM-L6-v2` L2-normalizes its output, so `||q|| = ||d|| = 1` and `cos(q,d) = q · d`. Ranking is monotonic either way; normalization just makes the math cheaper and the scores bounded.

### Why reranking helps

The bi-encoder compresses each text into a fixed vector **before** seeing the query, so query-specific nuance is lost. The cross-encoder sees the query and the passage **together**, so attention aligns the specific query terms with passage spans. That extra interaction raises precision, and its cost is why it runs only on the candidate set.

```
N × K candidates ──bi-encoder recall──►  high recall, medium precision
      │
      ▼ cross-encoder
top-k            ──►  high precision → better context → better generation
```

---

## 🔬 Deep Dive: Vector DBs, ANN, and Indexing

Embeddings are only half the story. The retriever (Session 4.1) hands a query vector to Qdrant and expects the nearest stored vectors back. How the DB finds them determines accuracy, latency, and cost.

### Why a vector DB and not a plain index

A standalone index such as FAISS does similarity search well but lacks data management. A vector DB adds CRUD, metadata filtering, scalability, real-time updates, backups, ecosystem integration, and access control — the things a production system needs. The project's feature store uses Qdrant for exactly this reason.

### ANN: approximate nearest neighbor

Exact nearest-neighbor search compares the query against every vector: O(N) per query, too slow at scale. ANN algorithms return *approximate* top matches. Empirically the approximations are good enough, and the accuracy/latency trade-off favors them. This is the first thing to understand about vector search: **you are not getting the exact top-k, you are getting a very good approximation.**

### Index algorithms (book)

| Algorithm | Idea | Trade-off |
|-----------|------|-----------|
| **Random projection** | Project vectors into a lower dimension with a random matrix; preserves relative distances | Faster search, some distortion |
| **PQ (product quantization)** | Split vectors into sub-vectors and quantize each to a code | Big memory savings, approximate distances |
| **LSH (locality-sensitive hashing)** | Hash similar vectors into the same buckets | Fast, approximate, bucket-tuning sensitive |
| **HNSW** | Multi-layer graph; similar nodes connected; navigate to nearest neighbors | Excellent recall/speed, more memory |

Qdrant's default index is HNSW. HNSW builds a navigable small-world graph so a search walks a few hops instead of scanning everything.

```
HNSW layers (coarse -> fine)
  layer 2:   A ─────────────── B          (long-range links)
  layer 1:   A ── C ── D ── E ── B        (medium hops)
  layer 0:   A-C-D-F-G-E-H-I-...-B        (all points, short hops)
  search enters at the top, greedily descends to the query's neighborhood
```

### Filtering: before or after the vector search

Qdrant can filter results based on payload metadata before or after the ANN search. Both have trade-offs:

- **Pre-filter** shrinks the candidate pool first, then searches. Faster when the filter is selective, but can hurt recall if the graph was built assuming all points.
- **Post-filter** searches first, then discards non-matching points. Simpler, but you may retrieve k results and keep fewer than k.

The retriever uses `query_filter=` with `Filter(must=[FieldCondition(key="author_id", ...)])`, i.e., a filtered vector search. This is the "retrieval stage" optimization from the book: combine embedding similarity with a metadata constraint.

### Operational features a vector DB must have

- **Sharding and replication** — partition data across nodes for scale; replicate for availability.
- **Monitoring** — track query latency and RAM/CPU/disk to catch degradation early.
- **Access control** — role-based access so only authorized clients read or write.
- **Backups** — regular snapshots for disaster recovery.

### Hybrid search

Vector search matches semantics; keyword search (BM25) matches exact terms. Hybrid search runs both in parallel, normalizes their scores (they live on different scales), and merges them with a weight `alpha`:

```
score = alpha * normalized_vector_score + (1 - alpha) * normalized_bm25_score
```

The book's motivation: technical content contains exact terms (RAG, LLM, SageMaker) that a pure embedding may blur, while keyword search misses synonyms. Hybrid gets both. This is an exercise-grade extension on top of the current retriever.

---

## 🔬 Deep Dive: Choosing and Adapting an Embedding Model

### The selection criteria

The best model changes with time and use case. The book's guide is the **MTEB** (Massive Text Embedding Benchmark) leaderboard on Hugging Face. Decide based on:

1. **Accuracy** on a benchmark close to your domain.
2. **Memory footprint** and latency on your hardware.
3. **Dimension** (drives vector DB size and search cost).
4. **Max tokens** (must accommodate your chunks).
5. **Licensing** if commercial use matters.

SentenceTransformers and Hugging Face make switching models a one-line change, so experimentation is cheap — but see the re-index pitfall.

### Fine-tuning vs instructor embeddings

When off-the-shelf quality is insufficient for domain jargon, you have two options:

- **Fine-tune the embedding model** on (query, positive, negative) triplets from your domain. Highest quality, most compute and data.
- **Use an instructor model** such as `hkunlp/instructor-base`, which takes an instruction/prompt and guides the embedding:

```python
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR("hkunlp/instructor-base")
sentence = "RAG Fundamentals First"
instruction = "Represent the title of an article about AI:"
embeddings = model.encode([[instruction, sentence]])
print(embeddings.shape)   # (1, 768)
```

The project does not fine-tune embeddings; it relies on the pretrained MiniLM and invests in retrieval techniques instead. That is a deliberate trade-off: fine-tuning consumes compute and human effort, while query expansion / reranking / filtering give large gains with less cost.

### Prefix conventions (a real recall killer)

Some models were trained with prefixes and recall collapses without them:

| Model family | Convention |
|--------------|-----------|
| `e5` | prepend `query: ` to queries, `passage: ` to documents |
| `bge` | prepend an instruction to the query for retrieval |
| `all-MiniLM` | none |

If you swap to `e5` or `bge` and recall drops, the missing prefix is the first suspect.

### Multimodal embeddings (for context)

`clip-ViT-B-32` projects text and images into the same 512-dim space, so you can retrieve images with a sentence:

```python
from PIL import Image
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("clip-ViT-B-32")
img_emb = model.encode(image)
text_emb = model.encode(["A crazy cat smiling.", "A white and brown cat with a yellow bandana.", "A man eating in the garden."])
print(text_emb.shape)                       # (3, 512)
print(model.similarity(img_emb, text_emb))  # tensor([[0.3068, 0.3300, 0.1719]])
```

The matching sentence scores highest. The rule: use a specialized model (CLIP, not MiniLM) whenever you need distances *across* data categories.

### Applications of embeddings beyond RAG

- Encoding categorical variables fed to ML models.
- Recommender systems (encode users and items).
- Clustering and outlier detection.
- Data visualization (UMAP, t-SNE).
- Classification using embeddings as features.
- Zero-shot classification by comparing class descriptions.
- Long-term memory for agents.

### A model-selection decision tree

```
Need cross-modal distances?         -> CLIP / multimodal model
Need exact keyword + semantics?     -> keep embeddings, add hybrid BM25
Domain jargon hurts recall?         -> instructor model, then fine-tune
Chunks exceed 256 tokens?           -> all-mpnet (384) or bge (512)
Serving latency critical?           -> smaller dim model (MiniLM 384) on GPU
Index memory constrained?           -> lower dim + PQ in the vector DB
```

---

## 🛠️ Hands-On: Benchmark and Probe

### Step 1: Measure encoding throughput

```python
import time
from llm_engineering.application.networks import EmbeddingModelSingleton

model = EmbeddingModelSingleton()
texts = ["How does RAG work?"] * 64

start = time.perf_counter()
_ = model(texts, to_list=True)
elapsed = time.perf_counter() - start
print(f"{len(texts) / elapsed:.1f} texts/sec on {model._device}")
```

Run once with `RAG_MODEL_DEVICE=cpu`, once with `cuda`, and compare. Expect CPU to be slower per text but zero VRAM.

### Step 2: Probe dimensions and limits

```python
from llm_engineering.application.networks import EmbeddingModelSingleton

m = EmbeddingModelSingleton()
print("model_id      :", m.model_id)
print("embedding_size:", m.embedding_size)     # 384 for all-MiniLM-L6-v2
print("max_input_len :", m.max_input_length)   # 256 for all-MiniLM-L6-v2
print("device        :", m._device)
```

### Step 3: Compare candidate models (independent of the singleton)

```python
from sentence_transformers import SentenceTransformer

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
for model_id in MODELS:
    m = SentenceTransformer(model_id)
    vec = m.encode("How does RAG work?")
    print(f"{model_id:50s} dim={vec.shape[0]} max_tokens={m.max_seq_length}")
```

`SentenceTransformer` here bypasses the singleton, so you can inspect alternatives without restarting the process.

### Step 4: Rerank a candidate set and inspect score spread

```python
from llm_engineering.application.networks import CrossEncoderModelSingleton

cross = CrossEncoderModelSingleton()
pairs = [
    ("How does RAG work?", "RAG combines retrieval with generation."),
    ("How does RAG work?", "The Eiffel Tower is in Paris."),
    ("How does RAG work?", "Retrieval-augmented generation grounds LLMs in documents."),
]
for text, score in zip([p[1] for p in pairs], cross(pairs), strict=False):
    print(f"{score:8.4f}  {text}")
```

The two relevant passages should score clearly above the Eiffel Tower passage. A small spread means the candidate set is off-topic or the model does not fit the domain.

### Step 5: Confirm precision choice on this GPU

```python
import torch
print("CUDA available :", torch.cuda.is_available())
print("bf16 supported :", torch.cuda.is_bf16_supported())  # False on Turing RTX 5000
print("device name    :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

The RTX 5000 is Turing (compute capability 7.5). `torch.cuda.is_bf16_supported()` is `False`; use fp16 for any GPU work. This also means FlashAttention-2, which needs Ampere or newer, is unavailable (see Common Pitfalls).

---

## 📝 Exercise 1: Swap the Embedding Model End to End

### Task

Replace the embedding model and rebuild the vector index.

1. Set `TEXT_EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2` in `.env`.
2. Delete the existing Qdrant collections (or use a new `DATABASE_NAME`).
3. Re-run feature engineering to re-embed and re-insert chunks.
4. Confirm the new collection dimension is **768**, not 384 (`EmbeddingModelSingleton().embedding_size`).
5. Re-run a RAG query and inspect `chunk.metadata["embedding_model_id"]` and `["embedding_size"]`.

**Goal:** experience the ripple effect model id → dimension → collection size → re-index. It is the most common upgrade and the most common failure.

**Acceptance criteria:** retrieval returns chunks whose metadata shows the new model; no dimension-mismatch errors in the Qdrant logs.

---

## 📝 Exercise 2: Device and Batch-Size Study

### Task

Measure how device and batch size affect embedding and reranking on the RTX 5000.

1. For the default MiniLM, benchmark `Text.encode` with batch sizes 1, 8, 32, 64 at `device="cpu"` and `device="cuda"`.
2. Compute texts/sec and peak GPU memory (`torch.cuda.max_memory_allocated()`) for the CUDA runs.
3. Repeat the reranker benchmark with candidate counts 10, 50, 200.
4. Tabulate the trade-off and recommend a device policy for (a) offline indexing, (b) low-latency serving, (c) while fine-tuning a model.

**Goal:** build intuition that MiniLM is small enough for CPU indexing, and that on a 16 GB card the real risk is not embedding memory but **contention** with training.

**Deliverable:** a short table of `device × batch → throughput/latency/VRAM` with a one-line recommendation.

---

## 🧮 Worked Example: Sizing a Collection and Search Cost

### Vector storage

Each vector costs `dim × bytes_per_float` plus payload overhead. With float32 and 384 dims:

```
bytes per vector = 384 × 4 = 1536 bytes ≈ 1.5 KB
```

Worked numbers (excluding HNSW graph and payload):

| Chunks | Dim | Per-vector | Raw vectors | With HNSW overhead (~2×) |
|--------|-----|-----------|-------------|--------------------------|
| 10,000 | 384 | 1.5 KB | 15 MB | ~30 MB |
| 10,000 | 768 | 3 KB | 30 MB | ~60 MB |
| 10,000 | 1024 | 4 KB | 40 MB | ~80 MB |
| 1,000,000 | 384 | 1.5 KB | 1.5 GB | ~3 GB |
| 1,000,000 | 1024 | 4 KB | 4 GB | ~8 GB |

Switching from MiniLM (384) to bge-large (1024) multiplies raw storage and per-query comparison cost by ~2.7×. This is the concrete cost behind "dimension drives vector DB sizing."

### Query cost

A brute-force comparison is `dim` multiply-adds per stored vector. For 1M vectors at 384 dims:

```
1,000,000 × 384 = 384,000,000 multiply-adds per query
```

HNSW reduces the number of *comparisons* (not the per-comparison cost) to roughly logarithmic hops, which is why ANN makes million-scale search viable. The trade is recall for latency.

### Reranking cost

The cross-encoder adds one transformer forward pass per candidate. With ~6 transformer layers for `ms-marco-MiniLM-L-4`, and ~2 × params FLOPs per token:

```
rerank_flops ≈ candidates × seq_len × model_params × 2
```

For 30 candidates, 512 tokens, 22M params:

```
30 × 512 × 22,000,000 × 2 ≈ 6.8e11 FLOPs per query
```

Small in relative terms, but this is *sequential over candidates* at the pair level, so keep the candidate set small (Session 4.1 keeps it at `k // 3` per category).

---

## 🔁 End-to-End Trace: From Chunk to Query Vector

Tracing one chunk through the ingestion side and one query through the inference side shows where the embedding model is called and what must match.

```
INGESTION (feature pipeline, Session 2.3)
  cleaned_document.content
        │  ChunkingDataHandler.chunk()        (char windows: post 250/25, article 1000-2000, repo 1500/100)
        ▼
  Chunk(content=..., author_id=..., platform=...)
        │  EmbeddingDispatcher.dispatch(chunk)
        ▼
  EmbeddingHandlerFactory -> PostEmbeddingHandler / ArticleEmbeddingHandler / RepositoryEmbeddingHandler
        │  embed_batch(): [c.content for c in chunks] -> embedding_model(texts, to_list=True)
        ▼
  EmbeddingModelSingleton.__call__  -> SentenceTransformer.encode  -> 384-float vector
        │  map_model(): build EmbeddedChunk with metadata {model_id, size, max_input_length}
        ▼
  EmbeddedChunk.to_point()  -> PointStruct(id, vector, payload)
        ▼
  Qdrant upsert into embedded_posts / embedded_articles / embedded_repositories

INFERENCE (retrieval, Session 4.1)
  user string
        │  Query.from_str()
        ▼
  SelfQuery.generate()  -> author_id attached
        ▼
  QueryExpansion.generate() -> N Query variants (same author metadata)
        ▼
  EmbeddingDispatcher.dispatch(query) -> EmbeddedQuery(embedding=...)
        │  SAME EmbeddingModelSingleton as ingestion  ← the critical invariant
        ▼
  Qdrant search(query_vector=embedded_query.embedding, ...)
        ▼
  EmbeddedChunk results -> Reranker (cross-encoder) -> top-k
```

**The invariant to protect:** both sides must call `EmbeddingModelSingleton` with the same `model_id` and the same normalization/pooling. If the ingestion process ran with `TEXT_EMBEDDING_MODEL_ID=A` and the query process with `B`, the vectors are incomparable and every result is noise. Because the singleton is keyed by class and reads `.env` at first construction, a mismatched `.env` between two long-running processes is a real, silent failure mode.

**How to detect it:** read `chunk.metadata["embedding_model_id"]` on retrieved results. If it does not match the running `EmbeddingModelSingleton().model_id`, you have a mismatch.

---

## 🐛 Common Pitfalls

- **Singleton ignores constructor args after first build.** `EmbeddingModelSingleton(model_id="other")` returns the first instance. Change `.env` before the first call, or restart the process.
- **Dimension mismatch after a model swap.** Qdrant rejects points whose vector length differs from the collection. Recreate collections whenever `embedding_size` changes.
- **`max_input_length` truncation.** The encoder silently truncates text longer than `max_seq_length` (256 for MiniLM). The repo chunks by characters (post 250, article 1000-2000, repo 1500), so confirm chunk token counts stay under the limit.
- **`bge` / `e5` prefix requirements.** `e5` models need `query:` and `passage:` prefixes to match their training; using them raw degrades recall. `bge` recommends an instruction prefix for queries.
- **Cross-encoder cost.** Reranking 100 candidates is ~100 forward passes. Keep the candidate set small (Session 4.1 retrieves `k // 3` per category).
- **Device contention on 16 GB.** Running either model on CUDA while training competes for VRAM. Keep `RAG_MODEL_DEVICE=cpu` during training runs.
- **No bfloat16 on Turing.** `is_bfloat16_supported()` is `False` on the RTX 5000; use fp16 for GPU work.
- **No FlashAttention-2.** Turing (sm_75) lacks the required instructions; FlashAttention-2 needs Ampere (sm_80)+. Unsloth/transformers will fall back to vanilla attention on this card, so throughput and memory estimates from Ampere+ guides do not transfer.
- **Empty embeddings on error.** A failed `encode` returns `[]`; because `zip(..., strict=False)` is used downstream, the corresponding chunk is silently dropped. Grep logs for `Error generating embeddings`.
- **Metadata is the audit trail.** Always read `chunk.metadata["embedding_model_id"]` when debugging recall after a change; it tells you which model built the index.

---

## 🎓 Knowledge Check

1. **Why can a bi-encoder scale but a cross-encoder cannot?**
   Bi-encoders precompute document vectors independently; cross-encoders must process each (query, doc) pair at query time.

2. **What determines the Qdrant collection vector size?**
   `EmbeddingModelSingleton.embedding_size`, probed from the model with `encode("")`.

3. **What does `max_input_length` control?**
   The encoder's token budget (`max_seq_length`). Text beyond it is truncated, so chunking must stay under it.

4. **How does `Reranker.generate` order documents?**
   It scores all `(query, chunk)` pairs with the cross-encoder and sorts by score descending, keeping the top `keep_top_k`.

5. **Why is the embedding device CPU by default?**
   MiniLM is small, and CPU avoids competing with training for VRAM.

6. **What is stored on each vector that helps detect a model change?**
   `metadata.embedding_model_id`, `metadata.embedding_size`, `metadata.max_input_length`.

7. **Why is the SingletonMeta lock necessary here?**
   The retriever searches concurrently; without the lock, multiple threads could each build the model.

8. **What does `cos(q, d)` compute, and what is its range?**
   The cosine of the angle between vectors: dot product over the product of norms; range [-1, 1].

9. **Interpret `similarities[0,1] = -0.0389` in the book example.**
   The two sentences are nearly orthogonal, i.e., semantically unrelated.

10. **Why does normalization make cosine a dot product?**
    With `||q|| = ||d|| = 1`, the denominator is 1, so `cos = q · d`.

11. **What does `self._model.model.eval()` do for the cross-encoder?**
    Sets the inner HuggingFace transformer to eval mode (dropout off) for deterministic scores.

12. **Why wrap the models in singletons at all?**
    To load weights once and define a swappable interface.

13. **Which precision and attention kernel apply on the RTX 5000?**
    fp16 (no bf16) and no FlashAttention-2 (needs Ampere+).

14. **What happens if you change `TEXT_EMBEDDING_MODEL_ID` without re-indexing?**
    Query vectors and stored vectors are in different spaces; retrieval returns garbage.

15. **Name two embedding models with a higher dimension than MiniLM and their cost.**
    `all-mpnet-base-v2` (768) and `bge-large-en-v1.5` (1024); both are larger and slower.

---

## 🔎 Troubleshooting Checklist

When retrieval quality drops after any change to the embedding or reranking setup, work down this list in order:

1. **Model id matches the index.** Compare the running `EmbeddingModelSingleton().model_id` against `chunk.metadata["embedding_model_id"]` on a retrieved chunk. A mismatch means the index was built with another model.
2. **Dimension matches the collection.** `EmbeddingModelSingleton().embedding_size` must equal the Qdrant `VectorParams.size`. A mismatch raises on insert/search.
3. **Chunks fit the encoder.** Confirm chunk token counts are below `max_input_length`; character-based chunk windows can still exceed 256 tokens for dense text.
4. **Prefixes present.** If using `e5`/`bge`, confirm the `query:` / `passage:` prefixes are applied at both ends.
5. **Device is intentional.** `RAG_MODEL_DEVICE` should be `cpu` while training, `cuda` only when serving and not training.
6. **Reranker spread is sane.** Score a known-relevant and known-irrelevant pair; the relevant one must rank higher. A flat spread signals the wrong reranker or an off-topic candidate set.
7. **Precision is fp16.** On the RTX 5000, bf16 is unavailable; ensure nothing forces it.
8. **No silent empties.** Grep logs for `Error generating embeddings`; empty results drop chunks via `zip(..., strict=False)`.

---

## 📖 Glossary

- **Embedding** — a dense vector representation of an object preserving semantic relationships.
- **Bi-encoder** — encodes texts independently; fast, used for indexing and first-pass search.
- **Cross-encoder** — scores a (query, document) pair jointly; slow, used for reranking.
- **Mean pooling** — averaging token embeddings into one sentence vector.
- **L2 normalization** — scaling a vector to unit length; turns cosine into a dot product.
- **Cosine similarity** — dot product over the product of norms; measures vector direction alignment.
- **`max_seq_length`** — maximum tokens the encoder processes; excess is truncated.
- **ANN** — approximate nearest neighbor search used by the vector DB.
- **MS MARCO** — the passage-ranking dataset the default cross-encoder is trained on.
- **`RAG_MODEL_DEVICE`** — the setting that picks CPU or CUDA for both RAG models.
- **Metadata audit trail** — the `embedding_model_id`/`embedding_size` payload stored with each vector.
- **MTEB** — Massive Text Embedding Benchmark; leaderboard for choosing embedding models.
- **HNSW** — hierarchical navigable small world; the graph index Qdrant uses by default.
- **PQ** — product quantization; compresses vectors into codes to save memory.
- **LSH** — locality-sensitive hashing; maps similar vectors into the same buckets.
- **Hybrid search** — combining vector search with keyword search (BM25) via a weight `alpha`.
- **Pre-filter / post-filter** — applying a payload filter before or after the ANN search.
- **Instructor model** — an embedding model that takes an instruction to steer the embedding.
- **Normalization** — scaling a vector to unit length so cosine reduces to a dot product.

---

## 🔗 Next Session

**Session [5.1](session_5.1_sft.md): Supervised Fine-Tuning (SFT)**

We move from retrieval to training and fine-tune Llama 3.1 8B with LoRA/QLoRA on the instruction dataset.

Related: [Session 4.1](session_4.1_advanced_rag.md) (uses both models), [Session 2.2](session_2.2_text_preprocessing.md) (chunking that must respect `max_input_length`).

---

## 📚 Additional Resources

- [SentenceTransformers](https://www.sbert.net/)
- [Cross-Encoders in SentenceTransformers](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MS MARCO reranking models](https://huggingface.co/cross-encoder)
- [Qdrant distance metrics](https://qdrant.tech/documentation/concepts/collections/#distance-metrics)
- [UMAP for visualizing embeddings](https://umap-learn.readthedocs.io/en/latest/index.html)
- Source: `llm_engineering/application/networks/embeddings.py`, `networks/base.py`, `domain/base/vector.py`, `application/preprocessing/embedding_data_handlers.py`.

---

**Estimated Time**: 4-5 hours

**Prerequisites**: [Sessions 2.2](session_2.2_text_preprocessing.md), [2.3](session_2.3_feature_engineering.md), [4.1](session_4.1_advanced_rag.md)

**Outcome**: You can explain, benchmark, and swap embedding and reranking models, predict the effect on the vector index, and reason about precision and VRAM constraints on the RTX 5000.
