# Session 9.2: Performance Optimization

## 🎯 Learning Objectives

By the end of this session, you will:
- Identify every batching knob in the pipeline and what it controls
- Reason about VRAM, throughput, and latency tradeoffs
- Apply caching at the ZenML and application levels
- Understand quantization choices across training, serving, and evaluation
- Profile a workload before optimizing it
- Distinguish latency-bound from throughput-bound steps and fix the right one
- Recognize where the code duplicates expensive work (model loads, tokenizer loads, client construction)

---

## 🏗️ Architecture Overview

Performance in this project is governed by a handful of recurring patterns. Every knob trades one resource for another; none is free.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Performance levers                                │
│                                                                            │
│  1. Batching        ──  embed=10, load=4, generate=24, eval=5              │
│  2. Parallelism     ──  ThreadPoolExecutor across queries/collections      │
│  3. Caching         ──  ZenML step cache + cached_property + singletons    │
│  4. Quantization    ──  4-bit training, 8-bit serving, vLLM eval           │
│  5. Device choice   ──  RAG_MODEL_DEVICE=cpu vs cuda                       │
│  6. Retrieval size  ──  k and k//3 per collection                          │
└──────────────────────────────────────────────────────────────────────────┘
```

Four resource dimensions are always in play. A change to one knob moves you along at least two of them.

| Resource | Constrained by | Symptom when exhausted | Knob that relieves it |
|----------|----------------|------------------------|-----------------------|
| VRAM | 16 GB on the RTX 5000 | CUDA OOM | smaller batch, quantization, CPU offload, checkpointing |
| Throughput | GPU saturation / API concurrency | slow wall-clock | larger batch (GPU), more threads (API) |
| Latency | per-request critical path | slow single response | caching, smaller `k`, fewer query expansions |
| Money | OpenAI tokens, SageMaker hours | budget burn | fewer tokens, smaller models, caching |

**Rule: measure before optimizing.** Every claim below is a hypothesis until you profile the actual workload. The most common failure in this codebase is raising batch sizes because "bigger is faster", then hitting an OOM or a 429 retry storm that makes the run *slower*.

---

## 📁 Batching Knobs in the Code

### 1. Embedding batch - `chunk_and_embed` (batch = 10)

```python
# steps/feature_engineering/rag.py
        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)
```

The batch helper is a lazy generator, so only one batch is ever resident:

```python
# llm_engineering/application/utils/misc.py
def batch(list_: list, size: int) -> Generator[list, None, None]:
    yield from (list_[i : i + size] for i in range(0, len(list_), size))
```

**Tradeoff**: larger batches raise GPU utilization and throughput but raise peak activation memory. On a 16 GB card with a large embedding model, drop this to 4 or use CPU. The batch is per-document: chunks from one document are batched, then embedding continues with the next document, so a document with 100 chunks is embedded as ten batches of ten.

### 2. Qdrant load batch - `load_to_vector_db` (batch = 4)

```python
# steps/feature_engineering/load_to_vector_db.py
    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")
        for documents_batch in utils.misc.batch(documents, size=4):
            try:
                document_class.bulk_insert(documents_batch)
            except Exception:
                logger.error(f"Failed to insert documents into {document_class.get_collection_name()}")
                return False
```

**Tradeoff**: point upserts are network-bound. Small batches keep request payloads modest and avoid timeouts; larger batches reduce round-trips. Four is conservative. The step loads **both** cleaned documents and embedded chunks (the pipeline calls it twice), so the effective cost is 2× the collection size in round-trips.

### 3. LLM generation batch - `generate` (batch = 24)

```python
# llm_engineering/application/dataset/generation.py
            batches = utils.misc.batch(langchain_category_prompts, size=24)
            for batch in batches:
                batched_dataset_samples = chain.batch(batch, stop=None)
```

**Tradeoff**: `chain.batch` issues concurrent LLM requests. Higher batch = higher throughput until you hit OpenAI rate limits (HTTP 429). If you see rate-limit errors, lower this first. The SDK retries with backoff, so the observable failure is a *slower* run, not a crash — which is why this must be measured, not guessed.

### 4. Evaluation threads and batch

```python
# llm_engineering/model/evaluation/evaluate.py
def evaluate_answers(model_id: str, num_threads: int = 10, batch_size: int = 5) -> Dataset:
    ...
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(evaluate_batch, batch, start_index) for start_index, batch in batches]
```

**Tradeoff**: 10 concurrent judge calls balances throughput and rate limits; batch size 5 groups samples per thread. Tune against your OpenAI tier. Note that `evaluate_batch` constructs a **new `OpenAI` client per batch**:

```python
def evaluate_batch(batch, start_index):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return [(i, evaluate_answer(instr, ans, client)) for i, (instr, ans) in enumerate(batch, start=start_index)]
```

With `batch_size=5` and 1000 samples that is 200 client constructions. The client is cheap relative to a network round-trip, but hoisting it (one client shared across threads, or one per thread) removes avoidable allocation. This is a real, low-risk optimization.

### 5. Parallel I/O - `query_data_warehouse`

```python
# steps/feature_engineering/query_data_warehouse.py
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

Three Mongo queries run concurrently; wall-clock is the slowest query, not the sum.

```
Sequential:  ── articles ──►── posts ──►── repos ──►     = a + p + r
Parallel:    ── articles ──►
             ── posts ─────►                             = max(a, p, r)
             ── repos ─────►
```

**Failure mode**: a failing query is swallowed and replaced with `[]`, so the pipeline continues with partial data. That is deliberate (one flaky collection should not kill the whole run) but it means "0 documents" can mean "the query failed", not "the author has none". The `logger.exception` is the only signal.

### 6. Parallel retrieval - `ContextRetriever.search`

```python
# llm_engineering/application/rag/retriever.py
        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in n_generated_queries]

            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))
```

Each query variant searches in parallel, and **within a variant the three collections are searched serially**:

```python
    def _search(self, query: Query, k: int = 3) -> list[EmbeddedChunk]:
        assert k >= 3, "k should be >= 3"
        ...
            return data_category_odm.search(
                query_vector=embedded_query.embedding,
                limit=k // 3,          # 3 collections × k//3 ≈ k
                query_filter=query_filter,
            )
```

The `k // 3` is the per-collection budget: posts, articles, and repositories each return an equal share, so the union is roughly `k`. Increasing `expand_to_n_queries` (default 3) multiplies parallel work but also latency and cost.

**Integer-truncation edge case**: `k=3` gives `3 // 3 = 1` per collection, so 3 total before dedup. `k=4` gives `4 // 3 = 1`, also 3 — bumping `k` by one does nothing until `k=6`. If you need exactly `k` candidates, change the divisor logic, not just `k`.

---

## 🔬 Deep Dive: The Batch-Size Tradeoff

```
Throughput
    ▲
    │            ╭───────  GPU utilization saturates
    │        ╭───╯
    │    ╭───╯
    │╭───╯
    └────────────────────────► batch size
        │           │
     latency     memory
     floor       ceiling (OOM)
```

For GPU embedding/training, increasing batch size improves throughput until the GPU saturates; beyond that, only memory and latency grow. For API calls, increasing concurrency improves throughput until rate limits trigger retries, which can make it *slower*.

The two regimes look identical on a throughput chart but the operating advice is opposite:

| Regime | Bottleneck | More batch does | Correct fix |
|--------|-----------|-----------------|-------------|
| GPU compute | SM occupancy | raises throughput until saturation | increase batch up to VRAM |
| API concurrency | server rate limit | triggers 429 + backoff (slower) | cap concurrency, add jitter |
| Network I/O (Qdrant) | round-trips | fewer round-trips | batch until payload/timeout limit |
| CPU preprocessing | single core | no change | parallelize or vectorize |

**On the RTX 5000 (16 GB, Turing)**:
- **No bfloat16** - `is_bfloat16_supported()` returns `False`, so training uses fp16. fp16 is fine but has a narrower dynamic range; keep the learning rate reasonable.
- **Gradient checkpointing** is not enabled in this code. It trades ~20-30% extra compute for activation memory, and would help fit larger effective batches. It is the standard next lever when OOM happens at a *useful* batch size.
- **`RAG_MODEL_DEVICE=cpu`** keeps embedding/reranking off the GPU so training has the whole card. Switch to `cuda` only when not training. The default in `settings.py` is `cpu` for exactly this reason.
- **Turing has no Flash Attention 2**; vLLM falls back to its paged-attention kernel. Expect lower throughput than an Ada/Hopper card — do not size benchmarks by comparing to A100 numbers.

---

## 📁 Caching Layers

### 1. ZenML step cache

Every pipeline run has `enable_cache` (default `True`; `--no-cache` disables it and every `poe` task in `pyproject.toml` passes `--no-cache`). Unchanged steps reuse outputs, which is the biggest saving during iterative development.

```python
    pipeline_args = {"enable_cache": not no_cache}
```

**Why the repo disables it everywhere**: during development you change code or data constantly, and a stale cache produces a "finished in 3 seconds" run that did nothing. The cost of a cache miss is re-running; the cost of a false hit is a silent wrong result. For reproducible benchmark runs, `--no-cache` is the safe default.

### 2. `cached_property` for model metadata

```python
# llm_engineering/application/networks/embeddings.py
    @cached_property
    def embedding_size(self) -> int:
        dummy_embedding = self._model.encode("")
        return dummy_embedding.shape[0]
```

The model is probed once; repeated collection sizing calls are free. `cached_property` stores the value on the instance dict after first access, so it is only paid once per process — and because the model is a singleton, once per process *globally*.

### 3. Singleton model loading

```python
# llm_engineering/application/networks/base.py
class SingletonMeta(type):
    _instances: ClassVar = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
```

`EmbeddingModelSingleton` and `CrossEncoderModelSingleton` load once per process. This is the difference between a 5-second and a 50-second startup across a batch of operations.

**Caveat**: the lock is held only during construction, but a *double-checked* pattern would be faster on the hot path. More importantly, the singleton key is the class, not `(class, model_id, device)`. If you construct `EmbeddingModelSingleton("model-A")` and later `EmbeddingModelSingleton("model-B")`, you get the *first* instance back and `model-B` is silently ignored. The call in `embeddings.py` defaults to the settings values, so in practice there is one model per process — but this is a latent footgun for anyone parameterizing the model.

### 4. Application-level semantic caching (not implemented)

A production RAG service often caches `query → answer` with a TTL. This project does not, so repeated identical queries re-run the full path (self-query, query expansion, ~9 vector searches, rerank, LLM call). Adding a `TTLCache` in front of `rag()` is the highest-value next optimization (see the exercise).

---

## 📁 Quantization Across the Lifecycle

| Stage | Technique | Where | Effect |
|-------|-----------|-------|--------|
| Training | 4-bit NF4 base + LoRA | `load_in_4bit=True` | ~4× smaller base weights |
| Training optimizer | 8-bit AdamW | `optim="adamw_8bit"` | smaller optimizer state |
| Serving | 8-bit bitsandbytes | `HF_MODEL_QUANTIZE="bitsandbytes"` | fits endpoint GPU |
| Evaluation | vLLM (fp16/tp) | `vllm.LLM` | high-throughput generation |
| Embeddings | fp32/fp16 CPU | `RAG_MODEL_DEVICE=cpu` | small model, no GPU contention |

Quantization reduces memory at some quality cost. Match the technique to the constraint: 4-bit for fitting 8B training, 8-bit for serving with a quality margin.

A LPU-style decomposition of VRAM helps decide what to quantize:

```
VRAM = weights + KV cache + activations + framework overhead
       │          │          │              │
       4-bit for   grows with context,  grows with   CUDA context,
       the big     batch, and layers     batch        ~0.5-1 GB fixed
       lever
```

For serving, the **KV cache** is often the limit, not weights: it scales linearly with `MAX_TOTAL_TOKENS`. The `settings.py` values `MAX_INPUT_LENGTH=2048`, `MAX_TOTAL_TOKENS=4096`, `MAX_BATCH_TOTAL_TOKENS=4096` bound it.

---

## 🛠️ Hands-On: Profile Before Optimizing

### Step 1: Time the embedding step

```python
import time
from llm_engineering.application.networks import EmbeddingModelSingleton

model = EmbeddingModelSingleton()
texts = ["Sample text " * 50] * 64

for bs in (1, 10, 50):
    start = time.perf_counter()
    for i in range(0, len(texts), bs):
        model(texts[i : i + bs], to_list=True)
    print(f"batch={bs:3d}  {time.perf_counter() - start:.3f}s")
```

Run this twice: once with `RAG_MODEL_DEVICE=cpu` and once with `cuda`. The CPU/GPU crossover point tells you which batches are worth moving to the GPU.

### Step 2: Measure peak VRAM (GPU)

```python
import torch
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
torch.cuda.reset_peak_memory_stats()
m.encode([""] * 256)
print(torch.cuda.max_memory_allocated() / 1024**2, "MB")
```

`reset_peak_memory_stats` before the measured call and `max_memory_allocated` after isolate the peak of *that* operation, not the process high-water mark.

### Step 3: Time retrieval

```python
import time
from llm_engineering.application.rag.retriever import ContextRetriever

r = ContextRetriever(mock=False)
start = time.perf_counter()
_ = r.search("How does RAG work?", k=3)
print(f"retrieval: {time.perf_counter() - start:.2f}s")
```

### Step 4: Break down latency

Use the Opik trace (Session 7.2) to see where the seconds go: SelfQuery, QueryExpansion, search, rerank, or generation. Each is a span, so the waterfall shows the real bottleneck rather than the step you assumed.

### Step 5: Sweep `k` and record the cost curve

```python
import time
from llm_engineering.application.rag.retriever import ContextRetriever

r = ContextRetriever(mock=False)
for k in (3, 6, 9, 12):
    start = time.perf_counter()
    docs = r.search("How does RAG work?", k=k)
    print(f"k={k:2d}  docs={len(docs):2d}  {time.perf_counter() - start:.2f}s")
```

Expect a superlinear cost: more candidates mean more reranking pairs and a longer generation prompt.

---

## 📝 Exercise 1: Add Semantic Answer Caching

### Task

Cache RAG answers to eliminate repeated work.

```python
from cachetools import TTLCache, cached

# in the API module
_rag_cache = TTLCache(maxsize=1000, ttl=3600)


@cached(cache=_rag_cache)
def rag(query: str) -> str: ...
```

1. Add `cachetools` to dependencies.
2. Wrap `rag()`.
3. Measure the latency of a repeated query before and after.
4. Discuss invalidation: when the corpus changes, cached answers go stale. Add a version key (for example a collection hash) to the cache key.

**Goal**: Experience the biggest single latency win in a RAG service — and its correctness cost.

> Note: `@cached` keys on the function arguments only. `rag(query)` is keyed by the query string, so two users asking the same question share an answer regardless of the `author_id` embedded in the query. For a multi-tenant service, include the tenant/author in the key or you leak one user's context to another. This is a correctness *and* security issue, not just a stale-cache issue.

---

## 📝 Exercise 2: Make `compute_num_tokens` Cheap

### Task

`compute_num_tokens` reloads a tokenizer on **every call**:

```python
# llm_engineering/application/utils/misc.py
def compute_num_tokens(text: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(settings.HF_MODEL_ID)
    return len(tokenizer.encode(text, add_special_tokens=False))
```

The RAG endpoint calls it three times per request (`query_tokens`, `context_tokens`, `answer_tokens`), each triggering a `from_pretrained` lookup.

1. Cache the tokenizer at module scope or with `functools.lru_cache`.
2. Benchmark N=100 calls before and after.
3. Explain why Hugging Face's internal cache makes this cheaper than a true download but still not free (object construction, lock contention, and repeated cache lookups on the hot path).

**Goal**: Learn that "the library caches it" is not the same as "this is free on the hot path". The fix is a two-line `lru_cache`.

---

## 🔬 Deep Dive: A Latency Budget

Optimization without a budget is guessing. Write the budget first, then find the step that exceeds it. A plausible budget for the `/rag` endpoint:

```
                target p50    target p95    where it is spent
self-query       ~150 ms       ~300 ms      one LLM call (gpt-4o-mini)
query expansion  ~400 ms       ~900 ms      one LLM call producing N queries
vector search    ~40 ms        ~90 ms       3 collections × N variants
rerank           ~60 ms        ~150 ms      cross-encoder over candidates
generation       ~1.2 s        ~2.5 s       SageMaker endpoint, MAX_NEW_TOKENS
token counting   ~15 ms        ~40 ms       3 × AutoTokenizer lookups (fixable)
─────────────────────────────────────────
total            ~1.9 s        ~4.0 s
```

Three observations follow:
1. **Generation dominates.** Shrinking `MAX_NEW_TOKENS_INFERENCE` (default 150) has the largest single effect.
2. **Two LLM calls are hidden overhead** in retrieval (self-query + expansion). Caching retrieval for repeated queries removes both.
3. **Token counting is small but free to fix** (Exercise 2); it is a constant 15-40 ms you should not pay.

Use the Opik waterfall (Session 7.2) to replace these *estimates* with *measurements*. The budget tells you where to look; only measurement tells you where the time actually goes.

---

## 📁 A Measurement Harness

Ad-hoc timing is noisy. A small harness makes comparisons meaningful:

```python
import statistics
import time


def bench(fn, *, runs: int = 5, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "max": max(samples),
    }
```

Why the shape matters:
- **Warmup** removes first-call costs (model load, tokenizer cache, lazy CUDA init) that are not present in steady state.
- **Median** is robust to a single slow sample; the mean is not.
- **Max** exposes the tail, which is what users complain about.

Apply it to the exact functions you changed, before and after, on the same machine and inputs.

---

## 📁 Device Placement Decision

`RAG_MODEL_DEVICE` (default `cpu` in `settings.py`) decides where embeddings and reranking run. The choice is a scheduling problem, not a speed problem.

```
                     GPU (16 GB)                     CPU (i7-9850H, 6c/12t)
                     ───────────                     ──────────────────────
embeddings           fast, but competes with        slower, leaves GPU free
                     any training job

reranking            fast, same contention           slower, free GPU

training             needs the whole card            infeasible

Decision matrix:
  training in progress?              → RAG_MODEL_DEVICE=cpu
  serving only, latency-sensitive?   → cuda (if it fits)
  serving only, batch/offline?       → cpu is fine and simpler
```

**Why `cpu` is the default**: the project is used for training and RAG on the same workstation. A CPU embedding model (all-MiniLM-L6-v2 is ~80 MB) is small enough that CPU latency is acceptable, and it guarantees the trainer owns the GPU. The tradeoff is a slower retrieval path; the escape hatch is `cuda` when training is idle.

**VRAM check before choosing `cuda`**:

```python
import torch
free, total = torch.cuda.mem_get_info()
print(f"free {free/1024**2:.0f} MB / total {total/1024**2:.0f} MB")
```

Only move the RAG models to GPU if the free memory comfortably exceeds the models' working set plus the KV/activation headroom of whatever else runs.

---

## 📁 Profiling Methodology

Three tools, escalating in precision:

1. **Wall-clock + `bench` harness** (above): answers "is it faster?".
2. **`cProfile`**: answers "which function is hot?".

```python
import cProfile

cProfile.run("retriever.search('How does RAG work?', k=3)", sort="cumulative")
```

3. **`torch.profiler`**: answers "is the GPU idle or saturated?" — the definitive test for batching decisions.

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    model(texts, to_list=True)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Reading the result**: dual-LLM inference is a proven technique where one model generates and a smaller model evaluates, but for *performance* profiling the rule is simpler — if `cuda_time_total` is a small fraction of wall-clock, the GPU is starved and the bottleneck is Python, I/O, or the CPU fallback (a common cause: `RAG_MODEL_DEVICE=cpu` while you believed it was `cuda`).

---

## 📊 Worked Example: The Cost of Retrieval Expansion

`ContextRetriever.search` defaults to `expand_to_n_queries=3` and `k=3`. Trace the work:

```
base query
  └─ SelfQuery: 1 LLM call            → adds author_full_name filter
  └─ QueryExpansion: 1 LLM call       → produces 3 query variants
       └─ 3 variants (parallel)
            └─ each _search: 3 collections × (k // 3 = 1) = 3 candidates
                 → 9 candidates total (before dedup)
  └─ Rerank: 1 cross-encoder call over ~9 (query, chunk) pairs
  └─ keep_top_k = k = 3
```

Costs that scale with `expand_to_n_queries`:
- LLM tokens (one call, N outputs),
- number of parallel searches (N),
- dedup set size (up to N × k),
- reranking pairs (up to N × k).

So doubling to 6 variants roughly doubles retrieval work *and* may improve recall. Whether it is worth it is an empirical question — measure recall (Session 7.4) against latency (here). This is the definition of a tradeoff: a measurable gain for a measurable cost.

---

## 🐛 Common Pitfalls

- **Premature optimization**: raising batch sizes without measuring often causes OOM or rate-limit retries that are slower. Measure first.
- **GPU contention**: embedding and reranking on GPU during training starves the trainer. Keep `RAG_MODEL_DEVICE=cpu` while training.
- **Cache staleness in ZenML**: a "finished quickly" pipeline run may have skipped steps. Use `--no-cache` after data changes (the repo does this by default).
- **bfloat16 assumption**: Turing does not support it; forcing bf16 fails or falls back silently. Use `is_bfloat16_supported()`.
- **Unbounded `k`**: large `k` increases reranking and generation cost superlinearly. Start small.
- **The `k // 3` plateau**: due to integer truncation, increments of `k` below a multiple of 3 do not change the number of retrieved candidates.
- **Singleton keyed by class only**: constructing the singleton with a different `model_id` returns the first instance and ignores your argument.
- **Tokenizer reload per token count**: `compute_num_tokens` calls `AutoTokenizer.from_pretrained` every invocation; on the hot path that is three reloads per request.
- **Per-batch OpenAI client**: `evaluate_batch` builds a new client for every batch of 5.
- **Swallowed query failures**: `fetch_all_data` turns a failed Mongo query into `[]`, so "0 documents" is ambiguous.
- **Measuring wall-clock of a cached run**: a ZenML cache hit makes a step look instant when it did no work.

---

## 🎓 Knowledge Check

1. **What are the four batching values in the pipeline?**
   - Answer: embed=10, load-to-vector=4, LLM generate=24, evaluation batch=5 with 10 threads.

2. **What is the first knob to lower on OpenAI 429 errors?**
   - Answer: the generation batch size (24).

3. **Why is `embedding_size` a `cached_property`?**
   - Answer: It requires a model probe; caching avoids repeating it across calls.

4. **Why keep `RAG_MODEL_DEVICE=cpu` during training?**
   - Answer: To avoid VRAM contention with the training job.

5. **What precision does the RTX 5000 use for training, and why?**
   - Answer: fp16, because Turing lacks bfloat16 support.

6. **What is the highest-value unimplemented optimization for the RAG service?**
   - Answer: Semantic answer caching with a TTL and a corpus-version key.

7. **How many documents does `ContextRetriever._search` fetch per collection, and why?**
   - Answer: `k // 3`, so three collections sum to roughly `k`. Integer truncation means `k=3` and `k=4` both fetch 1 per collection.

8. **Why is the retrieval cost superlinear in `k`?**
   - Answer: More candidates mean more reranking pairs (cross-encoder work) and a longer generation context, both of which grow faster than linearly.

9. **What does `utils.misc.batch` return, and why does that matter?**
   - Answer: A generator, so only one batch is in memory at a time. Materializing it as a list would defeat the memory saving.

10. **Why does the singleton pattern in `base.py` matter for startup time?**
    - Answer: Models load once per process; without it, each operation would reload the transformer (seconds to tens of seconds).

11. **What is the risk of keying a RAG answer cache on `query` alone?**
    - Answer: It ignores the per-author filter, so one user can receive another user's context. Include tenant/author in the key.

12. **Why is `compute_num_tokens` a hidden cost?**
    - Answer: It calls `AutoTokenizer.from_pretrained` on every invocation; the RAG endpoint calls it three times per request.

13. **What failure mode does `fetch_all_data` hide?**
    - Answer: A failed Mongo query returns `[]`, indistinguishable from "no documents" except for a log line.

14. **When is gradient checkpointing the right next lever?**
    - Answer: When you need a larger effective batch than VRAM allows at a useful batch size; it trades compute for activation memory.

15. **What does `--no-cache` protect against in development?**
    - Answer: A stale ZenML step output that makes a run look successful while skipping the work you changed.

---

## 📖 Glossary

- **Batch**: A group of items processed together in one forward pass or API call.
- **Throughput**: Items processed per unit time.
- **Latency**: Time to complete one request.
- **Reranking**: Cross-encoder scoring of (query, document) pairs to reorder retrieval results; cost scales with candidate count.
- **Singleton**: A class with at most one instance; here, one model per process.
- **`cached_property`**: A property computed once and stored on the instance.
- **Quantization**: Representing weights in fewer bits (4/8) to cut memory.
- **KV cache**: Cached key/value tensors for already-generated tokens; grows with context and batch.
- **Gradient checkpointing**: Recompute activations in the backward pass to save memory at extra compute.
- **Rate limit (429)**: Server rejection of excess requests; the SDK retries, appearing as slowness.
- **Wall-clock**: Real elapsed time, as opposed to CPU time.
- **Roofline**: The model of whether a workload is compute- or memory-bandwidth-bound.

---

## 🔗 Next Session

**Session 9.3**: Security Best Practices

We harden secrets, the API, and the local stack.

---

## 📚 Additional Resources

- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [vLLM](https://docs.vllm.ai/)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes)
- [cachetools](https://cachetools.readthedocs.io/)
- [SentenceTransformers: efficiency and batching](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html)
- [OpenAI: rate limits guide](https://platform.openai.com/docs/guides/rate-limits)

---

## 🔎 References

- **Book**: *LLM Engineer's Handbook* — Chapter 4 (RAG, retrieval and reranking) and Chapter 8 (inference optimization and evaluation).
- **Repo**: `steps/feature_engineering/*.py`, `llm_engineering/application/networks/embeddings.py`, `llm_engineering/application/rag/retriever.py`, `llm_engineering/application/utils/misc.py`, `pipelines/feature_engineering.py`, `llm_engineering/model/evaluation/evaluate.py`.
- **Related sessions**: [`session_2.3_feature_engineering.md`](session_2.3_feature_engineering.md), [`session_4.2_embedding_models.md`](session_4.2_embedding_models.md), [`session_6.2_rag_inference_flow.md`](session_6.2_rag_inference_flow.md), [`session_7.2_opik_monitoring.md`](session_7.2_opik_monitoring.md), [`session_8.4_inference_optimization.md`](session_8.4_inference_optimization.md).
- **Curriculum**: [`../CURRICULUM.md`](../CURRICULUM.md).

---

**Estimated Time**: 4-5 hours

**Prerequisites**: [Session 2.3](session_2.3_feature_engineering.md), [Session 5.1](session_5.1_sft.md), [Session 6.2](session_6.2_rag_inference_flow.md), [Session 8.3](session_8.3_zenml.md)

**Outcome**: You can identify the performance levers in the project, profile a workload, and apply the right optimization at the right layer.
