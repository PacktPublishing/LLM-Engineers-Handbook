# Session 4.3: Batch vs Streaming & Change Data Capture

> Book reference: Chapter 4, *RAG Feature Pipeline* (pages 159-168) — the architecture half.
> Repo: `pipelines/feature_engineering.py`, `steps/feature_engineering/query_data_warehouse.py`, `steps/feature_engineering/load_to_vector_db.py`.

## 🎯 Learning Objectives

By the end of this session, you will:
- Define a batch pipeline and contrast it with streaming on the criteria that matter.
- Explain why the LLM Twin deliberately chose a **batch + pull CDC** architecture.
- Name the five core RAG feature-pipeline steps and map each to a repo file.
- Explain change data capture (CDC), its push/pull directions, and its three change-detection patterns.
- Justify why the project stores **two snapshots** of the logical feature store.
- Plan a timestamp-based CDC upgrade and articulate what it cannot capture (hard deletes).
- Reason about continuous training and where orchestration hooks in.

> Sessions 2.2 and 2.3 cover the hands-on cleaning/chunking/embedding. This session covers the **pipeline-design reasoning**: why this shape, what it costs, and when to change it.

---

## 🏗️ Architecture Overview

### The RAG feature pipeline in one picture

```
Source (MongoDB data warehouse)
      │  extract  (query_data_warehouse)
      ▼
┌──────────────────────────────────────────────────────────────┐
│  RAG feature pipeline (5 steps)                                │
│  extract → clean → chunk → embed → load                        │
└──────────────────────────────────────────────────────────────┘
      │                                   │
      ▼                                   ▼
 cleaned docs (Qdrant, no vector)   embedded chunks (Qdrant, vectors)
   → for fine-tuning                    → for RAG
      │                                   │
      └────── two snapshots of the logical feature store ───────┘

Sync strategy:  batch + pull CDC  (simple, correct for small data)
Orchestration:  ZenML (scheduled / manual / after ETL)  → continuous training
```

### Batch vs streaming in one diagram

```
BATCH                                    STREAMING
┌──────────────┐   accumulate           ┌──────────────┐   one event at a time
│ data source  │◄──────────┐            │ event bus    │──┐
└──────────────┘           │            │ (Kafka,      │  │
        │ every hour/day   │            │  Redpanda)   │  │
        ▼                  │            └──────────────┘  │
┌──────────────┐   bulk    │                   │          │
│ processing   │───────────┘                   ▼          ▼
└──────────────┘                        ┌──────────────┐
        │ write                          │ streaming    │
        ▼                                │ engine       │
┌──────────────┐                        │ (Flink,      │
│ warehouse /  │                        │  Bytewax)    │
│ feature store│                        └──────────────┘
└──────────────┘                               │ low latency
                                               ▼
                                        ┌──────────────┐
                                        │ online store │
                                        └──────────────┘
simpler, cheaper, higher latency         complex, costlier, low latency
```

### The decision heuristic

Pick batch unless a **freshness requirement forces streaming**. The LLM Twin's freshness requirement is "a few minutes is fine", so batch wins. The book states the rule plainly: start batch, move to streaming only when a requirement forces it.

---

## 📁 Key Files Explained

This session's reasoning is grounded in three repo files. Read them together to see the batch design in code.

### `pipelines/feature_engineering.py` — the DAG

```python
# pipelines/feature_engineering.py
@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    return [last_step_1.invocation_id, last_step_2.invocation_id]
```

The `wait_for` parameter is the orchestration seam: it lets `end_to_end_data` gate this pipeline on the ETL completion. The return value chains dataset generation.

### `steps/feature_engineering/query_data_warehouse.py` — the extract step

```python
# steps/feature_engineering/query_data_warehouse.py (excerpt)
@step
def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = []
    authors = []
    for author_full_name in author_full_names:
        first_name, last_name = utils.split_user_full_name(author_full_name)
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)

        results = fetch_all_data(user)
        user_documents = [doc for query_result in results.values() for doc in query_result]

        documents.extend(user_documents)
    ...
    return documents
```

This is the full-scan extract. `fetch_all_data` runs three parallel `bulk_find` calls (articles, posts, repositories) with no time filter — the exact behaviour this session asks you to upgrade.

### `steps/feature_engineering/load_to_vector_db.py` — the load step

```python
# steps/feature_engineering/load_to_vector_db.py (excerpt)
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

`group_by_class` sends cleaned docs to their unindexed collections and embedded chunks to their vector collections — the two snapshots in one step. Because `bulk_insert` upserts on deterministic ids, re-running the pipeline is safe.

---

## 📁 Batch Pipeline Anatomy (mapped to the repo)

The book's batch ETL describes three phases. Here is how each maps to real code, so the architecture stops being abstract.

| Phase | What it means | Repo artifact |
|-------|---------------|---------------|
| Collect | accumulate raw records | `ArticleDocument.save()` during ETL (Session 2.1) |
| Process | schedule bulk transform | `feature_engineering` pipeline steps |
| Load | write to warehouse/feature store | `load_to_vector_db` → Qdrant upsert |

The processing phase is itself a small DAG:

```
query_data_warehouse                       [extract]
        │  Mongo full scan per author
        ▼
clean_documents                            [transform: clean]
        │
        ├──────────────────────► load_to_vector_db  (cleaned, no vectors)   [load A]
        ▼
chunk_and_embed                            [transform: chunk + embed]
        │
        ▼
load_to_vector_db                          [load B]
```

Two design facts fall out of this shape:

1. **Clean is the fork point.** One cleaned snapshot serves two consumers: fine-tuning (Load A) and RAG (Load B). That is the two-snapshot design in code, not just prose.
2. **Batch granularity is the whole set.** `query_data_warehouse` has no `since` filter, so every run re-cleans and re-embeds everything the author has ever produced. Correct at hundreds of docs, wasteful at millions. This is the exact seam where CDC upgrades bolt on.

### Why the full scan is acceptable today

| Quantity | Rough order | Full-scan cost |
|----------|-------------|----------------|
| Authors | 2 | negligible |
| Docs per author | tens to low hundreds | negligible |
| Chunks after split | hundreds to low thousands | CPU embedding time dominates |
| Embedding model | 384-dim, 256-token | milliseconds per batch of 10 |

The bottleneck is **embedding compute**, not the Mongo read. That is why the project optimizes the embedding batch (size 10) and not the extract query. Adding a `last_updated` filter would help only once the read itself becomes material — which is precisely the CDC trigger.

---

## 📁 The Streaming Stack, Layer by Layer

If a freshness requirement ever forces streaming, this is what you would assemble and why each layer exists.

```
Producers                Event platform            Streaming engine          Sinks
─────────                ──────────────            ────────────────          ─────
Medium crawler ──┐                                 window / join /       ┌──► Qdrant
GitHub crawler ──┼──►  Kafka topic (durable,   ──► map / filter      ──►┼──► MongoDB
user events    ──┘     partitioned, replayable)    (Flink / Bytewax)     └──► warehouse
                                                        │
                                                        ▼
                                                  checkpoints / state
```

| Layer | Responsibility | Example tools | Failure it absorbs |
|-------|----------------|---------------|--------------------|
| Producer | emit change events | crawlers, DB triggers | n/a |
| Event platform | durable ordered log | Kafka, Redpanda | consumer downtime (replay) |
| Streaming engine | stateful processing | Flink, Bytewax | restarts via checkpoints |
| Sink | online/analytical stores | Qdrant, Mongo, warehouse | backpressure |

**What streaming buys**: sub-second to sub-minute freshness and incremental processing (only changed records).

**What streaming costs** (the "roughly twice as complex" claim):
- A new always-on platform to run, monitor, and upgrade.
- Exactly-once semantics or idempotent sinks to reason about (the deterministic md5 chunk IDs from Session 2.2 would help here).
- Backfill/replay strategy for when logic changes.
- Local development becomes harder: you cannot "just run the pipeline".

**The engineering rule**: do not pay for streaming until a *measured* freshness requirement demands it. The LLM Twin's requirement is "a few minutes", which batch satisfies at a fraction of the operational cost.

---

## 📁 Batch Pipelines

A **batch pipeline** collects, processes, and stores data in intervals and larger volumes.

1. **Data collection**: accumulate data from databases, logs, files.
2. **Scheduled processing**: hourly or daily bulk cleansing, transformation, aggregation.
3. **Data loading**: write to a warehouse, lake, or feature store.

**Advantages**:
- Efficient on large volumes (parallelism, bulk I/O).
- Supports complex transforms and aggregations that streaming engines make awkward.
- Simpler to build, test, and operate.

**Costs**:
- Latency: data is only as fresh as the last run.
- Repeated full work if not incremental (the LLM Twin currently re-pulls everything).

---

## 📁 Batch vs Streaming

| Aspect | Batch | Streaming |
|--------|-------|-----------|
| Schedule | regular intervals (minute/hour/day) | continuous, minimal latency |
| Unit of work | large volumes, parallel | single data points, immediate |
| Complexity | complex transforms/aggregations OK | optimized for high-velocity, low-latency |
| Typical use cases | warehousing, reporting, ETL, feature pipelines | real-time analytics, fraud, event-driven |
| System complexity | simpler | more complex (fault tolerance, extra tooling) |
| Cost to operate | lower | higher |

**Streaming examples**: a TikTok recommender (interests shift within minutes), Stripe/PayPal fraud detection (milliseconds matter), high-frequency trading.

**Batch examples**: offline e-commerce/streaming recommenders (behavior changes slowly), ETL/analytics pipelines.

**Core elements of streaming**: a distributed event platform (Apache **Kafka**, **Redpanda**) plus a streaming engine (Apache **Flink**, **Bytewax**). Message queues (RabbitMQ) simplify the event store but are not a full streaming platform.

### Why the LLM Twin uses batch

1. **No immediate processing needed**: a few minutes of delay for syncing the warehouse and feature store is acceptable. The pipeline could even run every minute.
2. **Small data**: the warehouse holds thousands, not millions, of records, so a full scan is cheap.
3. **Simplicity**: streaming is roughly twice as complex and more expensive to operate.

**Tooling on the spectrum**: for small-data batch the project uses **vanilla Python + LangChain + Sentence Transformers + Unstructured**, not a streaming stack.

---

## 📁 The Five Core RAG Feature Steps

1. **Extraction**: pull the latest articles, repositories, and posts from the MongoDB warehouse.
2. **Cleaning**: standardize text, remove duplicates, normalize, drop noise. The book calls this "more art than science"; the repo implementation is the minimal `clean_text` (Session 2.2).
3. **Chunking**: category-specific strategies (broader for code, sentence-level for prose) and never exceeding the embedding model's max input size.
4. **Embedding**: pass each chunk through the embedding model (a small SentenceTransformer by default).
5. **Loading**: combine the vector with metadata (author, document id, content, URL, platform) into a Qdrant point. Also push **cleaned** documents (no vectors) to Qdrant, whose metadata index behaves like a NoSQL store.

**In the repo**: this maps exactly to `pipelines/feature_engineering.py`:

```python
# pipelines/feature_engineering.py
@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    return [last_step_1.invocation_id, last_step_2.invocation_id]
```

| Step | Repo location |
|------|---------------|
| extract | `steps/feature_engineering/query_data_warehouse.py` |
| clean | `steps/feature_engineering/clean.py` |
| chunk | `steps/feature_engineering/rag.py` (via `ChunkingDispatcher`) |
| embed | `steps/feature_engineering/rag.py` (via `EmbeddingDispatcher`) |
| load | `steps/feature_engineering/load_to_vector_db.py` |

> Note: the book's prose mentions an `all-mpnet-base-v2` embedding model, while the repo `settings.py` defaults to `sentence-transformers/all-MiniLM-L6-v2`. Both are CPU-friendly; the id is configurable via `TEXT_EMBEDDING_MODEL_ID`.

---

## 📁 Change Data Capture (CDC)

CDC keeps two or more data stores in sync with minimal compute and I/O by capturing CRUD operations at the source and replicating them (optionally with preprocessing).

### Why the naive approach breaks at scale

The project's current approach pulls **everything**:

```python
# steps/feature_engineering/query_data_warehouse.py (excerpt)
def __fetch_articles(user_id) -> list[NoSQLBaseDocument]:
    return ArticleDocument.bulk_find(author_id=user_id)
```

This is fine for hundreds of documents. It breaks down when:
- Data grows to millions of records and a full scan is expensive.
- A record is **deleted** at the source and you need that reflected in the feature store.
- You want to process only new or updated items, not redo everything.

### Push vs pull

```
PUSH                                          PULL
source ──change──► target                      target ──"what changed?"──► source
   near-instant                                   periodic, lighter on the source
   target down => data loss                       still delayed
   buffer with a queue to avoid loss              still wants a queue for reliability
```

| Direction | Latency | Source load | Failure mode | Mitigation |
|-----------|---------|-------------|--------------|------------|
| Push | near-instant | higher (must notify) | target down loses changes | durable queue |
| Pull | periodic delay | lower | stale until next pull | watermark + queue |

### Three change-detection patterns

| Pattern | How | Pros | Cons |
|---------|-----|------|------|
| **Timestamp** | a `LAST_MODIFIED` column; query since last check | simple | misses deletes; still full-table scans without an index |
| **Trigger** | DB triggers write to an event table on INSERT/UPDATE/DELETE | complete change tracking | adds write overhead to the source DB |
| **Log-based** | read the DB transaction log | low impact, low latency, captures all CRUD | vendor-specific log formats, complex |

The industry-optimal choice is **log-based** (e.g. Debezium over the WAL/oplog), but it needs a queue and a streaming pipeline. The LLM Twin deliberately uses a **pull-all** approach for simplicity. A pull **timestamp** strategy is the natural first upgrade, and it requires adding a `last_updated` field — which the current schema lacks.

### Worked example: a timestamp CDC timeline

Suppose the feature store syncs every 15 minutes and stores a watermark `W` (the last successful sync time, taken from the source clock).

```
t=00:00  sync runs. watermark W = 00:00. pulls docs where last_updated > W.
t=00:03  article A created  (last_updated = 00:03)
t=00:07  article B created  (last_updated = 00:07)
t=00:09  article A edited   (last_updated = 00:09)   ← update overwrites row
t=00:12  article C deleted  (no row remains)         ← DELETE IS INVISIBLE
t=00:15  sync runs. W = 00:00 → pulls A (as of 00:09) and B. C never appears.
         feature store now: A, B.  watermark W = 00:15.
```

What the batch pull-all design would do instead: at `t=00:15` it re-reads **A, B** and recomputes their chunks and embeddings. It is correct, but it redoes work for unchanged documents, and it *also* misses the delete of C — because C is gone from the source, so `bulk_find` simply returns one fewer row and nothing marks the removal in the feature store.

**Lessons**:

- Timestamps fix the *wasted work* problem but not the *deletes* problem.
- A delete can only be detected by comparing the current source set against the feature-store set (reconciliation) or by reading an operation log (log-based CDC).
- The watermark must use source time and move strictly forward. If you advance `W` to wall-clock `00:15` but the source clock lags, you skip rows.

A minimal reconciliation sketch:

```python
# source ids for an author
source_ids = {str(d.id) for d in ArticleDocument.bulk_find(author_id=user_id)}
# feature-store ids derived from the cleaned snapshot
stored_ids = {
    str(c.document_id)
    for c in CleanedArticleDocument.bulk_find(author_id=user_id)[0]
}
deleted = stored_ids - source_ids          # documents to purge downstream
```

### Push CDC in practice

Push means the source (or a change-log listener) drives replication:

```
DB write ──trigger/log──► change event ──► queue ──► consumer ──► feature store
                                            (Kafka/RabbitMQ)
```

The queue is what makes push safe: if the consumer is down, events buffer and replay. Without a queue, a single unavailable target loses the change permanently. Log-based CDC (Debezium reading the DB transaction log) is the industry-standard push implementation because it captures INSERT/UPDATE/DELETE without adding application write overhead.

### Freshness vs cost

Choosing a sync strategy is buying freshness with operational cost. The curve is not linear.

| Strategy | Freshness | Added infrastructure | Delete-safe | Effort to adopt |
|----------|-----------|----------------------|-------------|-----------------|
| Pull-all (today) | run interval | none | no | none |
| Pull-timestamp | run interval | a `last_updated` field + index + watermark | no | low |
| Push via queue | seconds | broker + producer hooks | depends on source | medium |
| Log-based CDC | sub-second | broker + CDC connector + streaming engine | yes | high |

The project sits at row 1 and the natural next step is row 2. Rows 3-4 are only justified when a product requirement (real-time scoring, strict deletion compliance) materially needs them.

### The CDC decision tree

```
Do you need sub-minute freshness?
├── No  ──► BATCH
│           Is a full scan cheap (small data)?
│           ├── Yes ──► pull-all  (what the repo does today)
│           └── No  ──► pull timestamp  (add last_updated)
└── Yes ──► STREAMING
            Do you need complete CRUD including deletes?
            ├── Yes ──► log-based CDC (Debezium/Kafka) + streaming engine
            └── No  ──► push/pull timestamp over a queue
```

---

## 📁 Why Two Snapshots?

The logical feature store holds:

1. **Cleaned documents** (no embeddings) → for **fine-tuning**.
2. **Embedded chunks** → for **RAG**.

Reasons:

- Features should be read **only from the feature store** at training and inference time, keeping the design consistent.
- Processing data for a specific use case **inside the shared data warehouse is an antipattern**: other teams and use cases need different processing. Keeping the warehouse generic and pushing application-specific modeling downstream avoids a "spaghetti warehouse".
- A vector DB's metadata index doubles as a NoSQL store, so **cleaned** documents can live in Qdrant without vectors.

```
Logical feature store
├── Qdrant (online serving)     ── embedded chunks WITH vectors
│                               └─ cleaned docs WITHOUT vectors
└── ZenML artifacts (offline)   ── training datasets (later sessions)
```

**In the repo**: `CleanedDocument` collections use `use_vector_index = False`; `EmbeddedChunk` collections use `use_vector_index = True` (Session 1.2).

---

## 📁 Orchestration

ZenML orchestrates the batch pipeline: schedule it, trigger it manually, or run it after the ETL pipeline finishes. Orchestrating the feature pipeline is what enables **continuous training** (Session 11.1).

**In the repo**: `configs/feature_engineering.yaml` sets `parameters.author_full_names` and the SageMaker orchestrator settings; the `end_to_end_data` pipeline chains ETL → feature → datasets. Because `feature_engineering` returns its step invocation ids, the parent can gate on it.

---

## 📁 Migration Playbook: Batch to Streaming

If a future requirement forces streaming, migrate in stages rather than rewriting. Each stage is independently useful and reversible.

```
Stage 0  pull-all batch                    ← today
   │      add last_updated field + index
   ▼
Stage 1  pull-timestamp batch              ← low effort, big read savings
   │      add a durable queue in front of the sink
   ▼
Stage 2  push batch (queue-backed)         ← decouples producers from the sink
   │      introduce Kafka + a streaming engine
   ▼
Stage 3  log-based streaming CDC           ← full CRUD, delete-safe, sub-second
```

| Stage | Trigger to move on | Rollback |
|-------|--------------------|----------|
| 0 → 1 | full scans dominate run time | drop the filter, revert to pull-all |
| 1 → 2 | source load or target downtime causes gaps | drain the queue, run batch again |
| 2 → 3 | deletes or sub-minute freshness required | stop the connector, fall back to Stage 2 |

**Non-negotiables at every stage**:

- **Idempotent sinks.** The deterministic md5 chunk IDs (Session 2.2) already make Qdrant upserts idempotent, so replay after a failure does not duplicate points. Preserve this property in any streaming reimplementation.
- **A replay/backfill path.** Logic changes will require reprocessing history. Batch code is your backfill tool even after you stream.
- **Observability of lag and drift.** Track source-to-feature-store lag and periodically reconcile counts; streaming systems fail *silently* far more often than batch jobs that fail loudly.

**Why the project does not do this yet**: all four stages add moving parts, and none is needed for the LLM Twin's "few minutes" requirement and small data. The book's stance is explicit — start batch, adopt streaming only under a real constraint.

---

## 🛠️ Hands-On

### Step 1: Score three systems

| System | Freshness need | Data volume | Verdict |
|--------|----------------|-------------|---------|
| LLM Twin RAG | minutes OK | thousands | batch |
| Fraud detection | milliseconds | millions/day | streaming |
| Nightly content recommendations | hours OK | millions | batch |

### Step 2: Inspect the repo's sync behavior

```python
# The feature pipeline pulls ALL documents (no last-updated filter).
from llm_engineering.domain.documents import ArticleDocument
docs = ArticleDocument.bulk_find(author_id="<user-uuid>")   # full scan
print(len(docs))
```

`bulk_find` issues a Mongo `collection.find(filter_options)` with no time filter — the whole author's corpus, every run.

### Step 3: Plan a timestamp-based upgrade

1. Add `last_updated` to `NoSQLBaseDocument`.
2. Filter `bulk_find(last_updated={"$gt": since})`.
3. Persist the last sync watermark in a small state file or ZenML artifact.

---

## 📝 Exercise 1: Implement a pull-timestamp CDC

**Task**: add incremental syncing to `query_data_warehouse`.

1. Add a `last_updated` field defaulting to the current time on save/update.
2. Extend `query_data_warehouse` with an optional `since` parameter.
3. Store the watermark between runs (a JSON file via a file manager, or a ZenML artifact).
4. Verify a second run only processes new documents.

**Goal**: convert the coarse pull-all strategy into a scalable pull-timestamp CDC, and reason about deletes (which timestamps cannot capture).

Sketch:

```python
from datetime import datetime, timezone

def __fetch_articles(user_id, since=None):
    query = {"author_id": user_id}
    if since is not None:
        query["last_updated"] = {"$gt": since}
    return ArticleDocument.bulk_find(**query)
```

> **Edge case**: a hard delete at the source leaves an orphan in the feature store forever. Only log-based CDC (or a periodic reconciliation sweep) catches deletes.

---

## 📝 Exercise 2: Choose a sync strategy for a new system

**Task**: given the requirements below, pick batch vs streaming and a CDC pattern, and justify it in five sentences.

System: a support-ticket knowledge base feeding a RAG assistant. Tickets arrive ~200/day, edits are frequent, deletions are rare but must be honored within a day, and users tolerate up to 30 minutes of staleness.

1. Freshness (30 min) → batch is acceptable (run every 15 min).
2. Volume (200/day) → full scan is cheap; pull-all could even work early on.
3. But deletions matter → add pull-timestamp plus a daily reconciliation job, or adopt log-based CDC if deletes are frequent.
4. Volume-growth check → if tickets hit millions, move to log-based.
5. Write the one-paragraph decision.

**Goal**: practice the decision tree against a realistic brief.

---

## 🐛 Common Pitfalls

- **Premature streaming**: streaming is roughly twice the complexity. Start batch (the project's explicit choice).
- **Pull-all at scale**: works for thousands, fails for millions. Add CDC before the data grows.
- **Timestamp CDC misses deletes**: it cannot see hard deletes; log-based CDC can.
- **Storing use-case-specific data in the shared warehouse**: keep the warehouse generic; model per use-case in the feature store.
- **Duplicate vectors**: without deterministic IDs, re-running the pipeline duplicates points. The project's md5 chunk IDs (Session 2.2) prevent this.
- **Clock skew**: a timestamp watermark must use the source DB's clock (or a monotonic sequence), not the application host's, or changes can be skipped.
- **Unindexed timestamp scan**: adding `last_updated` without a Mongo index turns the "incremental" query into a full scan anyway.

---

## 🎓 Knowledge Check

1. **Why does the LLM Twin choose batch over streaming?**
   - Answer: minutes of latency are acceptable, the data is small, and batch is simpler and cheaper.

2. **Name the five core RAG feature-pipeline steps.**
   - Answer: extraction, cleaning, chunking, embedding, and loading.

3. **What does CDC do?**
   - Answer: capture CRUD changes at the source and replicate them to targets with minimal overhead.

4. **Which CDC pattern is considered optimal, and what is its cost?**
   - Answer: log-based; it needs a queue and a streaming pipeline and has vendor-specific log formats.

5. **Why is cleaned data stored in Qdrant as well as the warehouse?**
   - Answer: features are read only from the feature store, and the vector DB's metadata index works as a NoSQL store.

6. **What two stores make up the logical feature store?**
   - Answer: Qdrant (online serving) and ZenML artifacts (offline training datasets).

7. **What is the difference between push and pull CDC?**
   - Answer: push has the source notify targets (near-instant, needs a queue to survive target downtime); pull has targets request changes periodically (delayed, lighter on the source).

8. **Why can timestamp CDC not capture deletes?**
   - Answer: a deleted row no longer exists to carry a `LAST_MODIFIED` value; the change simply disappears from the query.

9. **What is an example of a system that genuinely needs streaming?**
   - Answer: fraud detection or high-frequency trading, where milliseconds matter.

10. **What repo setting distinguishes cleaned documents from embedded chunks?**
    - Answer: `use_vector_index` — `False` for cleaned, `True` for embedded.

11. **Why is modeling use-case-specific data in the shared warehouse an antipattern?**
    - Answer: other teams/use cases need different processing; a generic warehouse plus downstream modeling avoids spaghetti.

12. **What does `wait_for` enable in `feature_engineering`?**
    - Answer: gating the feature pipeline on completion of upstream ETL steps, supporting continuous training.

13. **What is the natural first CDC upgrade for this project?**
    - Answer: pull-timestamp, adding a `last_updated` field and a watermark persisted between runs.

14. **What could you build to catch deletes even with timestamp CDC?**
    - Answer: a periodic reconciliation sweep (or log-based CDC if deletes are frequent).

15. **Which repo step performs the "load" of the five RAG steps?**
    - Answer: `steps/feature_engineering/load_to_vector_db.py`.

---

## 📖 Glossary

- **Batch pipeline**: collects and processes data in intervals and bulk volumes.
- **Streaming pipeline**: processes each record continuously with minimal latency.
- **Event platform**: the distributed log/bus carrying events (Kafka, Redpanda).
- **Streaming engine**: the processor consuming events (Flink, Bytewax).
- **CDC (Change Data Capture)**: replicating source CRUD changes to other stores.
- **Push CDC**: source initiates replication.
- **Pull CDC**: target periodically requests changes.
- **Watermark**: the stored "last synced" position/time used by incremental pulls.
- **Reconciliation**: a periodic full comparison that repairs drift (including deletes).
- **Continuous training**: retraining triggered by fresh data, enabled by orchestration.
- **Logical feature store**: the union of online (Qdrant) and offline (ZenML artifacts) feature storage.
- **Freshness**: how old the newest data in the feature store can be before it is unacceptable.
- **Idempotent sink**: a write path that produces the same result whether run once or many times.
- **Backfill / replay**: reprocessing historical data after a logic change or outage.
- **Lag**: the delay between a change at the source and its appearance in the feature store.
- **Exactly-once semantics**: the stronger guarantee that each event affects the sink once, even across failures.
- **Reconciliation**: a periodic full comparison that repairs drift, including missed deletes.
- **Watermark store**: wherever the last-sync position persists between runs (file, table, artifact).
- **Sink / target**: the destination store that receives replicated changes.
- **Source of truth**: the authoritative store the feature store is derived from (here, MongoDB).

---

## 🔗 Next Session

**Session 5.1: Supervised Fine-Tuning** — the cleaned documents become the SFT instruction dataset.

---

## 📚 Additional Resources

- [What is Change Data Capture? (Confluent)](https://www.confluent.io/en-gb/learn/change-data-capture/)
- [Bytewax](https://bytewax.io/)
- [Apache Flink](https://flink.apache.org/)
- [Apache Kafka](https://kafka.apache.org/)
- [Redpanda](https://redpanda.com/)
- [Debezium (log-based CDC)](https://debezium.io/)
- [ZenML docs](https://docs.zenml.io/)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 2.2, 2.3, 8.3

**Outcome**: You can justify a batch design, trace the five RAG feature steps to repo files, explain push/pull and the three CDC patterns, and plan a timestamp-based incremental-sync upgrade.
