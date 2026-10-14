# Session 7.2: Prompt Monitoring with Opik

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how Opik traces the RAG path with `@opik.track`
- Read the trace hierarchy produced by one query
- Use `opik_context.update_current_trace` for tags and metadata
- Interpret token counts and latency per span
- Add a custom traced span
- Know the three things to monitor constantly: model config, token totals, and step duration
- Decide how much granularity a trace needs, and why tracing everything is harmful
- Attach feedback scores from user ratings and LLM judges
- Diagnose retrieval and generation failures from span data alone
- Compare hosted and local Opik for privacy and cost

---

## ✅ Prerequisites

- **Session 6.1 (FastAPI API)** — the business microservice being traced.
- **Session 6.2 (RAG Inference Flow)** — the retrieval and generation functions Opik wraps.
- **Session 7.1 (Comet ML)** — the shared `COMET_API_KEY` and project concept.
- A free [Comet ML account](https://www.comet.com/) and API key.
- `.env` with `COMET_API_KEY` and `COMET_PROJECT`.
- For the optional local server: the Opik self-hosted deployment (Docker).

---

## 🏗️ Architecture Overview

```
rag(query)                                    @opik.track → root trace
├── ContextRetriever.search                   @opik.track (name="ContextRetriever.search")
│   ├── SelfQuery.generate                    @opik.track (name="SelfQuery.generate")
│   ├── QueryExpansion.generate               @opik.track (name="QueryExpansion.generate")
│   ├── (parallel Qdrant searches)            not traced (fast)
│   └── Reranker.generate                     @opik.track (name="Reranker.generate")
└── call_llm_service(query, context)          @opik.track
    └── InferenceExecutor / SageMaker call    (HTTP span)

opik_context.update_current_trace(tags=["rag"], metadata={...tokens...})
```

Every `@opik.track`-decorated function becomes a span nested under its caller. Opik (by
Comet ML) collects inputs, outputs, timing, and any metadata you attach.

### Why a trace, not a log line

Standard logging tools break down for LLM applications because a prompt is not a flat message;
it is a structured chain where one prompt depends on the previous output. A trace groups those
dependent steps so you can see the whole request and drill into any node. The book's point:
you need an intuitive way to group these traces in a specialized dashboard, not plain text
logs.

### Where monitoring lives in the serving architecture

```
┌────────────────────────────┐        HTTP         ┌──────────────────────────┐
│  Business microservice      │ ─────────────────▶  │  LLM microservice         │
│  (FastAPI)  ◀── Opik here   │   prompt in body    │  (SageMaker endpoint)     │
│  - rag()                    │ ◀─────────────────  │  - generate text          │
│  - call_llm_service()       │   generated_text    │  narrow scope             │
│  - ContextRetriever         │                     │                           │
└────────────────────────────┘                     └──────────────────────────┘
```

The book is explicit: the LLM microservice has a narrow scope (prompt in, answer out), so the
**business microservice** — which coordinates the end-to-end flow — is the right place for
the monitoring pipeline. In our repo that is the FastAPI server in
`llm_engineering/infrastructure/inference_pipeline_api.py`.

---

## 📁 Key Files Explained

### 1. `infrastructure/opik_utils.py` - Configuration

```python
# llm_engineering/infrastructure/opik_utils.py
import os

import opik
from loguru import logger
from opik.configurator.configure import OpikConfigurator

from llm_engineering import settings


def configure_opik() -> None:
    if settings.COMET_API_KEY and settings.COMET_PROJECT:
        try:
            client = OpikConfigurator(api_key=settings.COMET_API_KEY)
            default_workspace = client._get_default_workspace()
        except Exception:
            logger.warning("Default workspace not found. Setting workspace to None and enabling interactive mode.")
            default_workspace = None

        os.environ["OPIK_PROJECT_NAME"] = settings.COMET_PROJECT

        opik.configure(api_key=settings.COMET_API_KEY, workspace=default_workspace, use_local=False, force=True)
        logger.info("Opik configured successfully.")
    else:
        logger.warning(
            "COMET_API_KEY and COMET_PROJECT are not set. Set them to enable prompt monitoring with Opik (powered by Comet ML)."
        )
```

**Key Concepts**:
- **`configure_opik()` must run before any traced call**. The API calls it at import time
  (`configure_opik()` at module level in `inference_pipeline_api.py`), and `tools/rag.py`
  calls it explicitly.
- **`_get_default_workspace()`** discovers the workspace the API key belongs to. It is a
  *private* method (leading underscore) — a deliberate reliance on library internals that
  could break on an Opik upgrade.
- **`OPIK_PROJECT_NAME` set from `COMET_PROJECT`** routes traces into the same project name
  used by training experiments. This is what unifies Comet and Opik under one project.
- **`use_local=False`** sends traces to the hosted Opik service. Set it to `True` for a fully
  local Opik server (privacy / air-gapped).
- **`force=True`** overwrites any existing Opik config on disk. Convenient but destructive of
  prior settings.
- **Best-effort**: if configuration fails, it warns and the application keeps running
  untraced — a deliberate availability choice. The cost is silent loss of observability.

#### Mock-mode tracing

Opik tracing is orthogonal to the `mock` flag in the RAG steps. `ContextRetriever(mock=True)`
still produces spans; the sub-steps just return early. So you can develop offline and still
see trace structure.

---

### 2. The Traced Call Chain

**Root span - `rag`**:

```python
# llm_engineering/infrastructure/inference_pipeline_api.py
import opik
from opik import opik_context
...
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
```

**Key Concepts**:
- **`opik_context.update_current_trace`** enriches the root trace with a `rag` tag and
  model/token metadata. This is what makes a trace searchable and comparable.
- **Token counts are computed with the generator's tokenizer** (`misc.compute_num_tokens` uses
  `AutoTokenizer.from_pretrained(settings.HF_MODEL_ID)`), so they reflect what the endpoint
  processes.
- **The same key names** (`query_tokens`, `context_tokens`, `answer_tokens`) appear
  consistently, enabling dashboards that aggregate over traces.
- **The decorator has no parentheses** (`@opik.track`), so the function name `rag` becomes the
  span/trace name. Sub-steps use `@opik.track(name="...")` to override.

> **Book vs repo note.** The book shows `get_current_trace()` and
> `trace.update(tags=..., metadata=...)`, plus `feedback_scores`. The repo uses the newer,
> equivalent `opik_context.update_current_trace(...)`. Both attach the same information; trust
> the repo for the exact call.

**Retrieval spans**:

```python
# llm_engineering/application/rag/retriever.py
    @opik.track(name="ContextRetriever.search")
    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3,
    ) -> list:
```

```python
# llm_engineering/application/rag/query_expanison.py
    @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
```

```python
# llm_engineering/application/rag/self_query.py
    @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
```

```python
# llm_engineering/application/rag/reranking.py
    @opik.track(name="Reranker.generate")
    def generate(self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
```

**Key Concepts**:
- **`name=` overrides the span name** for readability (for example `SelfQuery.generate`
  instead of the default module-qualified name).
- Because `ContextRetriever.search` is traced and it calls the three sub-steps, Opik builds a
  tree: `search` contains `SelfQuery`, `QueryExpansion`, and `Reranker`.
- **Inputs and outputs are captured automatically**. You can see the extracted author, the
  generated query variants, and the reranked chunks in the trace.
- **The three sub-steps do not wrap the parallel Qdrant searches**; `_search` is untraced, so
  the vector calls are not individually visible. The book's guidance: trace meaningful units,
  not everything.

**LLM span**:

```python
@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()

    return answer
```

**Key Concepts**:
- The LLM call is a sibling span under `rag`, separate from retrieval. This split lets you
  attribute latency to retrieval versus generation.
- **No explicit token usage is set on this span** because the SageMaker invocation returns
  text, not usage stats. The `rag` trace computes tokens after the fact.
- Because `call_llm_service` is decorated, Opik records the **exact augmented prompt sent to
  the model** (the input) and the raw answer (the output) — the single most useful artifact for
  debugging generation quality.

---

### 3. The full chain, assembled

```
rag(query)                                                        ← trace root
│  tags=["rag"]
│  metadata: model_id, embedding_model_id, temperature,
│            query_tokens, context_tokens, answer_tokens
│
├── ContextRetriever.search(query, k=3, expand_to_n_queries=3)    ← span
│   │  input:  raw query string
│   │  output: list[EmbeddedChunk] (top-k after rerank)
│   │
│   ├── SelfQuery.generate(query)                                 ← span (OpenAI call)
│   │      output: Query with author_id / author_full_name
│   │
│   ├── QueryExpansion.generate(query, expand_to_n=3)             ← span (OpenAI call)
│   │      output: list[Query] (original + 2 variants)
│   │
│   └── Reranker.generate(query, chunks, keep_top_k=3)            ← span (cross-encoder)
│          output: top-3 chunks by cross-encoder score
│
└── call_llm_service(query, context)                              ← span (SageMaker HTTP)
       input:  full augmented prompt
       output: generated answer string
```

---

## 🔬 Deep Dive: What a Good Trace Shows

```
rag (trace)
  tags: ["rag"]
  metadata: model_id, embedding_model_id, temperature,
            query_tokens, context_tokens, answer_tokens
  │
  ├─ ContextRetriever.search            ~120 ms
  │    input:  query string
  │    output: list of EmbeddedChunk
  │    ├─ SelfQuery.generate            ~300 ms   (LLM call)
  │    │    output: author_id + author_full_name
  │    ├─ QueryExpansion.generate       ~400 ms   (LLM call)
  │    │    output: 3 variant queries
  │    └─ Reranker.generate             ~80 ms    (cross-encoder)
  │         output: top-3 chunks
  │
  └─ call_llm_service                   ~1.5 s
       input:  prompt (query + context)
       output: answer
```

**What to look for**:
- **Latency split**: two LLM calls (SelfQuery, QueryExpansion) plus the generation call
  dominate latency. Retrieval-only logic is fast.
- **Empty retrieval**: if `context_tokens` is 0, the model answered without grounding.
  Investigate the author filter and collection contents.
- **Author extraction**: confirm `SelfQuery` returned the right name and not `none`.
- **Variant quality**: read the expanded queries; poor variants correlate with poor recall.
- **Prompt fidelity**: open `call_llm_service`'s input to see exactly what the model received,
  including any brace-escaping issues.

**Why the two extra LLM calls matter**: SelfQuery and QueryExpansion add latency but improve
precision and recall. If latency is critical, cache them or disable expansion for simple
queries. The trace is how you find that tradeoff.

### The three monitoring dimensions (from the book)

The book states there are three main aspects to monitor constantly:

| Dimension | What to log | Why it matters |
|-----------|-------------|----------------|
| **Model configuration** | LLM id, embedding model id, reranker id, temperature | Generation behavior changes with model and temperature |
| **Total number of tokens** | input/prompt tokens, context tokens, answer tokens | Directly drives serving cost; a sudden rise signals a bug |
| **Duration of each step** | per-span latency | Finds bottlenecks; enables latency budgets and alerts |

The repo's `rag` metadata captures model ids, embedding model id, temperature, and
query/context/answer token counts — the first two dimensions. Per-span duration is captured
automatically by Opik for the third.

### Feedback scores

The book expands `update_current_trace` with `feedback_scores`: a user rating and an LLM-judge
score. The repo's current call passes only `tags` and `metadata`, but the API accepts
`feedback_scores` for the same trace:

```python
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
    feedback_scores=[
        {"name": "user_feedback", "value": 1.0, "reason": "The response was valuable and correct."},
        {"name": "llm_judge_score", "value": 0.85, "reason": "Runtime metric from an LLM judge."},
    ],
)
```

Feedback scores turn traces into a dataset you can filter and analyze: "show me all traces
where `user_feedback < 1`", "what is the average `llm_judge_score` when `context_tokens = 0`".

### How much granularity?

The book warns that tracing everything is dangerous: too much noise makes manual tracing hard.
Its rule of thumb is to trace the critical functions (`rag()`, `call_llm_service()`) and add
granularity gradually. This repo adds the retriever and its three sub-steps — a middle ground.

```
Too little          Just right                   Too much
─────────────────────────────────────────────────────────────
rag() only     rag, call_llm_service,     every helper, every
               retriever, sub-steps       loop iteration, every
                                          Qdrant call
```

---

## 🛠️ Hands-On: Produce and Inspect a Trace

### Step 1: Configure

```env
COMET_API_KEY=your-key
COMET_PROJECT=twin
```

### Step 2: Focus on retrieval via the retrieval harness

```bash
python -m tools.rag
```

This calls `ContextRetriever.search` directly. Even without the API, the `search`,
`SelfQuery`, `QueryExpansion`, and `Reranker` spans are recorded. Note that `tools/rag.py`
calls `configure_opik()` explicitly.

### Step 3: Run the full RAG

```bash
python -m tools.ml_service
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" \
  -d "{\"query\": \"My name is Paul Iusztin. Draft a post about RAG.\"}"
```

### Step 4: Inspect in Opik

Open the Opik project for `twin`. You should see the `rag` trace with the tag and metadata, the
retrieval subtree, and the LLM span. Check:
- `query_tokens`, `context_tokens`, `answer_tokens`.
- The SelfQuery output (author id/name).
- The three expanded queries (original plus two variants).
- The reranked chunks and their order.
- The exact prompt in `call_llm_service`.

### Step 5: Correlate with Comet

Because `OPIK_PROJECT_NAME` is set from `COMET_PROJECT`, the trace project name matches the
training experiment project. Use the same account to view both.

---

## 📝 Exercise: Add a Custom Span for Retrieval Filtering

### Task

Trace the Qdrant filtering decision explicitly.

```python
import opik
from opik import opik_context


@opik.track(name="build_qdrant_filter")
def build_query_filter(author_id):
    opik_context.update_current_span(metadata={"has_author_filter": author_id is not None})
    ...
```

1. Wrap `_search_data_category`'s filter construction in this traced helper.
2. Re-run a query with and without an author name.
3. Confirm the span shows `has_author_filter: true/false`.

**Goal**: Practice creating custom spans and span-level metadata, so any decision inside the
pipeline becomes observable.

---

## 🧪 Second Exercise: Add an LLM-Judge Feedback Score

### Task

After generation, score the answer for faithfulness and attach it as a feedback score on the
same trace.

```python
import opik
from opik import opik_context


@opik.track(name="judge_faithfulness")
def judge_faithfulness(query: str, context: str, answer: str) -> float:
    # In production, call a judge model. Here, a deterministic heuristic keeps the
    # exercise runnable offline: fraction of answer content words found in context.
    answer_terms = set(answer.lower().split())
    context_terms = set(context.lower().split())
    if not answer_terms:
        return 0.0
    overlap = len(answer_terms & context_terms) / len(answer_terms)
    return round(overlap, 3)


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)
    answer = call_llm_service(query, context)

    score = judge_faithfulness(query, context, answer)

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
        feedback_scores=[
            {"name": "faithfulness", "value": score, "reason": "Lexical overlap with context."},
        ],
    )
    return answer
```

1. Wire `judge_faithfulness` after generation and attach the score.
2. Run queries with and without a name (filtered vs unfiltered) and compare scores.
3. Filter the Opik project by `feedback_scores.faithfulness < 0.5` and inspect the failing
   traces.

**Goal**: Close the loop from trace data to quality signal. This is the bridge to the
evaluation work in Sessions 7.3 and 7.4.

---

## 🐛 Common Pitfalls

- **Opik not configured**: without a key, traces are silently dropped and only a warning is
  logged. Verify the log line "Opik configured successfully."
- **Decorating nested functions too much**: every traced call adds overhead. Trace meaningful
  units (retrieval stages, LLM calls), not tight loops or per-chunk helpers.
- **PII in traces**: queries and context are stored. Avoid sending sensitive data through the
  API, or use local Opik (`use_local=True`).
- **Consistent metadata keys**: renaming `context_tokens` breaks dashboards that aggregate on
  it. Keep names stable.
- **`configure_opik()` ordering**: calling a traced function before configuration means the
  early spans are lost. The API configures at import; scripts must call it first.
- **Reliance on a private API**: `OpikConfigurator._get_default_workspace` is underscore-prefixed
  and can change in a library update; pin the Opik version or guard the call.
- **Network egress from the host**: hosted Opik needs outbound access. In a restricted
  environment, spans buffer or drop. Use `use_local=True` with a local server.
- **Duplicate span names**: two functions with the same auto-name become ambiguous. Use
  explicit `name=` for the sub-steps, as the repo does.
- **Token counts from the wrong tokenizer**: `misc.compute_num_tokens` uses the generator's
  tokenizer. Comparing with an OpenAI tokenizer count will not match.
- **Tracing generated answers as metadata can grow unbounded**: only log summaries/counts,
  not full payloads, in metadata.

---

## 🧭 Edge Cases and Failure Modes

| Scenario | What Opik shows | Interpretation |
|----------|-----------------|----------------|
| `context_tokens == 0` | Retrieval subtree with empty output | Ungrounded answer; check filter/collections |
| SelfQuery output `none` | No author filter applied | Correct behavior when no name present |
| Fewer than N variants | QueryExpansion output shorter | LLM returned fewer alternatives; tolerated |
| High SelfQuery latency | Long span, small output | Cold OpenAI connection or rate limiting |
| No LLM span | `call_llm_service` missing | Endpoint unreachable or exception before span close |
| Duplicate traces | Two `rag` traces per request | Retry or double-invocation at the caller |
| Missing `rag` tag | Trace found but untagged | Metadata update failed or ran after an exception |
| Cross-project confusion | Traces under wrong project | `OPIK_PROJECT_NAME` not set from `COMET_PROJECT` |

---

## 🎓 Knowledge Check

1. **What decorator creates a trace or span?**
   - Answer: `@opik.track`.

2. **How do you attach tags and metadata to the whole request?**
   - Answer: `opik_context.update_current_trace(tags=..., metadata=...)`.

3. **Why is `call_llm_service` a separate span from retrieval?**
   - Answer: To attribute latency to generation versus retrieval.

4. **What does the `Reranker.generate` span capture?**
   - Answer: Its inputs (query, chunks) and output (top-k chunks).

5. **Where do Opik traces go by default?**
   - Answer: The hosted Opik service routed by `OPIK_PROJECT_NAME`, unless `use_local=True`.

6. **What does `context_tokens = 0` indicate?**
   - Answer: Retrieval returned no usable context, so the answer was ungrounded.

7. **Why does the repo call `configure_opik()` at import time in the API module?**
   - Answer: Traced calls must not run before Opik is configured, or their spans are lost.

8. **What does `use_local=True` change?**
   - Answer: Traces go to a local Opik server instead of the hosted service (privacy/air-gap).

9. **Which three aspects should you monitor constantly?**
   - Answer: Model configuration, total token counts, and the duration of each step.

10. **Why can tracing everything be harmful?**
    - Answer: Excess spans add overhead and noise, making traces hard to read; trace critical
      functions and add granularity gradually.

11. **How do you attach a feedback score to a trace?**
    - Answer: Pass `feedback_scores=[{name, value, reason}, ...]` to
      `update_current_trace`.

12. **What is the difference between the `rag` span and the retrieval sub-spans' naming?**
    - Answer: `rag` uses the bare `@opik.track`, so it takes the function name; sub-steps
      override with `name=` for readability.

13. **Which function's input gives you the exact prompt sent to the LLM?**
    - Answer: `call_llm_service`; its input is the augmented prompt.

14. **What happens if `COMET_API_KEY` is unset?**
    - Answer: `configure_opik` warns and the app runs untraced; no spans are recorded.

15. **How do Comet experiments and Opik traces end up in the same project?**
    - Answer: `OPIK_PROJECT_NAME` is set from `COMET_PROJECT`, so both use that project name
      under the same API key.

---

## 📖 Glossary

- **Trace** — the end-to-end record of one request, containing nested spans.
- **Span** — one traced unit of work (a function call) with input, output, timing, metadata.
- **`@opik.track`** — decorator that turns a function into a span; nested calls form a tree.
- **`opik_context`** — module for accessing the current trace/span at runtime.
- **`update_current_trace`** — attach tags, metadata, and feedback scores to the root trace.
- **`update_current_span`** — attach metadata to the current span.
- **Feedback score** — a named quality value (user rating or judge metric) on a trace.
- **`OPIK_PROJECT_NAME`** — environment variable that routes traces to a project.
- **`use_local`** — Opik configure flag to target a local server.
- **LLM-as-judge** — using a model to score another model's output.
- **Granularity** — how many spans a trace contains; a noise/observability tradeoff.
- **Business microservice** — the service coordinating the flow; where monitoring belongs.

---

## 🔗 Next Session

**Session 7.3**: Model Evaluation

We score SFT, DPO, and Instruct models with an LLM-as-a-judge pipeline.

Related reading in this repo:
- [Session 6.1: FastAPI API](session_6.1_fastapi_api.md)
- [Session 6.2: RAG Inference Flow](session_6.2_rag_inference_flow.md)
- [Session 7.1: Comet ML](session_7.1_comet_ml.md)
- [Session 7.4: RAG Evaluation](session_7.4_rag_evaluation.md)

---

## 📚 Additional Resources

- [Opik Documentation](https://www.comet.com/docs/opik/)
- [Opik Tracing Quickstart](https://www.comet.com/docs/opik/tracing)
- [Opik source on GitHub](https://github.com/comet-ml/opik)
- [Comet ML](https://www.comet.com/)

---

## 📑 References

- Iusztin, P. & Labonne, M. *LLM Engineer's Handbook.* Packt, 2024. **Chapter 11, "MLOps and
  LLMOps"** (pp. 480-486): prompt monitoring with Opik, `@track`, `update_current_trace`,
  feedback scores, granularity guidance, the three monitoring dimensions, and the serving
  architecture. **Chapter 2** (pp. 75-76): why prompt monitoring needs a specialized tool.
- Repo source of truth: `llm_engineering/infrastructure/opik_utils.py`,
  `llm_engineering/infrastructure/inference_pipeline_api.py`,
  `llm_engineering/application/rag/retriever.py`,
  `llm_engineering/application/rag/query_expanison.py`,
  `llm_engineering/application/rag/self_query.py`,
  `llm_engineering/application/rag/reranking.py`,
  `llm_engineering/application/utils/misc.py`, `llm_engineering/settings.py`.

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 6.1, 6.2, 7.1

**Outcome**: You can read and extend Opik traces, attach metadata and feedback scores, and
diagnose retrieval and generation issues from span data.
