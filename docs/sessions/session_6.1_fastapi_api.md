# Session 6.1: FastAPI REST API

## 🎯 Learning Objectives

By the end of this session, you will:
- Read the FastAPI app that exposes the LLM as an HTTP service
- Understand the `Inference` and `DeploymentStrategy` domain abstractions
- Use `InferenceExecutor` to build a prompt and call SageMaker
- Add request validation and error handling with Pydantic
- Trace a request through ASGI, routing, validation, RAG, and generation
- Diagnose the synchronous-blocking problem and plan a fix
- Run the API locally with uvicorn and call it with curl
- Extend the app with a health check and a streaming endpoint

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI process                                │
│                                                                        │
│   POST /rag  { "query": "..." }                                        │
│        │                                                               │
│        ▼                                                               │
│   rag(query)  ──► ContextRetriever.search  (Session 6.2)              │
│        │                                                               │
│        ▼                                                               │
│   call_llm_service(query, context)                                     │
│        │                                                               │
│        ▼                                                               │
│   LLMInferenceSagemakerEndpoint  (implements Inference ABC)           │
│        │  boto3 sagemaker-runtime invoke_endpoint                      │
│        ▼                                                               │
│   InferenceExecutor.execute()  ──►  answer  (HTTP 200)                 │
└──────────────────────────────────────────────────────────────────────┘
```

### The request lifecycle, step by step

```
1. Client → POST /rag  {"query": "..."}
2. Uvicorn (ASGI server) parses the HTTP request
3. FastAPI matches the route and the HTTP method
4. Pydantic validates the JSON body against QueryRequest
      ├─ valid   → continue
      └─ invalid → 422 Unprocessable Entity, rag_endpoint never runs
5. rag_endpoint(request) runs
6. rag(query):
      a. ContextRetriever(mock=False).search(query, k=3)   (CPU + network I/O)
      b. EmbeddedChunk.to_context(documents)               (string building)
      c. call_llm_service(query, context)                  (network I/O)
7. InferenceExecutor formats the prompt, sets the payload, calls inference()
8. LLMInferenceSagemakerEndpoint.inference() → boto3 invoke_endpoint (GPU)
9. Response parsed, extracted as [0]["generated_text"], returned
10. FastAPI serializes QueryResponse to JSON → 200 OK
```

### What runs where

Because the LLM was split into its own microservice (Session 5.3), the FastAPI
process is deliberately light:

| Step | Bound | Where it runs |
|------|-------|---------------|
| Retrieval | Network I/O + CPU | FastAPI process |
| Embedding | CPU | FastAPI process |
| Context building | CPU (string) | FastAPI process |
| Generation | GPU | SageMaker endpoint |
| Tracing | Network I/O | Opik/Comet cloud |

This is why the API server can be a cheap, GPU-less machine: the expensive
compute is behind an HTTP hop.

### Domain Abstractions

```
domain/inference.py
├── DeploymentStrategy (ABC)   ← implemented by SagemakerHuggingfaceStrategy (Session 5.3)
└── Inference (ABC)            ← implemented by LLMInferenceSagemakerEndpoint
        set_payload(inputs, parameters)
        inference() -> dict
```

The `Inference` ABC lets the API depend on an interface, not boto3. A future
local-vLLM executor only has to implement the same two methods.

---

## 📁 Key Files Explained

### 1. `domain/inference.py` - The Contract

```python
# llm_engineering/domain/inference.py
from abc import ABC, abstractmethod


class DeploymentStrategy(ABC):
    @abstractmethod
    def deploy(self, model, endpoint_name: str, endpoint_config_name: str) -> None:
        pass


class Inference(ABC):
    """An abstract class for performing inference."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def set_payload(self, inputs, parameters=None):
        pass

    @abstractmethod
    def inference(self):
        pass
```

**Key Concepts**:
- **`Inference` has two abstract methods**: `set_payload` (build the request) and
  `inference` (execute it). The API only programs against these.
- **`__init__` sets `self.model = None`**, giving subclasses a slot for a model
  handle. The SageMaker subclass does not use it, but a local executor would.
- **`DeploymentStrategy`** is implemented in infrastructure (Session 5.3),
  keeping the deploy mechanism swappable.
- **Dependency inversion**: `run.py` and the API import `Inference` from the
  domain, so they never import boto3 directly.

### 2. `model/inference/inference.py` - SageMaker Client

```python
# llm_engineering/model/inference/inference.py
class LLMInferenceSagemakerEndpoint(Inference):
    def __init__(
        self,
        endpoint_name: str,
        default_payload: Optional[Dict[str, Any]] = None,
        inference_component_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.client = boto3.client(
            "sagemaker-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.endpoint_name = endpoint_name
        self.payload = default_payload if default_payload else self._default_payload()
        self.inference_component_name = inference_component_name

    def _default_payload(self) -> Dict[str, Any]:
        return {
            "inputs": "How is the weather?",
            "parameters": {
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "top_p": settings.TOP_P_INFERENCE,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "return_full_text": False,
            },
        }

    def set_payload(self, inputs: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        self.payload["inputs"] = inputs
        if parameters:
            self.payload["parameters"].update(parameters)

    def inference(self) -> Dict[str, Any]:
        try:
            logger.info("Inference request sent.")
            invoke_args = {
                "EndpointName": self.endpoint_name,
                "ContentType": "application/json",
                "Body": json.dumps(self.payload),
            }
            if self.inference_component_name not in ["None", None]:
                invoke_args["InferenceComponentName"] = self.inference_component_name
            response = self.client.invoke_endpoint(**invoke_args)
            response_body = response["Body"].read().decode("utf8")
            return json.loads(response_body)
        except Exception:
            logger.exception("SageMaker inference failed.")
            raise
```

**Key Concepts**:
- **`sagemaker-runtime` client** is distinct from the `sagemaker` control-plane
  client used for deployment. Runtime invokes; control plane manages.
- **Payload shape matches the TGI container**: `{"inputs": ..., "parameters": {...}}`.
- **`return_full_text=False`** makes the container return only the generated
  continuation, not the prompt + continuation.
- **`InferenceComponentName`** is added only when set and not the string
  `"None"`. The check `not in ["None", None]` defends against a stringified
  `None` coming from an environment variable. It is required for
  component-based endpoints.
- **`set_payload` mutates in place**: `self.payload["parameters"].update(...)`
  means a second call reuses the first call's parameters unless explicitly
  overwritten. For a per-request instance this is fine; do not share one
  instance across requests.
- **Error handling re-raises after `logger.exception`** (which includes the
  traceback), so the API layer can translate it to an HTTP error.

### 3. `model/inference/run.py` - Prompt Builder and Executor

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
- **`InferenceExecutor` is the template-method glue**: it formats the prompt, sets
  the payload, calls the model, and extracts `[0]["generated_text"]`.
- **The default prompt is a RAG prompt**: it instructs the model to use the
  provided context as the primary source of information.
- **`context if context else ""`** guarantees `str.format` never sees `None`.
- **`repetition_penalty=1.1`** discourages loops, which is important for a small
  `max_new_tokens` budget.
- **`temperature=0.01`** (from settings) makes the endpoint nearly deterministic,
  appropriate for a factual assistant.
- **`prompt` is overridable**, so callers can supply a different template without
  subclassing.
- **`inference()` returns a list of dicts**, which is the TGI response shape;
  `[0]["generated_text"]` is the first (and only) completion.

### 4. `infrastructure/inference_pipeline_api.py` - The FastAPI App

```python
# llm_engineering/infrastructure/inference_pipeline_api.py
import opik
from fastapi import FastAPI, HTTPException
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

configure_opik()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


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
- **`configure_opik()` runs at import time** so every endpoint call is traced.
- **`QueryRequest` / `QueryResponse`** are the request/response contracts.
  FastAPI validates the body automatically and returns **422** on a missing
  `query`.
- **`response_model=QueryResponse`** documents and enforces the output shape in
  the OpenAPI schema; returning a different key fails validation.
- **`@opik.track` on `call_llm_service` and `rag`** creates nested spans under
  the request trace.
- **`ContextRetriever(mock=False)`** performs real retrieval. `mock=True` returns
  canned chunks, useful for a no-Qdrant smoke test.
- **The repo uses `k=3`**, not `k=3 * 3` as the book prints. Three chunks are
  fetched and passed to `to_context`.
- **`misc.compute_num_tokens`** measures token counts for query, context, and
  answer; these become Opik trace metadata for cost and drift analysis.
- **Error handling**: any exception becomes a `500` with the exception text. In
  production, log the detail and return a generic message to avoid leaking
  internals.
- **The endpoint is `async` but `rag()` is synchronous and blocks the event
  loop.** For a single-user local service this is fine. Under load, move the work
  to a thread pool (`await run_in_threadpool(rag, request.query)`) or make the
  whole chain async.

### 5. `tools/ml_service.py` - The Runner

```python
# tools/ml_service.py
from llm_engineering.infrastructure.inference_pipeline_api import app  # noqa

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tools.ml_service:app", host="0.0.0.0", port=8000, reload=True)
```

**Key Concepts**:
- **`reload=True`** hot-reloads on code changes - development only.
- **`host="0.0.0.0"`** binds all interfaces, which is convenient in Docker and
  risky on an open network.
- The string form `"tools.ml_service:app"` is required for reload workers to
  re-import the app.
- The equivalent poe task is:
  `poetry run uvicorn tools.ml_service:app --host 0.0.0.0 --port 8000 --reload`.

### 6. `infrastructure/opik_utils.py` - Tracing Setup

```python
# llm_engineering/infrastructure/opik_utils.py
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
        logger.warning("COMET_API_KEY and COMET_PROJECT are not set. ...")
```

**Key Concepts**:
- **Best-effort configuration**: if the key or workspace is missing, it warns and
  continues rather than crashing the API. The app still serves untraced.
- `OPIK_PROJECT_NAME` routes traces into the configured project (Session 7.2).
- **`force=True`** reconfigures even if a previous config exists, which matters
  across reloads.
- It calls the private `_get_default_workspace()`, wrapped in `try/except` so a
  private-API change degrades gracefully.

### 7. `model/inference/test.py` - Standalone Client

```python
# llm_engineering/model/inference/test.py
if __name__ == "__main__":
    text = "Write me a post about AWS SageMaker inference endpoints."
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, text).execute()
    logger.info(f"Answer: '{answer}'")
```

Run it with `poetry poe test-sagemaker-endpoint`. It exercises the same
`Inference` → `InferenceExecutor` path the API uses, without the HTTP layer.

---

## 🔬 Deep Dive: The ASGI / FastAPI / Uvicorn Stack

| Layer | Responsibility | In this project |
|-------|----------------|-----------------|
| ASGI | Async server interface between app and server | Uvicorn speaks ASGI |
| Uvicorn | Runs the ASGI app, parses HTTP | `tools/ml_service.py` |
| FastAPI | Routing, validation, serialization, schema | `inference_pipeline_api.py` |
| Pydantic | Data validation and typing | `QueryRequest`, `QueryResponse` |
| Starlette | Underlying toolkit (requests, responses) | Provided by FastAPI |

**Why ASGI matters here**: FastAPI is async-capable, but the handler calls a
synchronous function. ASGI gives you concurrency only if your code yields to the
event loop. A blocking call inside an `async def` monopolizes the event loop.

### Sync versus async decision

| Option | Code | Behavior under load |
|--------|------|---------------------|
| Current (`async` + sync call) | `async def rag_endpoint` calls `rag()` | Blocks event loop; requests queue |
| `def` handler | `def rag_endpoint` | FastAPI runs it in a threadpool; safe concurrency |
| Threadpool offload | `await run_in_threadpool(rag, request.query)` | Keeps async handler, offloads blocking work |
| Fully async clients | async boto3 / httpx for retrieval and inference | True non-blocking; most work to change |

The simplest safe fix is to declare the handler `def` instead of `async def`;
Starlette then runs it in a worker thread. The cleanest is the threadpool
offload.

### Pydantic validation outcomes

| Request body | Result |
|--------------|--------|
| `{"query": "hello"}` | 200 with `{"answer": "..."}` |
| `{}` | 422, field `query` missing |
| `{"query": 123}` | 422, `query` is not a valid string (coerced only if allowed) |
| `{"query": "hi", "extra": 1}` | 200; extra fields are ignored by default |
| `not json` | 422, JSON decode error |

FastAPI returns the 422 before `rag_endpoint` runs, so no LLM call is made on
invalid input. That is a real cost saving.

---

## 🛠️ Hands-On: Run and Call the API

### Step 1: Prerequisites

- Qdrant running with embedded chunks (`docker compose up -d`, feature
  engineering done).
- A SageMaker endpoint created (Session 5.3), or a local stand-in.
- `COMET_API_KEY` and `COMET_PROJECT` set if you want tracing.

### Step 2: Start the service

```bash
poetry poe run-inference-ml-service
# equivalent to:
python -m tools.ml_service
# Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Call the endpoint

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"My name is Paul Iusztin. Write a LinkedIn post about RAG.\"}"
```

Expected:

```json
{ "answer": "RAG (Retrieval-Augmented Generation) is ..." }
```

The repo's own example prompt is available as:

```bash
poetry poe call-inference-ml-service
```

### Step 4: Inspect the OpenAPI schema

Open `http://localhost:8000/docs` for the interactive Swagger UI. The
`QueryRequest` and `QueryResponse` models are documented automatically. The raw
schema is at `http://localhost:8000/openapi.json`.

### Step 5: Validate error handling

```bash
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d "{}"
# 422 Unprocessable Entity (missing 'query')
```

### Step 6: Check the trace

Open the Opik/Comet project and confirm a trace tagged `rag` with metadata for
`model_id`, `embedding_model_id`, `temperature`, and the three token counts.

---

## 📝 Exercise 1: Add a Health and a Streaming Endpoint

### Task 1: Health check

```python
@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Task 2: Streaming (sketch)

Add `POST /rag/stream` using `StreamingResponse` from `fastapi.responses`:

```python
from fastapi.responses import StreamingResponse


@app.post("/rag/stream")
async def rag_stream(request: QueryRequest):
    def token_gen():
        answer = rag(query=request.query)
        for token in answer.split():
            yield token + " "

    return StreamingResponse(token_gen(), media_type="text/plain")
```

**Goal**: Understand how the current design (blocking sync function) makes true
token streaming hard, and where you would need an async inference client. The
above only streams the words of a fully generated answer; real token streaming
requires TGI's SSE endpoint and an async consumer.

---

## 📝 Exercise 2: Offload the Blocking Call and Prove Concurrency

### Task

1. Reproduce the problem: add a temporary `time.sleep(3)` inside `rag()` and fire
   two concurrent requests:

   ```bash
   curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d "{\"query\": \"a\"}" &
   curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d "{\"query\": \"b\"}" &
   ```

   Note the wall-clock time: the current `async` handler serializes them (about 6
   seconds).
2. Change `async def rag_endpoint` to `def rag_endpoint` and repeat. Observe the
   two requests overlapping (about 3 seconds).
3. Alternatively keep `async` and wrap the call:

   ```python
   from starlette.concurrency import run_in_threadpool
   answer = await run_in_threadpool(rag, request.query)
   ```

4. Explain why the token counts and trace metadata are unaffected by this change.

**Goal**: Internalize that `async def` is not automatically concurrent; the
handler must not block the loop.

---

## 🐛 Common Pitfalls

- **Blocking the event loop**: `rag()` is synchronous. One slow SageMaker call
  blocks other requests. Use a `def` handler or a thread pool under concurrency.
- **500 leaks internals**: `detail=str(e)` exposes stack-derived messages. Log
  them server-side instead.
- **Missing env**: `LLMInferenceSagemakerEndpoint` needs AWS credentials; without
  them boto3 fails at request time.
- **`response_model` mismatch**: returning a key other than `answer` will fail
  validation.
- **CORS**: browser clients need `CORSMiddleware`; it is not configured by
  default. Without it, the browser blocks cross-origin calls.
- **Reusing one `LLMInferenceSagemakerEndpoint` across requests**: `set_payload`
  mutates the instance. The app creates a fresh instance per call, which is
  correct. Do not hoist it to a module global without adding locking.
- **`configure_opik()` import-time side effect**: if Opik is misconfigured it
  warns, but a hard failure in the private workspace call would surface on
  import. The `try/except` guards this.
- **`0.0.0.0` binding**: convenient locally and in Docker, unsafe on an open
  network. Bind `127.0.0.1` outside containers.
- **`reload=True` in production**: file-watch overhead and restarts; disable it
  when not developing.
- **`k=3` vs larger k**: more retrieved chunks mean a longer prompt, more tokens,
  higher latency, and possible context dilution. Tune `k` deliberately.
- **No request timeout**: a hung endpoint call ties up the request. boto3 clients
  accept timeouts and retries; configure them for production.

---

## 🎓 Knowledge Check

1. **What two methods must any `Inference` implementation provide?**
   - Answer: `set_payload` and `inference`.

2. **What request shape does the SageMaker TGI container expect?**
   - Answer: `{"inputs": "...", "parameters": {...}}`.

3. **What does `return_full_text=False` do?**
   - Answer: Returns only the generated continuation, not the prompt prefix.

4. **Why are `QueryRequest` and `QueryResponse` Pydantic models?**
   - Answer: They validate input/output and generate the OpenAPI schema.

5. **What is the main scalability weakness of this app?**
   - Answer: A synchronous blocking call inside an async endpoint, which stalls
     the event loop.

6. **How is tracing turned on?**
   - Answer: `configure_opik()` at import plus `@opik.track` decorators.

7. **What is the difference between the `sagemaker` and `sagemaker-runtime`
   clients?**
   - Answer: `sagemaker` is the control plane (create/delete endpoints);
     `sagemaker-runtime` invokes an existing endpoint.

8. **How many chunks does the repo retrieve per query, and how does that compare
   to the book?**
   - Answer: The repo uses `k=3`; the book snippet prints `k=3 * 3`.

9. **What does `InferenceComponentName` control, and when is it added?**
   - Answer: It targets a specific inference component on an endpoint; it is
     added only when set and not `"None"`.

10. **Where does `answer` come from in the TGI response?**
    - Answer: `response[0]["generated_text"]`, the first completion.

11. **What does `EmbeddedChunk.to_context` produce?**
    - Answer: A numbered string of chunk type, platform, author, and content for
      each retrieved chunk.

12. **Why is the API server cheap to run?**
    - Answer: Retrieval and embedding are CPU/network bound and generation runs
      on the remote GPU endpoint, so no local GPU is needed.

13. **What HTTP status does a missing `query` produce, and why does it matter
    cost-wise?**
    - Answer: 422; validation fails before the handler runs, so no LLM call is
      made.

14. **What is the risk of returning `detail=str(e)` from the 500 handler?**
    - Answer: It may leak internal details to the client.

15. **What single change makes the handler concurrency-safe with the least
    code?**
    - Answer: Change `async def rag_endpoint` to `def rag_endpoint` (Starlette
      runs it in a threadpool), or offload with `run_in_threadpool`.

---

## 📖 Glossary

- **ASGI**: Asynchronous Server Gateway Interface; the async contract between a
  Python web app and its server.
- **Uvicorn**: The ASGI server that runs the FastAPI app.
- **FastAPI**: The web framework providing routing, validation, and OpenAPI docs.
- **Pydantic**: The validation library behind FastAPI request/response models.
- **RAG**: Retrieval-Augmented Generation; retrieve context, then generate.
- **InferenceExecutor**: The glue that formats a prompt, sets the payload, calls
  the model, and extracts the answer.
- **Inference (ABC)**: The domain contract with `set_payload` and `inference`.
- **TGI**: Hugging Face Text Generation Inference engine behind the endpoint.
- **Opik**: The tracing tool (powered by Comet ML) used for prompt monitoring.
- **Event loop**: The single async thread that runs coroutines; blocking it
  stalls all requests.
- **422 Unprocessable Entity**: The status for a well-formed request with invalid
  data.

---

## 🔗 Next Session

**Session 6.2**: RAG Inference Flow

We trace a query from retrieval through context building to generation, and
inspect the Opik trace metadata.

See [Session 6.2: RAG Inference Flow](session_6.2_rag_inference_flow.md).

---

## 📚 Additional Resources

- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)
- [Uvicorn](https://www.uvicorn.org/)
- [Opik Tracing](https://www.comet.com/docs/opik/)
- [Starlette run_in_threadpool](https://www.starlette.io/concurrency/)
- Related sessions: [Session 5.3 SageMaker Deployment](session_5.3_sagemaker_deployment.md),
  [Session 6.2 RAG Inference Flow](session_6.2_rag_inference_flow.md)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 4.1, 4.2, 5.3

**Outcome**: You can run, call, validate, and extend the inference API, explain
its domain abstractions, and fix its blocking-handler weakness.
