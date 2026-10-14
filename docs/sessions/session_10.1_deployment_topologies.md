# Session 10.1: Deployment Topologies & Autoscaling (Book Chapter 10)

## 🎯 Learning Objectives

By the end of this session, you will:
- Reason about the four deployment requirements: throughput, latency, data, infrastructure
- Choose between online real-time, asynchronous, and offline batch deployment types
- Decide between monolithic and microservices architectures for model serving
- Understand the LLM Twin's two-microservice design
- Configure SageMaker autoscaling (scalable target + target tracking)
- Compute cost and latency tradeoffs with worked numbers
- Recognize the failure modes of each topology

> This session completes **Book Chapter 10: Inference Pipeline Deployment** (pp. 384-428). The hands-on SageMaker/FastAPI code is in `session_5.3_sagemaker_deployment.md` and `session_6.1_fastapi_api.md`; this session covers the architectural decision framework around it.

---

## 🏗️ Architecture Overview

```
Decision flow
  (1) What are my throughput / latency / data / infrastructure needs?
        │
  (2) Which deployment type?
        ├── Online real-time      client waits, synchronous
        ├── Asynchronous          queue + poll/push, decoupled
        └── Offline batch         pull-process-store, high throughput
        │
  (3) Monolith or microservices?
        ├── Monolith              one service, simple, harder to scale parts
        └── Microservices         split LLM vs business logic, scale independently
        │
  (4) Autoscale with a scalable target + target-tracking policy
```

### The full LLM Twin topology

```
                          ┌─────────────────────────────────────────────┐
                          │  Business microservice  (FastAPI)            │
 User ──HTTP request──►   │  1. receive query                            │
                          │  2. RAG retrieval   ──► Qdrant vector DB      │
                          │  3. advanced RAG (pre/retrieval/post)         │
                          │  4. prompt assembly                           │
                          │  5. call LLM service ───────HTTP──────────┐   │
                          │  6. wait for answer                        │   │
                          │  7. prompt monitoring ──► Opik             │   │
                          │  8. return answer to user                  │   │
                          └────────────────────────────────────────────┼───┘
                                                                       │
                          ┌────────────────────────────────────────────▼───┐
                          │  LLM microservice  (AWS SageMaker endpoint)     │
                          │   Hugging Face TGI DLC, quantized TwinLlama     │
                          │   strictly generation                           │
                          └─────────────────────────────────────────────────┘
```

This is an **online real-time** deployment with a **microservices** split: the retriever and business logic live in a CPU-oriented FastAPI service, and the GPU-bound model lives behind a SageMaker endpoint.

---

## 📁 Criteria for Choosing a Deployment Type

Four interacting requirements drive every decision: **throughput, latency, data, infrastructure**.

- **Throughput**: requests per second (RPS). High throughput needs scalable clusters and multiple GPUs.
- **Latency**: time to process one request (network I/O + serialization + inference). Low latency needs faster hardware, sometimes edge deployment.
- **The batching tension**: for non-batched systems, lower latency yields higher throughput; but with batching, higher latency can yield higher throughput (more requests per batch). Always respect a minimum acceptable latency.
- **Data**: input/output format, volume, and complexity shape latency and memory needs (text vs images vs tabular).
- **Infrastructure**: GPUs, memory, storage, networking. Optimizing for low latency often leaves the GPU underutilized, raising cost per request.

### Throughput vs latency, worked numbers

Without batching, latency and throughput are inverses:

```
latency 100 ms/request → throughput 10 RPS
latency  10 ms/request → throughput 100 RPS
```

With batching, the relationship flips because more requests ride in one forward pass:

```
20 batched requests in 100 ms → latency 100 ms, throughput 200 RPS
60 batched requests in 200 ms → latency 200 ms, throughput 300 RPS
```

Higher latency bought higher throughput. The constraint is the minimum acceptable latency for the user; below that, batching is unacceptable no matter the throughput gain.

```
Throughput
  (RPS)
   300 │                    ●  (60 req / 200 ms)
       │                  ╱
   200 │            ●  (20 req / 100 ms)
       │          ╱
   100 │        ╱
       │      ╱
     0 └──────────────────────────────► Latency (ms)
        0    100    200    300
```

### Questions to ask before choosing

- Throughput requirements (min/average/max)?
- Concurrent requests (1, 10, 1,000, 1 million)?
- Latency requirement (1 ms, 10 ms, 1 s)?
- How should it scale (CPU load, request count, queue size, data size)?
- What are the cost requirements?
- Data type and size (text, images; 100 MB, 1 GB, 10 GB)?

A useful anchor: Google found 53% of mobile visits are abandoned after ~3 seconds of load time.

**Why the user experience dominates**: shipping a brilliant model with high latency loses to a mediocre model that responds reliably. The book's framing is blunt - if the user cannot interact with it, the model has near-zero business value.

---

## 📁 The Three Deployment Types

### 1. Online Real-Time Inference

```
Client ──HTTP request──► ML service ──immediate result──► Client
```

- **REST** (JSON): accessible, slower; used for public APIs (e.g. OpenAI).
- **gRPC** (protobuf): faster, less flexible; used for internal services.
- **Streaming** with **WebSockets / Server-Sent Events (SSE)** for token-by-token output (ChatGPT, Claude).
- Needs load balancing, autoscaling, and high availability.
- **Weakness**: hard to scale; idle resources at low traffic.

**REST vs gRPC, in detail**:

| Aspect | REST (JSON) | gRPC (protobuf) |
|--------|-------------|-----------------|
| Payload | JSON text | compiled binary protobuf |
| Human-readable | yes | no |
| Wire speed | slower | faster |
| Client ergonomics | trivial (curl, any HTTP lib) | needs a protobuf schema + codegen |
| Best for | public / external APIs | internal service-to-service |

**When real-time is the right call**: chatbots, embeddings/reranking inside RAG, online recommenders (e.g. TikTok). The synchronous interaction - client waits for the answer - matches how users expect a chatbot to behave.

### 2. Asynchronous Inference

```
Client ──request──► Queue ──► ML service (later) ──► Result store
   ▲                                                    │
   └──────────────── poll / push notification ◄─────────┘
```

- The client is not blocked; requests are queued and processed at the service's pace.
- **Handles spikes** without scaling VMs linearly (e.g. 10 to 100 RPS on the same two machines).
- Good for long jobs (>5 minutes). **Costs**: higher latency and added complexity.

**The e-shop spike worked example** (from the book):

```
normal:   10 RPS  handled by 2 VMs
promotion: 100 RPS spike

Option A (scale VMs):   2 → 10 VMs, drastic cost
Option B (queue):       keep 2 VMs, let the queue absorb the spike,
                        process at the VMs' own rhythm, no timeouts
```

The queue is a shock absorber. You trade freshness for cost stability.

**When asynchronous is the right call**: extracting keywords, summarizing documents, deep-fake generation over videos, and any job that exceeds a comfortable wait. With a carefully tuned autoscaler it can also serve online recommendations.

### 3. Offline Batch Transform

```
Storage ──pull──► ML service ──process batch──► Storage ──► Client
```

- High throughput, permissive latency, cheapest per request, simplest to implement.
- Client decoupled from the service; results may be stale by design.
- Ideal for analytics, periodic reporting, and recommendations where a delay is acceptable.

**Freshness worked example** (from the book):

| Use case | Acceptable delay? |
|----------|-------------------|
| Video-streaming recommendations | one day is fine (low consumption frequency) |
| Social-media recommendations | one day or even one hour is unacceptable (constant freshness) |

The same batch architecture serves one and fails the other purely on freshness.

**Storage as a cache**: the results store acts like a large cache the client reads from. Clients can be notified when processing completes to become more responsive, but a delay always exists between computation and consumption.

**Comparison table**:

| Aspect | Online real-time | Asynchronous | Offline batch |
|--------|------------------|--------------|---------------|
| Client waits? | yes | no | no |
| Latency | lowest | medium-high | highest |
| Throughput | medium | high | highest |
| Cost per request | highest | medium | lowest |
| Gets latest input? | yes | yes (queued) | no (scheduled) |
| Implementation complexity | medium | high | lowest |
| Best for | chatbots, RAG | long jobs, spikes | analytics, reporting |

---

## 📁 Monolithic vs Microservices Serving

### Monolithic

LLM + business logic (pre/post-processing) in one service.

- **Pros**: simple to build and maintain; good for small teams and MVPs.
- **Cons**: cannot scale parts independently. The LLM wants GPU; the business logic is CPU/I/O-bound. Bundling them wastes GPU time (idle while business logic runs) and forces one tech stack.

```
Monolith

  ┌─────────────────────────────────────────────┐
  │  one service, one machine                    │
  │  [ preprocess ] → [ LLM (GPU) ] → [ post ]   │
  │        CPU            GPU            CPU      │
  └─────────────────────────────────────────────┘
  Problem: the GPU idles while CPU steps run, and vice versa.
```

### Microservices

Split the LLM service from the business logic; communicate over REST or gRPC.

- **Pros**: scale each independently (more GPU replicas for the LLM, cheap CPU for business logic); each service chooses its own stack.
- **Cons**: more deployment/monitoring complexity; network latency and more failure points.

```
Microservices

  ┌───────────────────────┐        ┌───────────────────────┐
  │ Business service (CPU)  │  REST  │  LLM service (GPU)     │
  │ pre/post, retrieval      │◄──────►│  generation only       │
  │ scale N cheap replicas   │        │  scale M GPU replicas   │
  └───────────────────────┘        └───────────────────────┘
  Each side scales to its own bottleneck.
```

**Choosing**: monolith for smaller teams and simple models that do not need GPU; microservices for larger systems with GPU-heavy LLMs. **Start monolithic but design for modularity** (separate modules/packages with clean interfaces) so you can split later without a rewrite.

**The modular monolith technique**: even on one machine, separate the ML and business logic into two Python modules that do not import each other, then glue them at a higher level (a service class, or the FastAPI layer). Better still, two distinct Python packages. When the time comes to split, you move a package instead of rewriting logic.

---

## 📁 The LLM Twin's Deployment Strategy

The project targets a **low-latency content-creation chatbot**, so it selects **online real-time inference** with a **microservices** split:

```
User ──► Business microservice (FastAPI)          LLM microservice (SageMaker)
           ├── RAG retrieval (Qdrant)               └── TGI DLC, quantized model
           ├── advanced RAG techniques                      ▲
           ├── prompt assembly ────────────HTTP────────────┘
           └── prompt monitoring (Opik)
```

- **Business microservice** (`tools/ml_service.py` + `llm_engineering/infrastructure/inference_pipeline_api.py`): owns RAG retrieval and augmentation, prompt assembly, and monitoring.
- **LLM microservice** (`llm_engineering/model/inference/inference.py` + the SageMaker endpoint): strictly generation, optimized by the TGI DLC.

**Why microservices**: an 8B model can serve on one `ml.g5.2xlarge` (A10G) after quantization; a 30B model needs an A100. Splitting lets you upgrade **only** the LLM service.

### The request flow, step by step

```
1. User ──HTTP──► FastAPI /rag
2. ContextRetriever.search(query, k=3) ──► Qdrant
3. EmbeddedChunk.to_context(documents)   → context string
4. InferenceExecutor(llm, query, context).execute()
5. prompt = template.format(query, context)
6. llm.set_payload(...) ──HTTP──► SageMaker endpoint
7. SageMaker TGI generates ──HTTP──► answer
8. opik_context.update_current_trace(tags=["rag"], metadata={...})
9. HTTP 200 {"answer": ...} ──► user
```

### The repo implementation

The FastAPI entry point (`llm_engineering/infrastructure/inference_pipeline_api.py`):

```python
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
- **`FastAPI` + Pydantic** give typed request/response contracts (`QueryRequest`, `QueryResponse`) and automatic validation.
- **`@opik.track`** wraps both the outer `rag` call and the inner `call_llm_service`, producing a trace tree with token counts as metadata - this is the monitoring seam (Session 7.2).
- **`ContextRetriever(mock=False)`** performs real retrieval from Qdrant with `k=3` documents.
- **The endpoint is `async`** but the service call inside is synchronous; FastAPI runs it in a threadpool. The LLM call is the latency bottleneck, so the design keeps it in one hop.

The inference client (`llm_engineering/model/inference/inference.py`) is the HTTP boundary to the LLM microservice:

```python
class LLMInferenceSagemakerEndpoint(Inference):
    """
    Class for performing inference using a SageMaker endpoint for LLM schemas.
    """

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
- **`boto3.client("sagemaker-runtime")`** is the data plane; `invoke_endpoint` sends the request.
- **`InferenceComponentName`** is added only when set - this is what enables per-component autoscaling (see below). When `None`, the call targets the endpoint as a whole.
- **The payload format** (`inputs` + `parameters`) is the Hugging Face TGI contract used by the DLC.
- **The class implements the `Inference` ABC** (`llm_engineering/domain/inference.py`), so it is interchangeable with any other backend behind the same interface.

The prompt assembly lives in `InferenceExecutor` (`llm_engineering/model/inference/run.py`):

```python
class InferenceExecutor:
    def __init__(self, llm: Inference, query: str, context: str | None = None, prompt: str | None = None) -> None:
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
- **The prompt template is the seam** where retrieval results become model input. Changing it changes behavior without touching the model.
- **`repetition_penalty=1.1`** discourages loops, a common failure in long generations.
- **`self.llm.inference()[0]["generated_text"]`** reads the TGI response shape: a list of objects with a `generated_text` field.
- **The `Inference` interface** lets `InferenceExecutor` accept any backend (SageMaker today, a local vLLM server tomorrow).

**SageMaker components**:
- **Endpoint**: the hosted API.
- **Model**: weights + computation logic.
- **Configuration**: instance type/count.
- **Inference component**: binds a model + config to an endpoint, and supports per-component autoscaling.

**Other clouds**: Azure OpenAI + Azure ML, Vertex AI, BentoML, Seldon, Modal, Hopsworks. The concepts transfer; only the tooling changes.

### Serving stack: why the Hugging Face TGI DLC

The model runs inside a Hugging Face Deep Learning Container (DLC), powered by Text Generation Inference (TGI). TGI provides:

- **Tensor parallelism** for larger models across GPUs.
- **Optimized transformers code** with FlashAttention for major architectures.
- **bitsandbytes quantization** to shrink the weights.
- **Continuous batching** to improve throughput as requests arrive.
- **safetensors** weight loading for faster startup.
- **Token streaming** over Server-Sent Events (SSE).

These are exactly the inference optimizations from Chapter 8; the DLC packages them so the deployment does not re-implement them.

### Training vs Inference Pipelines

| Aspect | Training pipeline | Inference pipeline |
|--------|-------------------|--------------------|
| Data access | offline batch, throughput-optimized (ZenML artifacts) | online DB, low latency (Qdrant) |
| Output | model weights (registry) | predictions to the user |
| Infrastructure | many GPUs, gradients in memory | fewer GPUs, no optimization step |
| Shared | pre/post-processing MUST match, or you get **training-serving skew** |

**Why training-serving skew is insidious**: the preprocessing that frames a training example and the preprocessing that frames an inference request are written in different files, often months apart. A whitespace change or a tokenizer-config drift can silently shift the input distribution. The defense is to import the same function in both paths, never copy-paste it.

---

## 📁 Autoscaling

SageMaker autoscaling uses **Application Auto Scaling** with two steps:

1. **Register a scalable target**: defines what to scale and the min/max capacity.
   - Resource: `inference-component/<name>`
   - Scalable dimension: `sagemaker:inference-component:DesiredCopyCount`
   - `MinCapacity` / `MaxCapacity`
2. **Create a scaling policy**: defines *how* to scale.
   - **Target tracking** on `SageMakerInferenceComponentInvocationsPerCopy`.
   - Cooldowns (scale-in/out) prevent flapping.

### The two concepts

Registering the target answers *"what can scale, and within what bounds?"* It does not decide when. The policy answers *"how and when to scale?"* by watching a metric against a target. You need both: a policy without a target has nothing to scale, and a target without a policy never changes.

```python
class AutoscalingSagemakerEndpoint:
    def setup_autoscaling(self):
        ScalableTarget(
            auto_scaling_client=self.auto_scaling_client,
            service_namespace="sagemaker",
            resource_id=f"inference-component/{self.inference_component_name}",
            scalable_dimension="sagemaker:inference-component:DesiredCopyCount",
            min_capacity=self.initial_copy_count,
            max_capacity=self.max_copy_count,
        ).register()

        TargetTrackingScalingPolicy(
            auto_scaling_client=self.auto_scaling_client,
            policy_name=self.endpoint_name,
            service_namespace="sagemaker",
            resource_id=self.resource_id,
            scalable_dimension=self.scalable_dimension,
            target_value=self.target_value + 1,
            scale_in_cooldown=200,
            scale_out_cooldown=200,
        ).apply_policy()
```

The full implementation is in `llm_engineering/infrastructure/aws/deploy/autoscaling_sagemaker_endpoint.py` (Session 5.3). Key defaults from the class:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `initial_copy_count` | 1 | MinCapacity - never scale below |
| `max_copy_count` | 6 | MaxCapacity - cost ceiling |
| `target_value` | 4.0 | Target invocations per copy |
| applied target | `target_value + 1` = 5.0 | What is actually sent to the policy |
| `scale_in_cooldown` | 200 s | Wait before removing a copy |
| `scale_out_cooldown` | 200 s | Wait before adding a copy |
| metric | `SageMakerInferenceComponentInvocationsPerCopy` | Predefined target-tracking metric |
| resource_id | `inference-component/<name>` | The scalable resource |

**The target-tracking loop, conceptually**:

```
every minute:
  observed = InvocationsPerCopy
  if observed > target: add copies (up to max), then wait scale_out_cooldown
  if observed < target: remove copies (down to min), then wait scale_in_cooldown
```

**Autoscaling requires `INFERENCE_COMPONENT_BASED` endpoints.** A model-based endpoint cannot be scaled by the copy autoscaler - this is the single most common misconfiguration. Cooldowns should exceed cold-start time; LLM containers are slow to boot (the project allows a 900-second health-check timeout).

**Why cooldowns must exceed cold start**: if the cooldown is shorter than the time a new container needs to become healthy, the autoscaler sees the load still unmet mid-boot and adds *more* copies. This "thundering herd" can double or triple your instance count and your bill. A 200-second cooldown assumes the container is up within that window; if not, raise it.

### Autoscaling cleanup

```python
    def cleanup_autoscaling(self):
        self.auto_scaling_client.delete_scaling_policy(
            PolicyName=self.endpoint_name,
            ServiceNamespace=self.service_namespace,
            ResourceId=self.resource_id,
            ScalableDimension=self.scalable_dimension,
        )

        self.auto_scaling_client.deregister_scalable_target(
            ServiceNamespace=self.service_namespace,
            ResourceId=self.resource_id,
            ScalableDimension=self.scalable_dimension,
        )
```

**Order matters**: delete the policy first, then deregister the target. If you deregister first, the policy is orphaned and may keep trying to act on a target that no longer exists.

---

## 📊 Cost Model: Worked Example

Assume `ml.g5.2xlarge` at roughly **$1.5/hour** (spot/on-demand varies by region).

| Strategy | Monthly cost | Latency profile |
|----------|--------------|-----------------|
| One always-on replica | `1.5 × 24 × 30` = **$1,080** | lowest, constant |
| Two always-on replicas | **$2,160** | lower under load, more headroom |
| Always-on + autoscale to 6 | $1,080 base, up to **$6,480** peak | absorbs spikes |
| Scaled-to-min (1 copy idle) + async queue | $1,080 plus queue infra | high latency, low cost per request |

**Break-even reasoning**: at 10 RPS the always-on replica is underutilized. If the workload tolerates a five-minute delay, an asynchronous or batch design that runs a single replica only during bursts can cut cost dramatically because the expensive GPU is not billed while idle.

**The batching cost lever**: batch transforms process many items per billed second, so the *cost per request* falls even if *latency* rises. This is the core tradeoff of the whole chapter.

**Why cooldowns must exceed the 900-second startup** (the exercise): if the container takes up to 900 seconds to become healthy but the scale-out cooldown is only 200 seconds, the autoscaler will fire repeatedly before the first new copy is serving, over-provisioning the endpoint.

---

## 🛠️ Hands-On: Map a Requirement Set to a Topology

### Step 1: Answer the four pillars for three products

```
A. Real-time code completion      latency < 200 ms, high RPS, text
B. Nightly recommendation batch    latency irrelevant, throughput huge
C. Document summarization scale    per-doc can take minutes, spiky
```

### Step 2: For each, choose a deployment type

- A → **online real-time** (streaming, microservices; LLM on GPU).
- B → **offline batch transform** (pull from S3, write back to S3).
- C → **asynchronous** (queue, poll/push when done).

### Step 3: Decide monolith vs microservice

For A and C with a GPU LLM, prefer **microservices**. For B with a non-GPU model, a **monolith** is usually fine.

### Step 4: Sketch the autoscaling policy

Pick a target metric (invocations per copy) and min/max copies that bound cost.

### Step 5: Walk the LLM Twin request path

Start the service and trace one request through the monitoring trace:

```bash
python -m tools.ml_service
# then, in another shell:
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d "{\"query\": \"What is an LLM Twin?\"}"
```

Open the Opik trace and confirm the `rag` → `call_llm_service` span tree with token metadata.

---

## 📝 Exercise: Cost vs Latency Tradeoff

### Task

For the LLM Twin endpoint, reason quantitatively.

1. With `ml.g5.2xlarge` at ~$1.5/hour, compute the monthly cost of one always-on replica.
2. Compare an always-on replica (low latency) with a scaled-to-zero/batch approach (high latency).
3. Decide the traffic level at which asynchronous or batch becomes cheaper.
4. Explain why autoscaling cooldowns must exceed the 900-second container startup.

**Goal**: Connect topology and autoscaling choices to real money.

### Second Exercise: Break the Autoscaler on Paper

Given `initial_copy_count=1`, `max_copy_count=6`, `target_value=5.0`, cooldown 200 s, and a cold start of 900 s:

1. Traffic spikes to 50 invocations per copy.
2. The autoscaler adds copies every 200 seconds up to the cap of 6.
3. Draw the copy count over time from t=0 to t=1800 s, assuming each new copy needs 900 s to serve.
4. Compute the wasted spend from copies that are booting but not yet serving.
5. Propose two fixes: raise the cooldown, or pre-warm a larger min capacity. State the tradeoff of each.

**Goal**: Learn that autoscaling parameters are a function of cold-start time, not arbitrary constants.

---

## 🔀 Design Alternatives and Edge Cases

### Choosing a deployment type by requirement

| Requirement dominates | Choose | Why |
|-----------------------|--------|-----|
| Immediate feedback (chatbot) | online real-time | synchronous, lowest latency |
| Huge throughput, delay OK | offline batch | cheapest per request |
| Spiky traffic, cost-sensitive | asynchronous | queue absorbs spikes |
| Fresh predictions required | online real-time | batch is stale by design |
| Per-item work > 5 minutes | asynchronous | client cannot block that long |
| No GPU, small model | monolith is fine | no independent scaling gain |

### Monolith vs microservices decision matrix

| Factor | Favors monolith | Favors microservices |
|--------|-----------------|----------------------|
| Team size | small | large / multiple teams |
| GPU need | none or cheap | expensive GPU |
| Scaling needs | uniform | components scale differently |
| Tech stack | single language | mixed (Rust model + Python logic) |
| Ops maturity | low | high |
| Time to first deploy | short | longer |

### Edge cases to plan for

- **Traffic turns to zero**: real-time endpoints still bill while idle. If the workload is bursty and delay-tolerant, prefer async or batch; otherwise accept the idle cost as the price of latency.
- **Model swap without downtime**: because the LLM is its own service, you can update the endpoint's model/config and leave the FastAPI business service untouched. In a monolith, that requires redeploying everything.
- **Retriever outage**: in the microservice split, if Qdrant is down the business service fails before ever calling the LLM. Add a degraded path (answer without context) or fail fast with a clear error.
- **Autoscaler thrash**: if the target metric is noisy, copies oscillate. Raise the cooldown or switch to a step-scaling policy with wider thresholds.
- **Cross-service latency budget**: the end-to-end latency is `retrieval + prompt assembly + network hop + generation + post-processing`. The network hop to SageMaker is small relative to generation, but it is not zero; measure it.
- **Per-component vs endpoint scaling**: only inference-component-based endpoints support `DesiredCopyCount`. A model-based endpoint must be scaled differently, or migrated.

### Repo-vs-book note (accuracy)

The repository's `AutoscalingSagemakerEndpoint` default `target_value=4.0` is applied as `self.target_value + 1` in `setup_autoscaling`, so the policy receives **5.0**. The code comment calls this "Example adjustment, should be based on specific use case." Treat the value as a tunable starting point, not a fixed rule. Likewise, the client adds `InferenceComponentName` to the invoke call only when `inference_component_name` is set and not the literal string `"None"`.

---

## 🐛 Common Pitfalls

- **Scaling a monolith to fix a GPU bottleneck**: the GPU stays idle when business logic runs. Split services instead.
- **Autoscaling a model-based endpoint**: only inference-component-based endpoints support the copy autoscaler.
- **Ignoring the batching/latency tension**: raising throughput via batching can break a latency SLA.
- **Training-serving skew**: preprocessing that differs between training and inference silently degrades quality. Reuse the same code.
- **Cold starts**: LLM containers can take minutes; set cooldowns and health-check timeouts accordingly.
- **Cooldown shorter than cold start**: causes over-provisioning as the autoscaler re-fires before new copies serve.
- **Deregistering the scalable target before deleting the policy**: orphans the policy. Delete policy first.
- **Adding tokens without bounding cost**: `InvocationComponentName` autoscaling scales to `max_copy_count`; if the ceiling is too high, a spike can blow the budget.
- **Synchronous blocking in FastAPI**: the LLM call is synchronous; if you add heavy CPU work in the endpoint, you block the event loop's threadpool. Offload or make it truly async.

---

## 🎓 Knowledge Check

1. **What are the four deployment requirements?**
   - Answer: throughput, latency, data, and infrastructure.

2. **When is asynchronous inference the best fit?**
   - Answer: spiky traffic or long jobs where a latency delay is acceptable and cost matters.

3. **Why does the LLM Twin choose microservices?**
   - Answer: to scale the GPU-heavy LLM service independently from the CPU-bound business logic.

4. **What is training-serving skew?**
   - Answer: a mismatch in preprocessing/postprocessing between training and inference that hurts quality.

5. **Which SageMaker endpoint type supports copy autoscaling?**
   - Answer: inference-component-based endpoints.

6. **What two steps does autoscaling require?**
   - Answer: register a scalable target, then create a scaling policy (target tracking).

7. **How do batching and latency relate under batching?**
   - Answer: with batching, higher latency can yield higher throughput because more requests ride in one forward pass.

8. **What is the scalable dimension for SageMaker inference components?**
   - Answer: `sagemaker:inference-component:DesiredCopyCount`.

9. **Which metric does the target-tracking policy watch?**
   - Answer: `SageMakerInferenceComponentInvocationsPerCopy`.

10. **Why must autoscaling cooldowns exceed cold-start time?**
    - Answer: Otherwise the autoscaler adds copies faster than containers become healthy, over-provisioning the endpoint.

11. **What is REST vs gRPC for real-time serving?**
    - Answer: REST/JSON is accessible but slower and public-facing; gRPC/protobuf is faster but needs schemas and is used internally.

12. **Why is offline batch the cheapest per request?**
    - Answer: large batches maximize GPU utilization per billed second and tolerate high latency.

13. **What does `InferenceComponentName` enable in the client?**
    - Answer: targeting a specific inference component so per-component autoscaling applies.

14. **Why split a monolith into modules before splitting into services?**
    - Answer: clean software seams let you move a module to its own service later without a rewrite.

15. **What is the LLM microservice's single responsibility?**
    - Answer: generation only; retrieval, augmentation, and prompt assembly live in the business microservice.

---

## 📖 Glossary

- **Throughput**: inference requests processed per second (RPS).
- **Latency**: time to process one request from receipt to result.
- **Batching**: grouping requests into one forward pass; raises throughput, raises latency.
- **Online real-time inference**: synchronous request/response serving.
- **Asynchronous inference**: queue-backed serving where the client does not block.
- **Offline batch transform**: pull-process-store serving on a schedule.
- **Monolith**: one service containing model and business logic.
- **Microservices**: independent services communicating over REST or gRPC.
- **Training-serving skew**: a preprocessing mismatch between training and serving.
- **Scalable target**: the registered resource and its min/max capacity bounds.
- **Target tracking**: a scaling policy that maintains a metric at a target value.
- **Cooldown**: the wait between successive scaling actions.
- **Inference component**: a SageMaker binding of model + config to an endpoint.
- **DLC**: Deep Learning Container; the prebuilt TGI image that runs the model.
- **TGI**: Text Generation Inference; the serving engine inside the DLC.
- **SSE**: Server-Sent Events; token streaming over HTTP.

---

## 🔗 Next Session

**Session 11.1 (book Chapter 11)**: MLOps and LLMOps - see `session_11.1_ct_pipeline_alerting.md` plus `session_7.1_comet_ml.md`, `session_7.2_opik_monitoring.md`, `session_8.1_docker.md`, `session_8.2_cicd.md`, `session_8.3_zenml.md`.

---

## 📚 Additional Resources

- [SageMaker Autoscaling](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
- [Hugging Face DLCs](https://huggingface.co/docs/sagemaker/inference)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [Chip Huyen, Building a Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html)
- [Google mobile site load-time statistics](https://www.thinkwithgoogle.com/consumer-insights/consumer-trends/mobile-site-load-time-statistics/)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 6.1, 6.2, 5.3

**Outcome**: You can choose a deployment topology from requirements and configure SageMaker autoscaling with a clear cost-and-latency rationale.
