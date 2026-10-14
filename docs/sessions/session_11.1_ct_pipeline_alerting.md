# Session 11.1: MLOps, LLMOps, and Continuous Training (Book Chapter 11)

## 🎯 Learning Objectives

By the end of this session, you will:
- Trace the lineage DevOps → MLOps → LLMOps
- Name the three dimensions of an ML application: code, data, model
- Distinguish CI/CD from CT, and trigger a downstream pipeline
- Deploy the pipelines to the cloud (MongoDB, Qdrant, ZenML, AWS)
- Add prompt monitoring and alerting to a production RAG service
- Design guardrails and a human-feedback loop around an LLM service
- Reason about which CT trigger (manual, event, schedule) fits which system

> This session completes **Book Chapter 11: MLOps and LLMOps** (pp. 430-488). CI/CD mechanics are in [`session_8.2_cicd.md`](session_8.2_cicd.md); Docker/ZenML infra is in [`session_8.1_docker.md`](session_8.1_docker.md) and [`session_8.3_zenml.md`](session_8.3_zenml.md); Opik basics are in [`session_7.2_opik_monitoring.md`](session_7.2_opik_monitoring.md). This session adds the LLMOps theory, the **CT pipeline**, **cloud deployment**, and **alerting** the earlier curriculum omitted.

---

## 🏗️ Architecture Overview

```
DevOps  ──►  MLOps  ──►  LLMOps
(code)       (code+data+model)   (+ prompts, guardrails, human feedback)

CI  (test/build on PR)          ──► code dimension
CD  (build image, push ECR)     ──► code dimension
CT  (orchestrate pipelines)     ──► data + model dimensions   ← the AI-only part

Production pillars
├── Prompt monitoring (Opik): config, tokens, latency, feedback
├── Guardrails + human-in-the-loop
└── Alerting (ZenML alerter): on failure and on success
```

The three dimensions are not a taxonomy; they are three independent sources of change. Any of them changing can break production, and each has its own automation.

```
                    ┌────────────── code ──────────────┐
                    │  CI: test  →  CD: build & ship    │
                    └──────────────────────────────────┘
   LLM application ─┤
                    ┌────────────── data ──────────────┐
                    │  CT: crawl → clean → embed → store │
                    └──────────────────────────────────┘
                    ┌────────────── model ─────────────┐
                    │  CT: dataset → train → evaluate →  │
                    │      register → deploy             │
                    └──────────────────────────────────┘
```

**The mental model to keep**: CI/CD is what *any* software team does. CT is the part that exists only because the system learns from data.

---

## 📁 From DevOps to LLMOps

### DevOps

- **Lifecycle**: plan → code → build → test → release → deploy → operate → monitor → repeat.
- **Core concepts**: CI (integrate often, test automatically), CD (deploy reliably), infrastructure as code, monitoring.
- **Key idea**: shorten the feedback loop between a change and knowing whether it was good.

### MLOps

ML systems have three independently-changing **dimensions**: **code**, **data**, and **model**. MLOps extends DevOps to cover all three.

- **Core components**: experiment tracking, model registry, feature store, metadata store, orchestrator.
- **ML vs MLOps engineering**: ML engineers build models; MLOps engineers operationalize the lifecycle (automation, versioning, testing, monitoring, reproducibility).
- **Continuous training (CT)**: automate retraining by trigger (schedule, new data, or a monitoring drop).

The reason MLOps is harder than DevOps is variance: the same code plus different data produces a different model, so "it works on my machine" now has two more axes.

### LLMOps

LLMOps adds concerns specific to LLMs:

- **Prompt monitoring**: log full traces (query, context, prompt, answer, tokens, latency), not just inputs/outputs.
- **Guardrails**: safety and quality filters on inputs/outputs.
- **Human feedback**: collect user ratings and use them to improve the system.
- **Why not train from scratch**: pretraining a frontier LLM costs millions; companies instead **prompt-engineer or fine-tune** open models.
- **Non-determinism**: the same prompt can yield different outputs (sampling temperature), so "the output changed" is not automatically a regression.
- **Evaluation is fuzzy**: there is no exact label; quality is judged by heuristics, LLM-as-judge, or humans (Session 7.3/7.4).

| Layer | Optimizes | Artifact versioned | Failure looks like |
|-------|-----------|--------------------|--------------------|
| DevOps | code | commit | build/test fails |
| MLOps | + data, model | dataset, model | drift, degraded metrics |
| LLMOps | + prompt, feedback | prompt template, traces | hallucination, refusal, toxicity |

---

## 📁 Deploying the Pipelines to the Cloud

The book walks through provisioning the production infrastructure:

- **MongoDB** and **Qdrant**: managed or self-hosted cloud instances (Qdrant Cloud via `USE_QDRANT_CLOUD`).
- **ZenML cloud**: managed orchestrator + artifact store; runs pipelines against the cloud stack.
- **Docker**: the CI/CD image (Session 8.1) is what ZenML runs on SageMaker.
- **Run on AWS**: ZenML executes step containers on the configured stack.
- **Troubleshooting `ResourceLimitExceeded`**: a SageMaker quota limit on the number of instances per type; request a quota increase or reduce instance size (`GPU_INSTANCE_TYPE`, e.g. `ml.g5.2xlarge`).

The same code runs locally and in the cloud; only the **stack** changes (Session 8.3). Switching is a command:

```bash
poetry poe set-local-stack      # zenml stack set default
poetry poe set-aws-stack        # zenml stack set aws-stack
poetry poe set-asynchronous-runs
```

**Troubleshooting table**:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ResourceLimitExceeded` | SageMaker instance quota | request quota increase or smaller instance |
| Steps run locally despite cloud stack | stack not active | `zenml stack set aws-stack`; verify `zenml stack describe` |
| Secret lookup fails on remote run | settings not exported | `poetry poe export-settings-to-zenml` |
| Pipeline hangs | synchronous orchestrator | `set-asynchronous-runs` |
| Image not found on AWS | CD did not push to ECR | check the CD job; verify the image tag |

**Why the stack abstraction matters**: the pipeline DAG is written once against abstract steps. The stack (orchestrator, artifact store, container registry, secrets) is pluggable. That is what lets the identical `feature_engineering` pipeline run on a laptop and on SageMaker without code changes.

---

## 📁 CI/CD/CT: The Three Pipelines

| Pipeline | Runs on | Purpose | Dimension |
|----------|---------|---------|-----------|
| **CI** | pull request | gitleaks, lint, format, tests | code |
| **CD** | merge to main | build Docker image, push to ECR | code |
| **CT** | schedule/event/manual | run data → feature → dataset → training → deploy | data + model |

**The key difference**: CI/CD handles the code dimension that *any* software has. CT leverages that code to automate the **data and model** dimensions that exist only in the AI world.

### CT triggers

- **Manual**: one click/CLI command starts the entire ML system.
- **REST API**: a watcher (e.g. new articles) calls the pipeline over HTTP.
- **Scheduled**: cron, e.g. `Schedule(cron_expression="* * 1 * *")`.

The LLM Twin uses a **manual trigger** because its data comes from a static list of links; the natural next step is a watcher that triggers the REST API on new content.

Choosing a trigger is a design decision, not a preference:

| Trigger | Latency to react | Cost | Best for | Risk |
|---------|------------------|------|----------|------|
| Manual | human-dependent | lowest | static data, demos | stale model if forgotten |
| Schedule (cron) | up to one period | predictable | slowly-changing corpora | wasted runs on no-change data |
| Event / watcher | seconds-minutes | highest | fast-moving data | thundering herd, duplicate runs |

### Triggering downstream pipelines

Rather than compressing everything into one pipeline, keep pipelines isolated and trigger the next one:

```python
from zenml import pipeline, step
from zenml.client import Client
from zenml.config import PipelineRunConfiguration


@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    user = get_or_create_user(user_full_name)
    crawl_links(user=user, links=links)
    trigger_feature_engineering_pipeline(user)


@step
def trigger_feature_engineering_pipeline(user):
    run_config = PipelineRunConfiguration(...)
    Client().trigger_pipeline("feature_engineering", run_configuration=run_config)
```

The book notes it compressed everything into a single master pipeline because the **ZenML cloud free tier limits runs to three pipelines**. If self-hosting or on a paid plan, prefer isolated pipelines triggered in sequence (easier to debug and monitor).

```python
@pipeline
def end_to_end_data(author_links, ...):
    wait_for_ids = []
    for author_data in author_links:
        last_step_invocation_id = digital_data_etl(
            user_full_name=author_data["user_full_name"], links=author_data["links"]
        )
        wait_for_ids.append(last_step_invocation_id)
    author_full_names = [a["user_full_name"] for a in author_links]
    wait_for_ids = feature_engineering(author_full_names=author_full_names, wait_for=wait_for_ids)
    generate_instruct_datasets(...)
    training(...)
    deploy(...)
```

This is the `end_to_end_data` pipeline in `pipelines/end_to_end_data.py` (Session 8.3), extended conceptually to include training and deploy. The real `feature_engineering` pipeline accepts `wait_for` to sequence after the ETL steps:

```python
# pipelines/feature_engineering.py
@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)
    ...
```

**Why keep pipelines isolated**: a single mega-pipeline re-runs the expensive ETL when only training changed, and a failure anywhere loses the whole run's intermediate artifacts. Isolated pipelines let ZenML cache each stage and let you re-run just the broken one. The book compressed them only to dodge a quota.

---

## 📁 Prompt Monitoring in Production

The **business microservice** is the right place to trace, because it coordinates the end-to-end flow (it owns retrieval and prompt assembly).

```python
# llm_engineering/infrastructure/inference_pipeline_api.py
import opik
from opik import opik_context

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
```

**Three things to monitor constantly**:
1. **Model configuration**: LLM and RAG-layer model IDs, temperature.
2. **Token counts**: input prompt tokens and total tokens, because they drive serving cost. A sudden rise signals a bug (e.g. a chunker change inflating context).
3. **Duration of each step**: to find latency bottlenecks.

**Granularity rule**: trace the critical functions (`rag`, `call_llm_service`), then add detail (retriever search, self-query) as needed. Tracing everything adds noise. Each `@opik.track` span is a tree node; the `ContextRetriever.search` method is itself tracked (`@opik.track(name="ContextRetriever.search")`), so the waterfall already shows retrieval separately.

**What to do with the traces**:
- Build a **dashboard** on token cost per day.
- Watch the **latency distribution**, not the average; the tail is what users feel.
- Sample traces for **quality** review and as an eval set.

The repo's `rag()` follows this pattern (Session 6.2), tracking model ids and query/context/answer tokens.

---

## 📁 Guardrails and Human Feedback

Monitoring tells you something is wrong. Guardrails prevent it, and feedback tells you how to fix it.

### Guardrails

LLMOps-specific gates applied at the boundaries:

- **Input guardrails**: reject or sanitize unsafe, oversized, or off-topic queries. This is where the query length limit from Session 9.3 lives.
- **Retrieved-context guardrails**: strip prompt-injection instructions from crawled content; escape `{`/`}`; cap context length.
- **Output guardrails**: filter toxicity, PII leakage, and hallucinated citations before returning the answer.
- **Tool/action guardrails**: if the LLM can act, require confirmation for destructive actions.

```python
def guarded_rag(query: str) -> str:
    if len(query) > settings.MAX_QUERY_LEN:
        raise ValueError("Query too long")
    if is_disallowed(query):
        return "I can't help with that request."
    answer = rag(query)
    if not passes_output_filter(answer):
        return "I couldn't produce a safe answer."
    return answer
```

### Human feedback

- **Collect implicit signals**: thumbs up/down, copy, follow-through.
- **Collect explicit signals**: a `POST /feedback` endpoint writing a score to the Opik trace via `feedback_scores`.
- **Close the loop**: feedback becomes a dataset for evaluation (Session 7.4) and, eventually, preference data for DPO (Session 5.2).

```
User ──► /rag ──► answer ──► user rates ──► /feedback ──► Opik feedback_score
                                                             │
                                             sampled into eval set / preference data
                                                             ▼
                                                    dataset → train → deploy
```

This is the human-in-the-loop that turns a frozen model into a learning system.

---

## 📁 Alerting

ZenML ships an **alerter** component. Wire pipeline callbacks for failure and success:

```python
from zenml import get_pipeline_context, pipeline, step
from zenml.client import Client

alerter = Client().active_stack.alerter


def notify_on_failure() -> None:
    alerter.post(message=build_message(status="failed"))


@step(enable_cache=False)
def notify_on_success() -> None:
    alerter.post(message=build_message(status="succeeded"))


@pipeline(on_failure=notify_on_failure)
def training_pipeline(...):
    ...
    notify_on_success()
```

Common channels: Slack, Discord, email, PagerDuty. Alert thresholds depend on the application; tune to avoid alert fatigue.

**Why `enable_cache=False` on the success step**: a cached success step would be skipped on re-runs and would not notify. Notification is a side effect, and cached steps must be side-effect-free by definition.

**Why alert on success too**: a pipeline that stops *running* produces no failure event. A success ping on a schedule is a heartbeat; its absence is the signal.

| Alert type | Trigger | Channel | Tuning |
|-----------|---------|---------|--------|
| Pipeline failure | `on_failure` | Slack/PagerDuty | fire immediately |
| Pipeline success | heartbeat step | Slack | low noise, scheduled runs only |
| Metric threshold | accuracy < 0.8 | PagerDuty | avoid fatigue; windowed |
| Drift | p-value < 0.05 | Slack | pair with data validation |
| Cost | token spend/day > budget | Email | tune to business budget |

**Alert fatigue** is the failure mode: too many low-value pages train the team to ignore them. Start with failure + heartbeat, add metric alerts only when you trust them.

---

## 🛠️ Hands-On: Reason About a CT System

### Step 1: Identify the trigger

For the LLM Twin, the trigger is **manual** because links are static. Design a watcher variant:

```
Watcher (daily) ──new links?──yes──► REST API trigger ──► digital_data_etl ──► feature ──► datasets ──► train ──► deploy
```

### Step 2: Add an alerter

1. Add an alerter to your ZenML stack (`zenml alerter register ...`).
2. Wire `on_failure` on the training pipeline.
3. Force a failure and confirm the notification.

### Step 3: Add production feedback

In `rag()`, add a `feedback_scores` entry from a user rating endpoint (e.g. `POST /feedback`), and confirm it appears in the Opik trace.

```python
opik_context.update_current_trace(
    tags=["rag"],
    metadata={...},
    feedback_scores=[
        {"name": "user_feedback", "value": 1.0, "reason": "The response was valuable and correct."},
    ],
)
```

### Step 4: Trace the blast radius of a corpus change

Ask: if tomorrow's crawl changes a chunking parameter, which of the three dimensions change? Answer: data (new chunks) and model (a retraining that sees different context). Code does not change. This is why CT exists separately from CI/CD.

---

## 📝 Exercise 1: Add a CT Schedule to ZenML

### Task

Schedule the feature pipeline to run hourly.

1. Create a ZenML schedule: `Schedule(cron_expression="0 * * * *")`.
2. Attach it to `feature_engineering` with `pipeline.with_options(schedule=...)`.
3. Explain why a schedule is acceptable for the LLM Twin (small data, minutes of latency OK) but insufficient for a social-media recommender.

**Goal**: Practice implementing a CT trigger and reason about schedule vs event-driven design.

---

## 📝 Exercise 2: Design a Guardrail and a Feedback Loop

### Task

Extend the inference API with a minimal safety and learning layer.

1. Add `POST /feedback` accepting `{trace_id, score, reason}` and forward it to Opik `feedback_scores`.
2. Add an input length guard (`MAX_QUERY_LEN`) that returns 422 on violation.
3. Add an output filter stub (`passes_output_filter`) and explain what it would check.
4. Describe how sampled feedback becomes a preference dataset for DPO, referencing the session that covers preference data.

**Goal**: Connect the production loop (monitor → guard → collect feedback → retrain) end to end. Cite [`session_3.2_preference_dataset.md`](session_3.2_preference_dataset.md) for where the feedback would land.

---

## 📁 LLMOps vs MLOps: A Detailed Comparison

The book treats LLMOps as MLOps plus LLM-specific concerns. Make that concrete.

| Concern | Classic MLOps | LLMOps |
|---------|--------------|--------|
| Primary artifact | trained model | prompt + model + retrieved context |
| Output | deterministic-ish | probabilistic, sampling-dependent |
| Evaluation | exact metrics (F1, RMSE) | fuzzy: heuristics, LLM-judge, humans |
| Cost driver | compute hours | tokens (input + output) |
| Failure mode | drift, degraded accuracy | hallucination, refusal, toxicity, prompt injection |
| Versioned unit | model weights + data | prompt template + model + data + guardrails |
| Feedback | labels | user ratings, implicit signals |
| New pipeline stage | — | retrieval and reranking quality |

**Implication for CT**: in classic MLOps, retraining is the whole story. In LLMOps, you may improve the system by editing a prompt, re-ranking retrieved documents, or adding a guardrail — no retraining at all. CT must therefore version and promote prompts, not just weights.

```
Classic MLOps CT:   data ──► train ──► evaluate ──► register ──► deploy
LLMOps CT:          data ──► build RAG index ──► (prompt change) ──► evaluate ──► deploy
                              │                                         ▲
                              └────────────► train / fine-tune ────────┘
```

---

## 📁 The Model Registry and Promotion Gates

A registry is where "best run" becomes "production version". Without gates, promotion is opinion; with gates, it is policy.

```
   experiment runs
        │  (metric improves?)
        ▼
   candidate ──► GATE 1: quality    eval score >= threshold
        │
        ▼
   candidate ──► GATE 2: safety     no toxic/unsafe outputs on a red-team set
        │
        ▼
   candidate ──► GATE 3: cost       tokens/request within budget
        │
        ▼
   candidate ──► GATE 4: latency    p95 within SLO
        │
        ▼
   registered version ──► deploy ──► canary ──► full rollout
```

Each gate is a pipeline step that can fail the promotion. In the repo, the Hugging Face Hub hosts `{workspace}/TwinLlama-3.1-8B` and `{workspace}/TwinLlama-3.1-8B-DPO`; the `model/evaluation/evaluate.py` job (accuracy + style via LLM-judge) is the concrete realization of GATE 1.

**Promotion discipline**: define thresholds *before* the run. A gate discovered after seeing the numbers is not a gate.

---

## 📁 Cloud Deployment Runbook

A repeatable sequence to stand up the production stack.

```
1. PROVISION
   - MongoDB (managed or private host) with strong, unique credentials.
   - Qdrant Cloud (USE_QDRANT_CLOUD=true, QDRANT_CLOUD_URL, QDRANT_APIKEY).
   - ECR repository for the image; SageMaker execution role (least privilege).

2. CONFIGURE SECRETS
   - Fill .env locally → poetry poe export-settings-to-zenml.
   - Verify "Loading settings from the ZenML secret store" in a remote run.

3. BUILD & SHIP
   - CD job builds linux/amd64 image and pushes to ECR (poetry poe build-docker-image).

4. REGISTER THE STACK
   - zenml stack register aws-stack ... ; poetry poe set-aws-stack.
   - poetry poe set-asynchronous-runs to avoid blocking orchestrator runs.

5. RUN CT
   - Trigger end_to_end_data (manual, schedule, or REST watcher).
   - Monitor the ZenML dashboard; confirm artifacts land in the store.

6. DEPLOY INFERENCE
   - poetry poe deploy-inference-endpoint; test with test-sagemaker-endpoint.
   - Wire the business API (tools.ml_service) to the endpoint.
```

**Ordering rationale**: secrets before stack, stack before CT, because a failed secret lookup silently falls back to dev defaults (Session 9.3). Deploying before the stack is correct wastes a build.

---

## 📊 Cost Model

LLMOps cost is dominated by tokens and endpoint hours. A simple model:

```
monthly_cost ≈ (requests × avg_input_tokens × price_in)
             + (requests × avg_output_tokens × price_out)      # per-LLM-call
             + (endpoint_hours × instance_hourly_price)          # always-on serving
             + (training_runs × run_hours × instance_hourly_price)
```

The RAG endpoint calls the LLM twice during retrieval (self-query + expansion) plus once for generation. So token cost is ~3× a naive single-call estimate. This is exactly why prompt monitoring logs `query_tokens`, `context_tokens`, and `answer_tokens`: those are the cost inputs.

| Lever | Effect | Tradeoff |
|-------|--------|----------|
| Cache repeated queries | removes all LLM calls for hits | staleness, tenancy correctness |
| Smaller judge model | cheaper evaluation | noisier scores |
| Lower `MAX_NEW_TOKENS_INFERENCE` | cheaper generation | shorter answers |
| Fewer query expansions | fewer tokens + searches | possible recall drop |
| Scale-to-zero endpoints | no idle cost | cold-start latency |

---

## 📁 Monitoring Worked Example: Detecting a Chunker Regression

A concrete story of monitoring earning its keep.

1. A change to the chunker halves average chunk size.
2. **Token counts** logged by `rag()` (`context_tokens`) drop sharply on the next day — the first signal.
3. **Retrieval quality** on a held-out eval set (Session 7.4) drops, confirming the change hurt.
4. **Alert** fires on the windowed context-token metric crossing a band.
5. The on-call inspects the Opik trace waterfall, sees fewer/shorter chunks, and bisects to the chunker commit.
6. Roll back or retrain; the eval set guards the fix.

Without step 2's metric, the quality drop would be discovered by users weeks later. The lesson: instrument the *causal* quantities (chunk size, token counts, retrieval counts), not only the outcome (user satisfaction), so you can diagnose before users notice.

---

## 🐛 Common Pitfalls

- **Compressing all pipelines into one**: the book did this only to dodge a free-tier limit; it hurts debuggability. Keep pipelines isolated when you can.
- **Monitoring everything**: too many spans make traces unreadable. Trace critical functions first.
- **No alert thresholds tuning**: false positives make alerts untrustworthy; teams then ignore real ones.
- **Alerting only on failure**: notify on success too, so a silent pipeline that stops running is detectable.
- **Missing guardrails**: production LLM output needs input/output safety filtering.
- **Forgetting `enable_cache=False` on a notification step**: the alert is skipped on re-runs.
- **Feedback collected but never used**: a rating with no path to a dataset is just telemetry.
- **Treating prompt changes as non-events**: the prompt is part of the model dimension; version it and re-evaluate after every change.
- **Scheduling on no-change data**: an hourly crawl of a static source wastes runs; add a change check (watcher) before triggering.

---

## 🎓 Knowledge Check

1. **What are the three dimensions of an ML application?**
   - Answer: code, data, and model.

2. **What does CT automate that CI/CD does not?**
   - Answer: the data and model dimensions (retraining and pipeline execution).

3. **Name the three CT trigger types.**
   - Answer: manual, REST API, and scheduled (cron).

4. **How do you trigger a downstream pipeline in ZenML?**
   - Answer: `Client().trigger_pipeline("name", run_configuration=...)` from within a step.

5. **What three things should you always monitor for an LLM service?**
   - Answer: model configuration, token counts, and per-step duration.

6. **How does ZenML alerting work?**
   - Answer: an alerter stack component posts messages via `on_failure` callbacks and a `notify_on_success` step.

7. **Why alert on success as well as failure?**
   - Answer: A pipeline that silently stops running emits no failure; a scheduled success heartbeat makes its absence visible.

8. **Why must a notification step disable caching?**
   - Answer: Notification is a side effect, and ZenML skips cached steps, so the alert would not fire on a re-run.

9. **Which single artifact makes the same pipeline run locally and in the cloud?**
   - Answer: The ZenML stack (orchestrator, artifact store, registry, secrets); the DAG code is identical.

10. **What does `ResourceLimitExceeded` mean and how do you fix it?**
    - Answer: A SageMaker instance quota is exceeded; request a quota increase or use a smaller instance type.

11. **Why is prompt monitoring owned by the microservice and not the training pipeline?**
    - Answer: The service coordinates retrieval and prompt assembly, so it observes the real query, context, and answer.

12. **What is the difference between a guardrail and monitoring?**
    - Answer: Monitoring observes and alerts; guardrails actively block or sanitize at the boundary.

13. **How does human feedback become training data?**
    - Answer: Ratings and traces are sampled into an evaluation set and, with preference pairs, into a preference dataset for DPO (Session 3.2).

14. **Why is schedule-based CT wrong for a fast-moving recommender?**
    - Answer: Between runs the data drifts enough that the model is stale; event-driven retraining reacts in near real time.

15. **Why does the book use a single master pipeline, and when should you not?**
    - Answer: It dodged the ZenML cloud free tier's three-pipeline limit; with self-hosting or a paid plan, isolated pipelines are easier to debug and cache.

---

## 📖 Glossary

- **CI (Continuous Integration)**: Automatically test and integrate code changes.
- **CD (Continuous Delivery/Deployment)**: Reliably build and ship artifacts.
- **CT (Continuous Training)**: Automatically retrain models on data or schedule triggers.
- **FTI**: Feature, Training, Inference — the three-pipeline architecture the book builds on.
- **Orchestrator**: The system that schedules and runs pipeline steps (ZenML, here).
- **Stack**: The pluggable combination of orchestrator, artifact store, registry, and secrets in ZenML.
- **Alerter**: A ZenML stack component that posts notifications.
- **Guardrail**: An input/output filter that blocks or sanitizes unsafe content.
- **Human-in-the-loop**: A design where human feedback influences system behavior.
- **Heartbeat**: A periodic success signal whose absence indicates a problem.
- **Trace**: A recorded execution tree of a call (Opik), with spans, metadata, and feedback.
- **Feedback score**: A numeric rating attached to a trace.
- **Resource quota**: A cloud-imposed cap on resource usage (e.g. SageMaker instances).

---

## 🔗 Course Complete

You have now worked through all 11 book chapters plus the Appendix:

| Session | Book chapter |
|---------|--------------|
| `session_1.1`-`session_1.3` | Ch 1-3 |
| `session_2.1`-`session_2.3`, `session_4.3` | Ch 3-4 |
| `session_3.1`, `session_5.4` | Ch 5 |
| `session_3.2` | Ch 6 |
| `session_7.3`, `session_7.4` | Ch 7 |
| `session_8.4` | Ch 8 |
| `session_4.1`, `session_6.2` | Ch 9 |
| `session_5.3`, `session_6.1`, `session_10.1` | Ch 10 |
| `session_7.1`, `session_7.2`, `session_8.1`-`session_8.3`, `session_9.1`, `session_11.1` | Ch 11 |
| `appendix_mlops_principles` | Appendix |

---

## 📚 Additional Resources

- [ZenML Alerter](https://docs.zenml.io/stacks/stack-components/alerters)
- [ZenML: Trigger a pipeline from REST API](https://docs.zenml.io/v/docs/how-to/trigger-pipelines/trigger-a-pipeline-from-rest-api)
- [Chip Huyen, Building a Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html)
- [Google Cloud: What is LLMOps](https://cloud.google.com/discover/what-is-llmops)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Opik documentation](https://www.comet.com/docs/opik/)

---

## 🔎 References

- **Book**: *LLM Engineer's Handbook* — Chapter 11 (MLOps and LLMOps, pp. 430-488).
- **Repo**: `pipelines/feature_engineering.py`, `pipelines/end_to_end_data.py`, `llm_engineering/infrastructure/inference_pipeline_api.py`, `llm_engineering/application/rag/retriever.py`, `pyproject.toml` (stack/poe tasks).
- **Related sessions**: [`session_7.2_opik_monitoring.md`](session_7.2_opik_monitoring.md), [`session_8.2_cicd.md`](session_8.2_cicd.md), [`session_8.3_zenml.md`](session_8.3_zenml.md), [`session_10.1_deployment_topologies.md`](session_10.1_deployment_topologies.md), [`appendix_mlops_principles.md`](appendix_mlops_principles.md).
- **Curriculum**: [`../CURRICULUM.md`](../CURRICULUM.md) and [`../BOOK-MAP.md`](../BOOK-MAP.md).

---

**Estimated Time**: 5-6 hours

**Prerequisites**: [Session 7.1](session_7.1_comet_ml.md), [Session 7.2](session_7.2_opik_monitoring.md), [Session 8.2](session_8.2_cicd.md), [Session 8.3](session_8.3_zenml.md)

**Outcome**: You can design a CI/CD/CT system, add prompt monitoring, guardrails, feedback, and alerting, and explain LLMOps as an extension of MLOps.
