# Roadmap: LLM Engineer's Handbook → a production LLM Twin

This is the single source of truth for planned work. It answers, in order: what we are building, in what sequence, how each step is proven, and where every decision lives. The other docs are the references; this is the plan.

- `BOOK-MAP.md` — chapter → session → code, plus per-chapter Definition of Done.
- `CURRICULUM.md` — the chapter-by-chapter path.
- `READING-PLAN.md` — the prioritized study strategy and read-depth rules.
- `DECISIONS.md` — the Architecture Decision Record (ADR) index.
- `GETTING_STARTED.md` — setup and run commands.
- `CODEBASE-INTELLIGENCE.md` — graph tooling and architecture findings.

## Mission

Master engineering an LLM Twin from concept to production. An LLM Twin is a system, not a model: a feature pipeline, a training pipeline, and an inference pipeline that share one feature layer. Mastery means rebuilding that system from a clean machine, justifying every decision, and producing a runnable artifact at every milestone.

## What we are building (the FTI architecture)

The repository is the reference implementation. The roadmap gates each milestone on your own decisions and evidence, not on copying it.

- **Feature pipeline**: `pipelines/digital_data_etl.py`, `pipelines/feature_engineering.py`, `steps/etl/`, `steps/feature_engineering/`, and `llm_engineering/application/crawlers/`, `preprocessing/`, `networks/`.
- **Training pipeline**: `pipelines/generate_datasets.py`, `pipelines/training.py`, `llm_engineering/application/dataset/generation.py`, `llm_engineering/model/finetuning/`.
- **Inference pipeline**: `llm_engineering/application/rag/`, `llm_engineering/model/inference/`, `llm_engineering/infrastructure/inference_pipeline_api.py`, and the RAG API at `tools/rag.py` (`POST /rag`).
- **Shared domain layer**: `llm_engineering/domain/` (`NoSQLBaseDocument`, `VectorBaseDocument`, `DataCategory`, `Query`, documents and chunks).
- **Stores**: MongoDB (`infrastructure/db/mongo`) and Qdrant (`infrastructure/db/qdrant`).

## Principles

- Read the table of contents as a menu of decisions, not a page order (see `READING-PLAN.md`).
- Build the evaluation harness before any fine-tuning; a frozen eval set is what makes SFT and DPO measurable.
- A milestone is done only when its evidence exists. A label without evidence is a claim, not a fact.
- Record every non-trivial choice as an ADR in `DECISIONS.md` before trusting the book's default.
- Run in fp16 on the RTX 5000 (Turing sm_75): no bfloat16, no FlashAttention-2. Quantize when a model otherwise does not fit.

## Current baseline

Documentation is complete; implementation is not started.

- 31 session files cover all 11 chapters and the Appendix.
- `BOOK-MAP.md`, `CURRICULUM.md`, `READING-PLAN.md`, `DECISIONS.md` are in place.
- Every chapter is at status `Covered` and every ADR at `Proposed`.
- The code intelligence graph is indexed (1234 nodes / 4184 edges) and code-only (no LLM API key configured).

## Milestones

Six milestones, in dependency order. Each lists the chapters to read, the sessions to study, what to build, the decisions to record, and the evidence that closes it. Milestone status is independent of the chapter `Covered` status in `BOOK-MAP.md`.

### Milestone 1 — Foundation: architecture and data

- **Goal**: internalize the FTI architecture and stand up a working data pipeline with a clear data contract.
- **Chapters**: 1 (FTI), 3 (data engineering), and 2 only for setup.
- **Sessions**: `session_1.1_project_overview`, `session_1.2_domain_layer`, `session_1.3_infrastructure_layer`, `session_2.1_web_crawling`.
- **Build**: an FTI diagram; a crawler that pulls your own writing; cleaning and warehouse ingestion; a data contract as Pydantic models and ODM documents.
- **Decisions (ADR)**: ADR-001 FTI boundaries, ADR-002 local-first tooling, ADR-003 data sources and warehouse.
- **Evidence**: document counts and schema tests; you can explain the data flow from raw source to inference without the book.
- **Status**: Not started.

### Milestone 2 — RAG feature pipeline

- **Goal**: turn cleaned text into searchable features.
- **Chapters**: 4 (RAG feature pipeline).
- **Sessions**: `session_2.2_text_preprocessing`, `session_2.3_feature_engineering`, `session_4.2_embedding_models`, `session_4.3_streaming_cdc`.
- **Build**: chunking, embedding, and indexing into Qdrant; the dispatcher/handler pattern; the OVM (`VectorBaseDocument`).
- **Decisions (ADR)**: ADR-004 chunking, ADR-005 embedding model.
- **Evidence**: indexed chunk count and a retrieval smoke test that returns relevant chunks with scores.
- **Status**: Not started.

### Milestone 3 — RAG inference

- **Goal**: close the retrieval loop into a cited answer.
- **Chapters**: 9 (RAG inference pipeline).
- **Sessions**: `session_4.1_advanced_rag`, `session_6.2_rag_inference_flow`, `session_4.2_embedding_models`.
- **Build**: query expansion, self-query metadata filtering, reranking, and generation, exposed as an API with citations and retrieval traces.
- **Decisions (ADR)**: ADR-010 retriever and reranker.
- **Evidence**: sample queries with citations and traces.
- **Status**: Not started.

### Milestone 4 — Evaluation harness

- **Goal**: build the frozen eval set and baseline before any fine-tuning.
- **Chapters**: 7 (evaluating LLMs).
- **Sessions**: `session_7.3_model_evaluation`, `session_7.4_rag_evaluation`.
- **Build**: a model eval and a RAG eval suite (Ragas or ARES), versioned and repeatable; a frozen eval set.
- **Decisions (ADR)**: ADR-008 eval set, metrics, judge.
- **Evidence**: versioned baseline metrics (Comet ML or Opik identifiers).
- **Status**: Not started.

### Milestone 5 — Fine-tuning and alignment

- **Goal**: make the model measurably better than base on your own data.
- **Chapters**: 5 (SFT), 6 (DPO).
- **Sessions**: `session_3.1_instruction_dataset`, `session_5.4_data_curation`, `session_5.1_sft`, `session_3.2_preference_dataset`, `session_5.2_dpo`.
- **Build**: an instruction dataset generated from your data and a QLoRA run; then a preference dataset and a DPO run.
- **Decisions (ADR)**: ADR-006 base model and QLoRA configuration, ADR-007 preference data generation.
- **Evidence**: fine-tuned versus base, and DPO versus SFT, on the same frozen eval set from Milestone 4.
- **Status**: Not started.

### Milestone 6 — Optimize, deploy, operate

- **Goal**: take the system to production.
- **Chapters**: 8 (inference optimization), 10 (deployment), 11 and Appendix (MLOps).
- **Sessions**: `session_8.4_inference_optimization`, `session_5.3_sagemaker_deployment`, `session_6.1_fastapi_api`, `session_10.1_deployment_topologies`, `session_7.1_comet_ml`, `session_7.2_opik_monitoring`, `session_8.1_docker`, `session_8.2_cicd`, `session_8.3_zenml`, `session_9.1_data_warehouse`, `session_11.1_ct_pipeline_alerting`, `appendix_mlops_principles`.
- **Build**: a before/after benchmark (latency, throughput, VRAM, quality); a containerized FastAPI inference service; CI/CD; monitoring and alerting on drift.
- **Decisions (ADR)**: ADR-009 quantization and serving stack, ADR-011 deployment topology, ADR-012 CI/CD and monitoring, ADR-013 testing and seeding.
- **Evidence**: a benchmark table, a health check and smoke test, and a workflow run that fails on regression.
- **Status**: Not started.

## Dependency order

```
M1 (architecture + data)
 └─ M2 (RAG features)
     └─ M3 (RAG inference)
         └─ M4 (evaluation harness)
             └─ M5 (SFT, then DPO)
                 └─ M6 (optimize, deploy, operate)
```

The evaluation harness (M4) is scaffolded as early as M2 and completed after M3, but it must gate M5. Nothing is tuned or aligned until a frozen baseline exists.

## Cross-cutting tracks

These run continuously rather than inside one milestone.

- **Evaluation discipline**: start a retrieval eval set in M2; freeze it in M4; run it before and after every change thereafter.
- **Code intelligence**: re-index the graph after significant code changes (`codebase-memory_index_repository`, `graphify extract ... --code-only`).
- **Decision logging**: every milestone closes only after its ADRs are written and accepted in `DECISIONS.md`.

## Decisions backlog

All ADRs are `Proposed`. Each is opened in the milestone that first forces the decision.

| ADR | Decision | Opens in |
|-----|----------|----------|
| ADR-001 | FTI architecture boundaries | M1 |
| ADR-002 | Local-first tooling and environment | M1 |
| ADR-003 | Data sources and warehouse choice | M1 |
| ADR-004 | Chunking strategy | M2 |
| ADR-005 | Embedding model selection | M2 |
| ADR-010 | Retriever and reranker selection | M3 |
| ADR-008 | Evaluation set, metrics, and judge | M4 |
| ADR-006 | Base model and QLoRA configuration | M5 |
| ADR-007 | Preference data generation | M5 |
| ADR-009 | Inference optimization and serving stack | M6 |
| ADR-011 | Deployment topology and autoscaling | M6 |
| ADR-012 | CI/CD and monitoring stack | M6 |
| ADR-013 | Testing and seeding strategy | M6 |

## Cloud-gated backlog

These steps need cloud accounts or external API keys and are deferred until needed. Use the local substitute to keep momentum.

| Item | Milestone | Blocker | Local substitute |
|------|-----------|---------|------------------|
| SageMaker training and deployment | M5, M6 | AWS account and credentials | QLoRA locally, Docker Compose |
| TGI Deep Learning Container | M6 | AWS and SageMaker | vLLM on the RTX 5000 |
| Cloud CT pipeline | M6 | AWS | GitHub Actions and local Opik |
| External judge APIs | M4 | Gemini, OpenAI, or Anthropic keys | a local judge model |
| LLM-based dataset generation | M5 | API key for a frontier model | a local model, or a small curated seed set |

## Risks and constraints

- **Precision**: Turing sm_75 has no bfloat16 and no FlashAttention-2; everything trains and serves in fp16. A 7B model fits locally only when quantized to 4-bit.
- **Memory budget**: weights plus KV cache plus activations plus quantization; never infer fit from file size alone.
- **Keys**: no LLM API keys are set, which affects dataset generation, LLM-as-judge, and semantic code indexing.
- **Deferred depth**: AWS/SageMaker/ZenML-cloud and detailed CI/CD are concept-only until the core system works locally.

## Status convention

Milestone status: `Not started`, `In progress`, `Done`. Chapter status in `BOOK-MAP.md` uses the ladder `Covered → Implemented → Tested → Deployed`, with `Blocked` for cloud-gated steps. A milestone is `Done` only when its evidence exists and its ADRs are accepted.

## Definition of done

The roadmap is finished when every milestone is `Done` and four artifacts exist: a RAG API that returns cited answers, a frozen eval suite run before and after every change, an accepted ADR for every decision above, and a fine-tuned model that measurably beats its own baseline. At that point the system is reproducible from a clean machine.

## Immediate next step

Start Milestone 1: draw the FTI diagram on paper, write ADR-001 in `DECISIONS.md`, then run the setup verification in `GETTING_STARTED.md`. Do not move to Milestone 2 until the data flow can be explained from memory.
