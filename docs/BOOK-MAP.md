# Book Map: LLM Engineer's Handbook → Sessions → Code

This document maps every chapter of **LLM Engineer's Handbook** (Packt, Iusztin and Labonne, 2024; 523 pages, 11 chapters + Appendix) to its session walkthroughs and the repository code it documents.

It has two layers. The first is navigation: find the chapter you are studying and jump to its sessions and code. The second is an execution contract: for each chapter, the deliverable you must produce, the gate that proves it works, the evidence to keep, the decision to record, and the local substitute when the book step is cloud-gated.

Use it to navigate by the book you have open, rather than by the derived session numbering, and to track mastery from `Covered` to `Deployed`.

**Study strategy**: see [`READING-PLAN.md`](./READING-PLAN.md) for the prioritized menu of decisions and the read-depth rules; this map is the reference underneath it.

## Chapter-to-session map

| Book chapter | Pages | Sessions | Primary repository code | Status |
|--------------|-------|----------|-------------------------|--------|
| **1. Understanding the LLM Twin Concept and Its Architecture** | 30–52 | `session_1.1_project_overview.md` | `llm_engineering/`, FTI architecture, `pipelines/` | Covered |
| **2. Tooling and Installation** | 54–82 | `session_1.3_infrastructure_layer.md`, `GETTING_STARTED.md` | `pyproject.toml`, `Dockerfile`, `docker-compose.yml` | Covered |
| **3. Data Engineering** | 84–125 | `session_2.1_web_crawling.md`, `session_1.2_domain_layer.md`, `session_1.3_infrastructure_layer.md` | `llm_engineering/application/crawlers/`, `llm_engineering/domain/documents.py`, `tools/data_warehouse.py` | Covered |
| **4. RAG Feature Pipeline** | 128–203 | `session_2.2_text_preprocessing.md`, `session_2.3_feature_engineering.md`, `session_4.2_embedding_models.md`, `session_4.3_streaming_cdc.md` | `llm_engineering/application/preprocessing/`, `llm_engineering/application/networks/`, `pipelines/feature_engineering.py`, `steps/feature_engineering/` | Covered |
| **5. Supervised Fine-Tuning** | 206–255 | `session_3.1_instruction_dataset.md`, `session_5.4_data_curation.md`, `session_5.1_sft.md` | `llm_engineering/application/dataset/generation.py`, `llm_engineering/model/finetuning/finetune.py` | Covered |
| **6. Fine-Tuning with Preference Alignment** | 258–287 | `session_3.2_preference_dataset.md`, `session_5.2_dpo.md` | `PreferenceDatasetGenerator`, `DPOTrainer` in `finetune.py` | Covered |
| **7. Evaluating LLMs** | 290–316 | `session_7.3_model_evaluation.md`, `session_7.4_rag_evaluation.md` | `llm_engineering/model/evaluation/evaluate.py` | Covered |
| **8. Inference Optimization** | 318–343 | `session_8.4_inference_optimization.md` | vLLM (evaluation), TGI DLC + bitsandbytes (deployment) | Covered |
| **9. RAG Inference Pipeline** | 346–380 | `session_4.1_advanced_rag.md`, `session_6.2_rag_inference_flow.md`, `session_4.2_embedding_models.md` | `llm_engineering/application/rag/` | Covered |
| **10. Inference Pipeline Deployment** | 384–428 | `session_5.3_sagemaker_deployment.md`, `session_6.1_fastapi_api.md`, `session_10.1_deployment_topologies.md` | `llm_engineering/infrastructure/aws/deploy/`, `llm_engineering/model/inference/`, `llm_engineering/infrastructure/inference_pipeline_api.py` | Covered |
| **11. MLOps and LLMOps** | 430–488 | `session_7.1_comet_ml.md`, `session_7.2_opik_monitoring.md`, `session_8.1_docker.md`, `session_8.2_cicd.md`, `session_8.3_zenml.md`, `session_9.1_data_warehouse.md`, `session_11.1_ct_pipeline_alerting.md` | `.github/workflows/`, `pipelines/`, `llm_engineering/infrastructure/opik_utils.py` | Covered |
| **Appendix: MLOps Principles** | 490–503 | `appendix_mlops_principles.md` | `tests/`, CI, ZenML artifacts, seeds | Covered |

## Status key

- **Covered**: a session walkthrough exists; the topic is documented.
- **Implemented**: you ran the code or built a local version.
- **Tested**: the chapter gate passed, with recorded evidence (command, output, date).
- **Deployed**: running with monitoring and failure handling in place.
- **Blocked**: cannot progress on this workstation; see Cloud-gated backlog for the reason and the local substitute.

A chapter is only `Tested` when its evidence pointer is filled in. A label without evidence is a claim, not a fact.

## Definition of Done

Each chapter is signed off only when its deliverable exists, its gate passes, and its evidence is recorded. `Gate` is `automated` (a command), `review` (an explanation a reviewer accepts), or `runtime` (an observable behavior). Decision logs live in `docs/DECISIONS.md`.

### Chapter 1 — Understanding the LLM Twin Concept and Its Architecture
- **Deliverable**: an architecture diagram and ADR-001 describing FTI boundaries and the data flow from raw source to inference.
- **Gate**: review — explain the full data flow from memory, without the book.
- **Evidence**: `docs/DECISIONS.md#adr-001` plus the diagram.
- **Decision log**: ADR-001 (FTI boundaries), ADR-002 (local-first tooling).
- **Local substitute**: none required; fully local.
- **Prerequisites**: none.

### Chapter 2 — Tooling and Installation
- **Deliverable**: a working local clone that installs and starts (Poetry plus Docker Compose, MongoDB and Qdrant containers).
- **Gate**: automated — `poetry install` and `docker compose config` succeed, and the containers start.
- **Evidence**: install log and `docker compose ps` output.
- **Decision log**: ADR-002 (local-first tooling).
- **Local substitute**: WSL2 Ubuntu 24.04 with Docker Desktop GPU.
- **Prerequisites**: Chapter 1.

### Chapter 3 — Data Engineering
- **Deliverable**: crawler, cleaning, and warehouse ingestion that runs end to end.
- **Gate**: automated — the ingest pipeline produces clean documents and is rerunnable.
- **Evidence**: document count and a warehouse snapshot.
- **Decision log**: ADR-003 (data sources and warehouse).
- **Local substitute**: MongoDB in Docker and local crawl targets.
- **Prerequisites**: Chapter 2.

### Chapter 4 — RAG Feature Pipeline
- **Deliverable**: chunking, embedding, and indexing pipeline writing into Qdrant.
- **Gate**: automated — a retrieval smoke test returns relevant chunks with scores.
- **Evidence**: Qdrant collection stats and sample retrievals.
- **Decision log**: ADR-004 (chunking), ADR-005 (embedding model).
- **Local substitute**: the Athar corpus with Arabic-aware normalization and chunking.
- **Prerequisites**: Chapter 3.

### Chapter 5 — Supervised Fine-Tuning
- **Deliverable**: an instruction dataset and a QLoRA run producing an adapter.
- **Gate**: automated — fine-tuned versus baseline on the frozen eval set from Chapter 7.
- **Evidence**: eval metrics and the adapter checkpoint path.
- **Decision log**: ADR-006 (base model and QLoRA configuration).
- **Local substitute**: Baligh instruction data, QLoRA in fp16 on the RTX 5000 (Turing has no bfloat16).
- **Prerequisites**: Chapter 3 and Chapter 7.

### Chapter 6 — Fine-Tuning with Preference Alignment
- **Deliverable**: a preference dataset and a DPO run.
- **Gate**: automated — SFT versus DPO on the same frozen eval prompts.
- **Evidence**: win-rate or preference metrics.
- **Decision log**: ADR-007 (preference data generation).
- **Local substitute**: Baligh preference pairs.
- **Prerequisites**: Chapter 5 and Chapter 7.

### Chapter 7 — Evaluating LLMs
- **Deliverable**: a model eval and a RAG eval suite (Ragas or ARES), versioned.
- **Gate**: automated — a repeatable run that emits versioned metrics.
- **Evidence**: Comet ML or Opik experiment identifiers.
- **Decision log**: ADR-008 (eval set, metrics, judge).
- **Local substitute**: a local judge model instead of an external judge API, and Athar queries for RAG eval.
- **Prerequisites**: Chapter 4 and Chapter 9 for RAG eval; none for model eval.

### Chapter 8 — Inference Optimization
- **Deliverable**: a before and after benchmark of quantization, batching, and the serving stack.
- **Gate**: automated — a benchmark table with latency, throughput, VRAM, and quality.
- **Evidence**: benchmark output and `nvidia-smi` readings.
- **Decision log**: ADR-009 (quantization and serving stack).
- **Local substitute**: vLLM locally in fp16 on 16 GB (no bfloat16, no FlashAttention-2); the TGI deep learning container is cloud-gated.
- **Prerequisites**: Chapter 5.

### Chapter 9 — RAG Inference Pipeline
- **Deliverable**: retrieval, reranking, and generation exposed as an API with citations.
- **Gate**: runtime — an answer returns with citations and a retrieval trace.
- **Evidence**: sample responses and their traces.
- **Decision log**: ADR-010 (retriever and reranker).
- **Local substitute**: Athar.
- **Prerequisites**: Chapter 4 and Chapter 7.

### Chapter 10 — Inference Pipeline Deployment
- **Deliverable**: a containerized inference API (FastAPI plus Docker).
- **Gate**: runtime — a health check and a smoke test against the deployed endpoint.
- **Evidence**: `curl` output and container logs.
- **Decision log**: ADR-011 (deployment topology and autoscaling).
- **Local substitute**: Docker Compose locally; SageMaker deployment is cloud-gated.
- **Prerequisites**: Chapter 8 and Chapter 9.

### Chapter 11 — MLOps and LLMOps
- **Deliverable**: CI/CD, monitoring, and alerting.
- **Gate**: runtime — a pull request fails on a test or regression, and an alert fires on degradation.
- **Evidence**: a workflow run and an Opik alert.
- **Decision log**: ADR-012 (CI/CD and monitoring stack).
- **Local substitute**: GitHub Actions and local Opik; the cloud CT pipeline is cloud-gated.
- **Prerequisites**: Chapter 8 and Chapter 10.

### Appendix — MLOps Principles
- **Deliverable**: tests, seeds, and reproducibility notes.
- **Gate**: automated — `pytest` passes and the pipeline reruns deterministically.
- **Evidence**: test run output and ZenML artifacts.
- **Decision log**: ADR-013 (testing and seeding strategy).
- **Local substitute**: fully local.
- **Prerequisites**: Chapters 5 to 11.

## Dependency order (not book order)

The book's chapter order is a reading order, not a build order. Evaluation must exist before fine-tuning, or the tuning has nothing to beat.

`Ch 1–3` → `Ch 4` (start the Ragas scaffold) → `Ch 9` RAG inference → `Ch 7` evaluation → `Ch 5` SFT → `Ch 6` DPO → `Ch 8` benchmark → `Ch 10` deployment → `Ch 11` LLMOps → `Appendix` MLOps principles.

## Cloud-gated backlog

These steps need cloud accounts or external API keys. They are `Blocked` on this workstation until a key or account is provided; use the local substitute to keep progressing.

| Item | Chapters | Blocker | Local substitute |
|------|----------|---------|------------------|
| SageMaker training and deployment | 5, 10 | AWS account and credentials | QLoRA locally, Docker Compose |
| TGI Deep Learning Container | 8 | AWS and SageMaker | vLLM on the RTX 5000 |
| Cloud CT pipeline | 11 | AWS | GitHub Actions and local Opik |
| External judge APIs | 7 | Gemini, OpenAI, or Anthropic keys | a local judge model |

## Arabic local substitutes

Two flagship projects turn the book into working practice instead of stubs:

- **Athar**: RAG feature pipeline (Chapter 4), evaluation (Chapter 7), and RAG inference (Chapter 9). It provides an Arabic corpus with normalization and Quran/Hadith-aware chunking, so the retrieval and evaluation work is done on real data rather than the book's sample corpus.
- **Baligh**: supervised fine-tuning (Chapter 5) and preference alignment (Chapter 6). It provides Arabic instruction and preference data, and the fine-tuning runs locally in fp16 on the RTX 5000.

## Sessions that are not book chapters

These are cross-cutting or supporting sessions:

| Session | Purpose |
|---------|---------|
| `session_4.2_embedding_models.md` | Bi-encoder vs cross-encoder deep dive (Ch 4 and Ch 9) |
| `session_8.1_docker.md` | Containerization (Ch 11 infra) |
| `session_8.2_cicd.md` | GitHub Actions CI/CD (Ch 11) |
| `session_8.3_zenml.md` | ZenML orchestration (Ch 2 and Ch 11) |
| `session_9.1_data_warehouse.md` | Backup and restore (Ch 11 operations) |
| `session_9.2_performance.md` | Batching, caching, quantization tuning (Ch 4 and Ch 8) |
| `session_9.3_security.md` | Secret management and API hardening (cross-cutting) |
| `session_4.3_streaming_cdc.md` | Batch vs streaming and CDC (Ch 4) |
| `session_10.1_deployment_topologies.md` | Deployment types and autoscaling (Ch 10) |

## Book terminology notes

- **ODM vs OVM**: the MongoDB mapper (`NoSQLBaseDocument`) is the **ODM** (object-document mapping); the Qdrant mapper (`VectorBaseDocument`) is the **OVM** (object-vector mapping). See `session_1.2_domain_layer.md`.
- **FTI architecture**: the feature/training/inference pipeline design from Chapter 1 underpins every pipeline in `pipelines/`.
- **Repository URL**: the book references `PacktPublishing/LLM-Engineering`; the current upstream is `PacktPublishing/LLM-Engineers-Handbook`.
- **Embedding model**: the book's prose uses `all-mpnet-base-v2` while `settings.py` defaults to `all-MiniLM-L6-v2`; both are configurable via `TEXT_EMBEDDING_MODEL_ID`.

## Decision log index

Architectural decisions are recorded in `docs/DECISIONS.md`. Each chapter's Definition of Done links its decision log to one or more of these entries.

| ADR | Title | Chapter | Status |
|-----|-------|---------|--------|
| ADR-001 | FTI architecture boundaries | 1 | Proposed |
| ADR-002 | Local-first tooling and environment | 2 | Proposed |
| ADR-003 | Data sources and warehouse choice | 3 | Proposed |
| ADR-004 | Chunking strategy for Arabic RAG | 4 | Proposed |
| ADR-005 | Embedding model selection | 4 | Proposed |
| ADR-006 | Base model and QLoRA configuration | 5 | Proposed |
| ADR-007 | Preference data generation | 6 | Proposed |
| ADR-008 | Evaluation set, metrics, and judge | 7 | Proposed |
| ADR-009 | Inference optimization and serving stack | 8 | Proposed |
| ADR-010 | Retriever and reranker selection | 9 | Proposed |
| ADR-011 | Deployment topology and autoscaling | 10 | Proposed |
| ADR-012 | CI/CD and monitoring stack | 11 | Proposed |
| ADR-013 | Testing and seeding strategy | Appendix | Proposed |

## Coverage status

All 11 chapters and the Appendix now have dedicated session coverage. The previously missing topics that were added:

- Chapter 8 inference optimization (`session_8.4_inference_optimization.md`).
- Chapter 7 RAG evaluation, Ragas, and ARES (`session_7.4_rag_evaluation.md`).
- Chapter 10 deployment taxonomy and autoscaling (`session_10.1_deployment_topologies.md`).
- Chapter 11 CT pipeline, cloud deployment, and alerting (`session_11.1_ct_pipeline_alerting.md`).
- Chapter 5 data curation (`session_5.4_data_curation.md`).
- Chapter 4 streaming and CDC (`session_4.3_streaming_cdc.md`).
- Appendix MLOps principles (`appendix_mlops_principles.md`).
