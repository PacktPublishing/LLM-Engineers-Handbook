# LLM Engineer's Handbook - Learning Curriculum

## 📚 Overview

This is a chapter-by-chapter learning path for the **LLM Engineer's Handbook** by Paul Iusztin and Maxime Labonne. It follows the book's real structure: **11 chapters + an Appendix**. Each chapter links to detailed session walkthroughs grounded in the repository code.

**See also:** [`BOOK-MAP.md`](./BOOK-MAP.md) for the exact chapter → session → code mapping, each chapter's Definition of Done, the dependency order, and the cloud-gated backlog, [`READING-PLAN.md`](./READING-PLAN.md) for the prioritized study strategy, [`ROADMAP.md`](./ROADMAP.md) for the six-milestone plan with evidence gates, and [`DECISIONS.md`](./DECISIONS.md) for the Architecture Decision Record (ADR) log.

### 🎯 Learning Outcomes

By completing this curriculum, you will be able to:
- Design an end-to-end LLM system with the **FTI (feature/training/inference)** architecture
- Build data collection and RAG feature pipelines (crawling, cleaning, chunking, embedding)
- Generate instruction and preference datasets, and curate them to production quality
- Fine-tune (SFT/LoRA/QLoRA) and align (DPO) an LLM
- Evaluate models and RAG systems with the right benchmarks
- Optimize inference (KV cache, batching, speculative decoding, quantization)
- Deploy a RAG microservice to AWS SageMaker and FastAPI
- Operate it with CI/CD/CT, prompt monitoring, and alerting

---

## 📖 Chapter-by-Chapter Path

### Chapter 1 - Understanding the LLM Twin Concept and Its Architecture
- **Sessions:** [`session_1.1_project_overview.md`](./sessions/session_1.1_project_overview.md)
- **Topics:** what an LLM Twin is, the MVP, the FTI pipeline design, the layered (DDD) architecture.
- **Hands-on:** map the repository layers and the dependency flow.

### Chapter 2 - Tooling and Installation
- **Sessions:** [`GETTING_STARTED.md`](./GETTING_STARTED.md), [`session_1.3_infrastructure_layer.md`](./sessions/session_1.3_infrastructure_layer.md)
- **Topics:** Poetry, Poe the Poet, ZenML, Comet ML, Opik, MongoDB, Qdrant, AWS, SageMaker.
- **Hands-on:** bring up the local stack and verify both databases.

### Chapter 3 - Data Engineering
- **Sessions:** [`session_2.1_web_crawling.md`](./sessions/session_2.1_web_crawling.md), [`session_1.2_domain_layer.md`](./sessions/session_1.2_domain_layer.md), [`session_1.3_infrastructure_layer.md`](./sessions/session_1.3_infrastructure_layer.md)
- **Topics:** the data collection pipeline, crawler dispatcher, Selenium crawlers, the NoSQL data warehouse, the **ODM** pattern.
- **Hands-on:** add a custom crawler and inspect the warehouse.

### Chapter 4 - RAG Feature Pipeline
- **Sessions:** [`session_2.2_text_preprocessing.md`](./sessions/session_2.2_text_preprocessing.md), [`session_2.3_feature_engineering.md`](./sessions/session_2.3_feature_engineering.md), [`session_4.2_embedding_models.md`](./sessions/session_4.2_embedding_models.md), [`session_4.3_streaming_cdc.md`](./sessions/session_4.3_streaming_cdc.md)
- **Topics:** RAG basics, embeddings, vector DBs, advanced RAG overview, cleaning, chunking, embedding, the **OVM** pattern, batch vs streaming and CDC.
- **Hands-on:** run the feature pipeline and search the vector store.

### Chapter 5 - Supervised Fine-Tuning
- **Sessions:** [`session_3.1_instruction_dataset.md`](./sessions/session_3.1_instruction_dataset.md), [`session_5.4_data_curation.md`](./sessions/session_5.4_data_curation.md), [`session_5.1_sft.md`](./sessions/session_5.1_sft.md)
- **Topics:** instruction dataset creation and curation (filtering, dedup, decontamination, augmentation), SFT, LoRA/QLoRA, training parameters.
- **Hands-on:** generate and curate an instruction dataset, then fine-tune.

### Chapter 6 - Fine-Tuning with Preference Alignment
- **Sessions:** [`session_3.2_preference_dataset.md`](./sessions/session_3.2_preference_dataset.md), [`session_5.2_dpo.md`](./sessions/session_5.2_dpo.md)
- **Topics:** preference datasets (chosen/rejected), RLHF vs DPO, the DPO objective and beta.
- **Hands-on:** build a preference dataset and run DPO.

### Chapter 7 - Evaluating LLMs
- **Sessions:** [`session_7.3_model_evaluation.md`](./sessions/session_7.3_model_evaluation.md), [`session_7.4_rag_evaluation.md`](./sessions/session_7.4_rag_evaluation.md)
- **Topics:** ML vs LLM evaluation, general/domain/task-specific benchmarks, LLM-as-a-judge, RAG evaluation with **Ragas** and **ARES**.
- **Hands-on:** run the project's judge and a Ragas evaluation.

### Chapter 8 - Inference Optimization
- **Sessions:** [`session_8.4_inference_optimization.md`](./sessions/session_8.4_inference_optimization.md)
- **Topics:** KV cache, continuous batching, speculative decoding, optimized attention, model parallelism, quantization (GGUF, GPTQ, EXL2, AWQ), inference engines (TGI, vLLM, TensorRT-LLM).
- **Hands-on:** estimate a KV cache, compare quantization formats, measure batching.

### Chapter 9 - RAG Inference Pipeline
- **Sessions:** [`session_4.1_advanced_rag.md`](./sessions/session_4.1_advanced_rag.md), [`session_6.2_rag_inference_flow.md`](./sessions/session_6.2_rag_inference_flow.md), [`session_4.2_embedding_models.md`](./sessions/session_4.2_embedding_models.md)
- **Topics:** query expansion, self-querying, filtered vector search, reranking, the end-to-end inference flow.
- **Hands-on:** trace a query through retrieval to generation.

### Chapter 10 - Inference Pipeline Deployment
- **Sessions:** [`session_5.3_sagemaker_deployment.md`](./sessions/session_5.3_sagemaker_deployment.md), [`session_6.1_fastapi_api.md`](./sessions/session_6.1_fastapi_api.md), [`session_10.1_deployment_topologies.md`](./sessions/session_10.1_deployment_topologies.md)
- **Topics:** deployment types (online/asynchronous/batch), monolithic vs microservices, Hugging Face DLCs, SageMaker endpoints, FastAPI, autoscaling.
- **Hands-on:** deploy the model and call the FastAPI RAG service.

### Chapter 11 - MLOps and LLMOps
- **Sessions:** [`session_7.1_comet_ml.md`](./sessions/session_7.1_comet_ml.md), [`session_7.2_opik_monitoring.md`](./sessions/session_7.2_opik_monitoring.md), [`session_8.1_docker.md`](./sessions/session_8.1_docker.md), [`session_8.2_cicd.md`](./sessions/session_8.2_cicd.md), [`session_8.3_zenml.md`](./sessions/session_8.3_zenml.md), [`session_9.1_data_warehouse.md`](./sessions/session_9.1_data_warehouse.md), [`session_11.1_ct_pipeline_alerting.md`](./sessions/session_11.1_ct_pipeline_alerting.md)
- **Topics:** DevOps → MLOps → LLMOps, CI/CD/CT, cloud deployment, prompt monitoring, guardrails, human feedback, alerting.
- **Hands-on:** wire a CT trigger and an alerter, and add production feedback.

### Appendix - MLOps Principles
- **Sessions:** [`appendix_mlops_principles.md`](./sessions/appendix_mlops_principles.md)
- **Topics:** automation, versioning, experiment tracking, testing, monitoring, reproducibility.
- **Hands-on:** audit the project against the six principles and close a gap.

---

## 🔧 Quick Reference

### Essential Commands

```bash
# Environment
.venv\Scripts\activate

# Local infrastructure
poetry poe local-docker-infrastructure-up
poetry poe local-zenml-server-up
poetry poe set-local-stack

# Pipelines
python -m tools.run --run-etl --no-cache
python -m tools.run --run-feature-engineering --no-cache
python -m tools.run --run-generate-instruct-datasets --no-cache
python -m tools.run --run-generate-preference-datasets --no-cache
python -m tools.run --run-training --no-cache
python -m tools.run --run-evaluation

# Inference
python -m tools.ml_service
python -m tools.rag
```

### Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **ZenML** | http://localhost:8237 | username: `default`, password: (empty) |
| **Qdrant** | http://localhost:6333/dashboard | None (local) |
| **MongoDB** | mongodb://127.0.0.1:27017 | llm_engineering / llm_engineering |
| **Comet ML** | https://www.comet.com/ | Your account |
| **Opik** | https://www.comet.com/opik | Your account |

### Tools and Technologies

| Category | Tools |
|----------|-------|
| **Core** | Python 3.11, Pydantic, FastAPI |
| **ML/LLM** | Transformers, Unsloth, Sentence Transformers, LangChain, vLLM |
| **Databases** | MongoDB, Qdrant |
| **Cloud** | AWS SageMaker, Docker, GitHub Actions |
| **Monitoring** | Comet ML, Opik |
| **Orchestration** | ZenML |

---

## 📈 Progress Tracking

Track by chapter:

- [ ] Chapter 1: Concept and architecture
- [ ] Chapter 2: Tooling and installation
- [ ] Chapter 3: Data engineering
- [ ] Chapter 4: RAG feature pipeline
- [ ] Chapter 5: Supervised fine-tuning
- [ ] Chapter 6: Preference alignment
- [ ] Chapter 7: Evaluating LLMs
- [ ] Chapter 8: Inference optimization
- [ ] Chapter 9: RAG inference pipeline
- [ ] Chapter 10: Inference pipeline deployment
- [ ] Chapter 11: MLOps and LLMOps
- [ ] Appendix: MLOps principles

---

**Next Steps**: Start with [`READING-PLAN.md`](./READING-PLAN.md) to learn in priority order, then open the book at Chapter 1 and follow `session_1.1`. Use [`BOOK-MAP.md`](./BOOK-MAP.md) to jump to any chapter's sessions, track each chapter's status (Covered → Implemented → Tested → Deployed), and record engineering decisions in [`DECISIONS.md`](./DECISIONS.md).
