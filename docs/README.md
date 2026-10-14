# 📚 LLM Engineer's Handbook - Documentation Index

Welcome to the comprehensive learning resource for the **LLM Engineer's Handbook** project!

---

## 🎯 Quick Start

### First Time Here?

1. **Start with the Curriculum**: [CURRICULUM.md](./CURRICULUM.md)
2. **Follow Session Order**: Complete sessions sequentially
3. **Hands-On Practice**: Code along with each session
4. **Build Projects**: Apply learning to capstone projects

### Already Familiar with Basics?

- **RAG Deep Dive**: [Session 4.1](./sessions/session_4.1_advanced_rag.md)
- **LLM Training**: Sessions 5.1-5.3
- **Production Deployment**: Sessions 8.1-8.3

---

## 📖 Available Documentation

### 🎓 Curriculum & Learning Path

| Document | Description | Link |
|----------|-------------|------|
| **Curriculum** | Chapter-by-chapter learning path (11 chapters + Appendix) | [CURRICULUM.md](./CURRICULUM.md) |
| **Book Map** | Chapter -> session -> code mapping and per-chapter Definition of Done | [BOOK-MAP.md](./BOOK-MAP.md) |
| **Reading Plan** | Prioritized study strategy: menu of decisions, from scratch to production | [READING-PLAN.md](./READING-PLAN.md) |
| **Roadmap** | The six-milestone plan from architecture to production, with evidence gates | [ROADMAP.md](./ROADMAP.md) |
| **Decisions** | Architecture Decision Records (ADR log) referenced by the Book Map | [DECISIONS.md](./DECISIONS.md) |
| **Getting Started** | Setup and first steps | [GETTING_STARTED.md](./GETTING_STARTED.md) |
| **Code Intelligence** | Graphs, architecture, and tooling | [CODEBASE-INTELLIGENCE.md](./CODEBASE-INTELLIGENCE.md) |
| **Session 1.1** | Project Overview & DDD | [session_1.1_project_overview.md](./sessions/session_1.1_project_overview.md) |
| **Session 1.2** | Domain Layer - Data Modeling (ODM/OVM) | [session_1.2_domain_layer.md](./sessions/session_1.2_domain_layer.md) |
| **Session 1.3** | Infrastructure Layer - Connections | [session_1.3_infrastructure_layer.md](./sessions/session_1.3_infrastructure_layer.md) |
| **Session 2.1** | Web Crawling with Selenium | [session_2.1_web_crawling.md](./sessions/session_2.1_web_crawling.md) |
| **Session 2.2** | Text Preprocessing Pipeline | [session_2.2_text_preprocessing.md](./sessions/session_2.2_text_preprocessing.md) |
| **Session 2.3** | Feature Engineering Pipeline | [session_2.3_feature_engineering.md](./sessions/session_2.3_feature_engineering.md) |
| **Session 3.1** | Instruction Dataset Creation | [session_3.1_instruction_dataset.md](./sessions/session_3.1_instruction_dataset.md) |
| **Session 3.2** | Preference Dataset for DPO | [session_3.2_preference_dataset.md](./sessions/session_3.2_preference_dataset.md) |
| **Session 4.1** | Advanced RAG Architecture | [session_4.1_advanced_rag.md](./sessions/session_4.1_advanced_rag.md) |
| **Session 4.2** | Embedding Models & Cross-Encoders | [session_4.2_embedding_models.md](./sessions/session_4.2_embedding_models.md) |
| **Session 4.3** | Batch vs Streaming & CDC (Ch 4) | [session_4.3_streaming_cdc.md](./sessions/session_4.3_streaming_cdc.md) |
| **Session 5.1** | Supervised Fine-Tuning (SFT) | [session_5.1_sft.md](./sessions/session_5.1_sft.md) |
| **Session 5.2** | Direct Preference Optimization | [session_5.2_dpo.md](./sessions/session_5.2_dpo.md) |
| **Session 5.3** | AWS SageMaker Deployment | [session_5.3_sagemaker_deployment.md](./sessions/session_5.3_sagemaker_deployment.md) |
| **Session 5.4** | Instruction Data Curation (Ch 5) | [session_5.4_data_curation.md](./sessions/session_5.4_data_curation.md) |
| **Session 6.1** | FastAPI REST API | [session_6.1_fastapi_api.md](./sessions/session_6.1_fastapi_api.md) |
| **Session 6.2** | RAG Inference Flow | [session_6.2_rag_inference_flow.md](./sessions/session_6.2_rag_inference_flow.md) |
| **Session 7.1** | Experiment Tracking (Comet ML) | [session_7.1_comet_ml.md](./sessions/session_7.1_comet_ml.md) |
| **Session 7.2** | Prompt Monitoring (Opik) | [session_7.2_opik_monitoring.md](./sessions/session_7.2_opik_monitoring.md) |
| **Session 7.3** | Model Evaluation | [session_7.3_model_evaluation.md](./sessions/session_7.3_model_evaluation.md) |
| **Session 7.4** | RAG Evaluation, Ragas & ARES (Ch 7) | [session_7.4_rag_evaluation.md](./sessions/session_7.4_rag_evaluation.md) |
| **Session 8.1** | Docker & Local Infrastructure | [session_8.1_docker.md](./sessions/session_8.1_docker.md) |
| **Session 8.2** | CI/CD with GitHub Actions | [session_8.2_cicd.md](./sessions/session_8.2_cicd.md) |
| **Session 8.3** | ZenML Orchestration | [session_8.3_zenml.md](./sessions/session_8.3_zenml.md) |
| **Session 8.4** | Inference Optimization (Ch 8) | [session_8.4_inference_optimization.md](./sessions/session_8.4_inference_optimization.md) |
| **Session 9.1** | Data Warehouse Operations | [session_9.1_data_warehouse.md](./sessions/session_9.1_data_warehouse.md) |
| **Session 9.2** | Performance Optimization | [session_9.2_performance.md](./sessions/session_9.2_performance.md) |
| **Session 9.3** | Security Best Practices | [session_9.3_security.md](./sessions/session_9.3_security.md) |
| **Session 10.1** | Deployment Topologies & Autoscaling (Ch 10) | [session_10.1_deployment_topologies.md](./sessions/session_10.1_deployment_topologies.md) |
| **Session 11.1** | MLOps, CT Pipeline & Alerting (Ch 11) | [session_11.1_ct_pipeline_alerting.md](./sessions/session_11.1_ct_pipeline_alerting.md) |
| **Appendix** | MLOps Principles | [appendix_mlops_principles.md](./sessions/appendix_mlops_principles.md) |

### 📝 Code Snippets

Located in `code_snippets/`:

| File | Topic | Description |
|------|-------|-------------|
| `03_orm.py` | MongoDB ORM | NoSQL document operations |
| `03_custom_odm_example.py` | Custom ODM | Advanced document patterns |
| `08_text_embeddings.py` | Text Embeddings | Sentence transformer usage |
| `08_instructor_embeddings.py` | Instructor Models | Advanced embeddings |

### 🔧 Configuration Files

Located in `configs/`:

| File | Purpose |
|------|---------|
| `digital_data_etl_*.yaml` | ETL pipeline configuration |
| `feature_engineering.yaml` | Feature engineering params |
| `generate_*.yaml` | Dataset generation configs |
| `training.yaml` | Training hyperparameters |
| `evaluating.yaml` | Evaluation settings |

---

## 🗺️ Learning Modules Overview

### Module 1: Foundations (Week 1-2)

**Goal**: Understand project architecture and DDD principles

```
Session 1.1: Project Overview & DDD Principles
├── Layered architecture
├── Dependency flow
├── Settings management
└── Environment setup

Session 1.2: Domain Layer - Data Modeling
├── MongoDB documents
├── Vector documents
├── Pydantic models
└── CRUD operations

Session 1.3: Infrastructure Layer
├── MongoDB connection
├── Qdrant connection
├── Singleton pattern
└── Database operations
```

---

### Module 2: Data Engineering (Week 3-5)

**Goal**: Build data collection and preprocessing pipelines

```
Session 2.1: Web Crawling with Selenium
├── Base crawler implementation
├── Crawler dispatcher
├── GitHub crawler
├── Medium crawler
└── Custom crawlers

Session 2.2: Text Preprocessing
├── Text cleaning
├── Chunking strategies
├── Handler pattern
└── Quality filtering

Session 2.3: Feature Engineering
├── Embedding generation
├── Vector database loading
├── Batch processing
└── Feature store
```

---

### Module 3: Dataset Generation (Week 6-7)

**Goal**: Generate training datasets with LLMs

```
Session 3.1: Instruction Dataset Creation
├── Prompt engineering
├── LLM-based generation
├── Structured output parsing
└── Dataset formatting

Session 3.2: Preference Dataset (DPO)
├── Chosen/rejected pairs
├── Quality filtering
├── DPO theory
└── Dataset preparation
```

---

### Module 4: Advanced RAG (Week 8-9)

**Goal**: Implement multi-stage retrieval

```
Session 4.1: Advanced RAG Architecture
├── Query expansion
├── Self-query (metadata extraction)
├── Reranking with cross-encoders
└── Parallel retrieval

Session 4.2: Embedding Models
├── Sentence transformers
├── Cross-encoders
├── Embedding comparison
└── Performance optimization
```

---

### Module 5: LLM Training (Week 10-12)

**Goal**: Fine-tune LLMs with SFT and DPO

```
Session 5.1: Supervised Fine-Tuning (SFT)
├── LoRA implementation
├── Unsloth optimization
├── Training configuration
└── Experiment tracking

Session 5.2: Direct Preference Optimization
├── DPO theory
├── Dataset preparation
├── DPO training
└── SFT vs DPO comparison

Session 5.3: AWS SageMaker Deployment
├── IAM roles
├── Endpoint configuration
├── Deployment strategies
└── Inference testing
```

---

### Module 6: Inference & APIs (Week 13-14)

**Goal**: Build production inference APIs

```
Session 6.1: FastAPI REST API
├── Endpoint design
├── Request validation
├── Async inference
└── Error handling

Session 6.2: RAG Inference Flow
├── End-to-end RAG
├── Context integration
├── Prompt engineering
└── Response generation
```

---

### Module 7: Monitoring & Evaluation (Week 15-16)

**Goal**: Track experiments and evaluate models

```
Session 7.1: Experiment Tracking (Comet ML)
├── Metric logging
├── Dashboard creation
├── Experiment comparison
└── Best practices

Session 7.2: Prompt Monitoring (Opik)
├── Trace configuration
├── Prompt analysis
├── Performance metrics
└── Debugging

Session 7.3: Model Evaluation
├── LLM-as-a-judge
├── Pairwise comparison
├── Automated metrics
└── Evaluation pipelines
```

---

### Module 8: Production Deployment (Week 17-18)

**Goal**: Deploy to production with CI/CD

```
Session 8.1: Docker & Local Infrastructure
├── Containerization
├── Docker Compose
├── Environment management
└── Service orchestration

Session 8.2: CI/CD with GitHub Actions
├── Workflow configuration
├── Automated testing
├── Deployment automation
└── Quality gates

Session 8.3: ZenML Orchestration
├── Pipeline design
├── Step implementation
├── Artifact management
└── Cloud deployment
```

---

### Module 9: Advanced Topics (Week 19-20)

**Goal**: Master advanced techniques

```
Session 9.1: Data Warehouse Operations
├── Export/import
├── Backup strategies
├── Data versioning
└── Migration scripts

Session 9.2: Performance Optimization
├── Profiling
├── Batching
├── Caching
└── Quantization

Session 9.3: Security Best Practices
├── Authentication
├── Rate limiting
├── Input validation
└── Secrets management
```

---

## 🎯 Capstone Projects

### Project 1: Build Your LLM Twin

**Objective**: Create a personalized LLM that mimics your writing style

**Steps**:
1. Collect your writing samples (Medium, LinkedIn, GitHub)
2. Generate custom instruction dataset
3. Fine-tune Llama 3.1 8B
4. Deploy RAG system with your knowledge

**Technologies**: All modules combined

---

### Project 2: Production RAG System

**Objective**: Build a production-ready RAG system

**Steps**:
1. Implement advanced RAG (HyDE, query rewriting)
2. Add monitoring and evaluation
3. Deploy to AWS with auto-scaling
4. Create analytics dashboard

**Technologies**: RAG, FastAPI, AWS, Monitoring

---

### Project 3: Multi-Tenant LLM Platform

**Objective**: Support multiple users with isolated data

**Steps**:
1. Implement user isolation
2. Add billing and quotas
3. Create admin dashboard
4. Deploy with Kubernetes

**Technologies**: Multi-tenancy, Kubernetes, Billing

---

## 🔧 Quick Reference

### Essential Commands

```bash
# Environment Setup
.venv\Scripts\activate
python --version  # Should be 3.11

# ZenML Operations
zenml up --port 8237        # Start server
zenml down                  # Stop server
zenml stack set default     # Set default stack
zenml pipeline list         # List pipelines

# Pipeline Execution
python -m tools.run --run-etl --no-cache
python -m tools.run --run-feature-engineering --no-cache
python -m tools.run --run-generate-instruct-datasets --no-cache
python -m tools.run --run-training --no-cache

# Inference
python -m tools.ml_service  # Start API
python -m tools.rag         # Test RAG
```

### Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **ZenML** | http://localhost:8237 | username: `default`, password: (empty) |
| **Qdrant** | http://localhost:6333/dashboard | None (local) |
| **MongoDB** | mongodb://127.0.0.1:27017 | llm_engineering / llm_engineering |
| **Comet ML** | https://www.comet.com/ | Your account |
| **Opik** | https://www.comet.com/opik | Your account |

---

## 📚 Additional Resources

### Official Documentation

- [ZenML](https://docs.zenml.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LangChain](https://python.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic](https://docs.pydantic.dev/)

### Books & Courses

- **LLM Engineer's Handbook** (Amazon): [Link](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/)
- **Hugging Face Course**: [Link](https://huggingface.co/learn)
- **LangChain Academy**: [Link](https://academy.langchain.com/)

### Community

- **GitHub Issues**: Report bugs and request features
- **Discord**: Join LLM engineering communities
- **LinkedIn**: Connect with other learners

---

## 🎓 Progress Tracking

Use this checklist to track your progress:

### Foundation (Week 1-2)
- [ ] Session 1.1: Project Overview
- [ ] Session 1.2: Domain Layer
- [ ] Session 1.3: Infrastructure Layer

### Data Engineering (Week 3-5)
- [ ] Session 2.1: Web Crawling
- [ ] Session 2.2: Text Preprocessing
- [ ] Session 2.3: Feature Engineering

### Dataset Generation (Week 6-7)
- [ ] Session 3.1: Instruction Datasets
- [ ] Session 3.2: Preference Datasets

### RAG System (Week 8-9)
- [ ] Session 4.1: Advanced RAG
- [ ] Session 4.2: Embedding Models

### LLM Training (Week 10-12)
- [ ] Session 5.1: SFT
- [ ] Session 5.2: DPO
- [ ] Session 5.3: SageMaker Deployment

### Inference (Week 13-14)
- [ ] Session 6.1: FastAPI
- [ ] Session 6.2: RAG Inference

### Monitoring (Week 15-16)
- [ ] Session 7.1: Comet ML
- [ ] Session 7.2: Opik
- [ ] Session 7.3: Evaluation

### Production (Week 17-18)
- [ ] Session 8.1: Docker
- [ ] Session 8.2: CI/CD
- [ ] Session 8.3: ZenML

### Advanced (Week 19-20)
- [ ] Session 9.1: Data Warehouse
- [ ] Session 9.2: Performance
- [ ] Session 9.3: Security

### Capstone
- [ ] Project 1: LLM Twin
- [ ] Project 2: Production RAG
- [ ] Project 3: Multi-Tenant Platform

---

## 🤝 Contributing

Found an error? Want to improve the documentation?

1. Fork the repository
2. Create a branch: `git checkout -b docs/improvement`
3. Make your changes
4. Submit a PR

---

## 📞 Support

- **Technical Issues**: Check existing [GitHub Issues](https://github.com/PacktPublishing/LLM-Engineers-Handbook/issues)
- **Learning Questions**: Create a new issue with the `question` label
- **General Discussion**: Join the community Discord

---

## 📝 License

This documentation follows the same license as the main repository.

---

**Last Updated**: March 29, 2026

**Maintained By**: The LLM Engineer's Handbook Community

**Start Learning Now**: Begin with [Session 1.1](./sessions/session_1.1_project_overview.md)
