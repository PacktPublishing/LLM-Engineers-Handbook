# Reading Plan: LLM Engineer's Handbook, from scratch to production

This is the "how to study" layer. `BOOK-MAP.md` maps chapters to sessions and code, `CURRICULUM.md` gives the chapter-by-chapter path, and `DECISIONS.md` holds the ADRs. This document is the strategy on top of all three: a prioritized menu of decisions, because the book is best read as decisions to make, not as pages to finish in order.

Goal: master engineering an LLM Twin from concept to production, one runnable artifact per phase.

## The principle

Do not read in order, and do not read word for word. Treat the table of contents as a menu of decisions. Go deep where a decision shapes your system, skim where the book is reference material or cloud-specific. End every P0 or P1 section with an artifact or a test; reading without one does not count.

## Reading priority

| Priority | Chapters | Why | Read depth |
|----------|----------|-----|------------|
| P0 | Ch 1: FTI pipelines and system architecture | the backbone: feature, training, inference | word for word; draw the diagram; write an ADR |
| P0 | Ch 3: Data Engineering | data contract, cleaning, deduplication | selective; apply immediately |
| P0 | Ch 4: RAG Feature Pipeline | chunking, embeddings, vector DB, CDC, feature store | the deepest chapter; word for word; apply immediately |
| P0 | Ch 7: Evaluating LLMs | eval harness before any fine-tuning; Ragas and ARES | word for word; build a frozen eval set |
| P1 | Ch 9: RAG Inference Pipeline | query expansion, filtering, reranking, generation | word for word on the architecture; apply |
| P1 | Ch 5: Supervised Fine-Tuning | instruction datasets, curation, LoRA/QLoRA, parameters | selective; start when fine-tuning |
| P1 | Ch 6: Preference Alignment | preference datasets and DPO | after SFT and an eval baseline |
| P2 | Ch 8: Inference Optimization | KV cache, batching, quantization, parallelism | concepts now; benchmark later |
| P2 | Ch 10: Deployment | deployment types, monolith vs microservices, autoscaling | principles now; Docker and FastAPI locally |
| P2 | Ch 11 + Appendix: MLOps | CI/CD, monitoring, versioning, reproducibility | principles deep, tools quick |
| P3 | Ch 2: Tooling | Poetry, ZenML, Comet, Opik, MongoDB, Qdrant | skim; return during setup |
| P3 | AWS-specific sections | SageMaker, roles, DLCs, cloud pipelines | concept only; keep in a backlog |

## The execution contract

Six phases. Each maps a set of chapters to something you build from scratch, and the evidence that proves it.

| Phase | Read | Build from scratch | Evidence |
|-------|------|--------------------|----------|
| 1 | Ch 1 + 3 | FTI diagram, data ingestion, cleaning, and a data contract | document counts + schema tests |
| 2 | Ch 4 | chunking + embeddings + vector indexing | indexed count + retrieval smoke test |
| 3 | Ch 9 | retrieval + reranking + a citations API | sample queries + traces |
| 4 | Ch 7 | a frozen eval set + a Ragas/ARES harness | baseline metrics |
| 5 | Ch 5 + 6 | SFT, then DPO | before/after eval on the same set |
| 6 | Ch 8 + 10 + 11 | benchmark + Docker/FastAPI + CI/monitoring | latency/throughput + a workflow run |

## The menu of decisions

These are the forks the book exposes. Make each one yourself, record it in `DECISIONS.md`, and only then check the book's default.

- Base model, and whether it runs locally or in the cloud (Ch 5).
- Embedding model and its dimension (Ch 4).
- Chunk size and overlap (Ch 4).
- Vector store (Ch 4).
- Instruction dataset generation: prompt and volume (Ch 5).
- LoRA rank and alpha, and 4-bit vs 8-bit (Ch 5).
- Preference data and the DPO beta (Ch 6).
- Quantization format and serving engine, GGUF/GPTQ/AWQ and vLLM/TGI (Ch 8).
- Deployment topology: online, async, or batch, monolith or microservice (Ch 10).
- Evaluation set, metrics, and judge (Ch 7).
- Monitoring and alerting on drift (Ch 11).

## The read-depth rule

- Read word for word: Ch 1, Ch 4, Ch 7, and the architectural parts of Ch 9.
- Read selectively: Ch 3, Ch 5, Ch 6.
- Skim: Ch 2, Ch 8, Ch 10, Ch 11, and every AWS-specific section.
- Always implement after every P0 or P1 section.

## Done means mastered

You have finished when, from a clean machine, you can rebuild the full FTI system, justify every decision in the menu above, and produce four artifacts: a RAG API that returns cited answers, a frozen eval suite, an ADR for each decision, and a fine-tuned model that measurably beats its own baseline.

See also: `BOOK-MAP.md`, `CURRICULUM.md`, `ROADMAP.md`, `DECISIONS.md`, `GETTING_STARTED.md`.
