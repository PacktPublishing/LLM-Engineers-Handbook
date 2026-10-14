# Session 1.1: Project Overview & DDD Principles

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand what an **LLM Twin** is and why the project is built the way it is
- Understand the project's layered architecture and the direction dependencies flow
- Understand the **FTI** (Feature/Training/Inference) pipeline design and how the LLM Twin maps onto it
- Know how `pipelines/`, `steps/`, and the `llm_engineering/` package divide responsibilities
- Be able to navigate the codebase effectively and run the project locally
- Set up your development environment with the exact tool versions the book uses

---

## 🏗️ Architecture Overview

### The Big Picture

This project builds an **LLM Twin** - an LLM that learns to write in *your* voice, style, and personality using RAG and fine-tuned open-source models. The book's thesis is simple: an LLM reflects the data it is trained on. Feed it your own posts, articles, and code, and it starts to sound like you.

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Flow                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Collection → Preprocessing → Features → Training → RAG     │
│      ↓            ↓             ↓           ↓         ↓      │
│   Selenium    Cleaning     Embeddings   SFT/DPO   Retrieval │
│   Git/Medium  Chunking     Vector DB    SageMaker  + LLM    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The end goal is a REST API: a user sends a query, the system retrieves relevant chunks of *their* past writing from a vector DB, augments a prompt with that context, and the fine-tuned model drafts a new piece of content in their style.

### Why This Architecture Exists (the "why")

Training a model is the easy part. The book (Chapter 1) is explicit that the hard part is everything *around* the model:

- Ingesting, cleaning, and validating fresh data
- Keeping training-time and inference-time features consistent (**training-serving skew**)
- Versioning datasets and models so you know what was trained on what
- Serving cost-effectively and monitoring what happens in production

A monolithic script that couples "compute features", "train", and "predict" solves training-serving skew by construction, but makes features non-reusable, blocks streaming, and makes teams unable to work in parallel. The project therefore adopts the **FTI architecture**: three logical pipelines with stable interfaces.

### The FTI Pipeline Design

```
┌──────────────────────────────────────────────────────────────┐
│                    FTI pipelines (logical layers)             │
│                                                               │
│  DATA ──► Feature pipeline ──► feature store ──┐              │
│                                                 │             │
│            Training pipeline ◄───────────────────┘             │
│                    │                                          │
│                    ▼                                          │
│              model registry ──► Inference pipeline ──► user    │
└──────────────────────────────────────────────────────────────┘
```

- The **feature pipeline** takes raw data in, produces features/labels, and writes them to a feature store. It is the shared contract between train and serve.
- The **training pipeline** reads features from the store and writes a model to the model registry.
- The **inference pipeline** reads the model and features to make predictions.

The LLM Twin adds a fourth component, the **data collection pipeline**, because the MVP is built by one small team that also owns data engineering. For the LLM Twin specifically, the book uses a *logical* feature store (a vector DB used as a NoSQL store plus versioned artifacts) instead of a dedicated feature-store product.

### Layered Architecture (DDD)

The `llm_engineering/` package follows Domain-Driven Design layering. Higher layers may import lower layers; the reverse never happens.

```
┌─────────────────────────────────────────────────────────────┐
│                    Dependency Flow                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  infrastructure (AWS, MongoDB, Qdrant, FastAPI)             │
│       ↓ imports                                              │
│  model (Training, Inference, Evaluation)                    │
│       ↓ imports                                              │
│  application (Crawlers, Preprocessing, RAG, Dataset Gen)    │
│       ↓ imports                                              │
│  domain (Core business entities: Documents, Chunks, Datasets)│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle**: Higher layers can depend on lower layers, but NEVER the reverse. The `domain` layer is the innermost ring and knows nothing about Selenium, MongoDB, or ZenML. This is what makes the entities testable in isolation and reusable outside the pipeline framework.

> **FACT**: `llm_engineering/domain/base/nosql.py` imports `llm_engineering.infrastructure.db.mongo.connection`. That is an intentional exception-to-the-rule: the persistence behavior lives on the entity itself (the ODM/OVM pattern), so the domain base classes bind to a connection. Concrete entities only declare fields.

### How `pipelines/`, `steps/`, and `llm_engineering/` relate

This three-way split is the single most important structural decision in the repo:

```
┌────────────────────────────────────────────────────────────────┐
│  pipelines/   ZenML @pipeline functions. Glue steps together.   │
│      │        e.g. digital_data_etl, feature_engineering        │
│      ▼                                                          │
│  steps/       ZenML @step functions. Thin wrappers that call    │
│      │        llm_engineering and attach artifact metadata.     │
│      ▼                                                          │
│  llm_engineering/   Pure Python application + domain logic.     │
│                     No ZenML imports in domain/ or most of       │
│                     application/. Reusable from a REST API.      │
└────────────────────────────────────────────────────────────────┘
```

The payoff: you can swap ZenML for another orchestrator, or call the application logic directly from FastAPI, without touching the business logic.

---

## 📁 Project Structure Deep Dive

### Root Level

```
LLM-Engineers-Handbook/
├── .env.example              # Template for environment variables
├── .python-version           # Pins Python for pyenv (3.11.8)
├── pyproject.toml            # Poetry dependencies + Poe the Poet tasks
├── poetry.lock               # Exact locked dependency versions
├── docker-compose.yml        # Local MongoDB + Qdrant
├── Dockerfile                # Container image
├── configs/                  # ZenML YAML run configs
├── llm_engineering/          # Main Python package (domain/app/model/infra)
├── pipelines/                # ZenML @pipeline definitions
├── steps/                    # ZenML @step components
├── tools/                    # CLI entry points (tools/run.py, tools/rag.py)
├── code_snippets/            # Standalone teaching examples
└── tests/                    # Pytest suite
```

### The `pipelines/` Folder (verified contents)

```
pipelines/
├── __init__.py
├── digital_data_etl.py        # crawl links for a user -> MongoDB
├── end_to_end_data.py         # ETL -> feature engineering -> datasets
├── evaluating.py              # run the evaluation pipeline
├── export_artifact_to_json.py # ZenML artifact -> JSON
├── feature_engineering.py     # clean + chunk + embed + load to Qdrant
├── generate_datasets.py       # instruct + preference datasets
└── training.py                # fine-tune on SageMaker
```

### The `steps/` Folder (verified contents)

```
steps/
├── __init__.py
├── etl/
│   ├── crawl_links.py         # dispatch crawlers, record metadata
│   └── get_or_create_user.py  # UserDocument.get_or_create
├── feature_engineering/
│   ├── query_data_warehouse.py# fetch raw docs for authors (threaded)
│   ├── clean.py               # CleaningDispatcher over raw docs
│   ├── load_to_vector_db.py   # group_by_class + bulk_insert in batches
│   └── rag.py
├── generate_datasets/
│   ├── query_feature_store.py
│   ├── create_prompts.py
│   ├── generate_intruction_dataset.py
│   ├── generate_preference_dataset.py
│   └── push_to_huggingface.py
├── training/
│   └── train.py
├── evaluating/
│   └── evaluate.py
└── export/
    ├── serialize_artifact.py
    └── to_json.py
```

### The `llm_engineering` Package

```
llm_engineering/
├── __init__.py               # exports settings, application, domain, infrastructure
├── settings.py               # Centralized configuration
│
├── domain/                   # Core business entities (no external deps)
│   ├── types.py              # DataCategory StrEnum
│   ├── base/
│   │   ├── nosql.py          # NoSQLBaseDocument  -> MongoDB (ODM)
│   │   └── vector.py         # VectorBaseDocument -> Qdrant (OVM)
│   ├── documents.py          # Raw document models
│   ├── cleaned_documents.py  # Cleaned document models
│   ├── chunks.py             # Text chunk models
│   ├── embedded_chunks.py    # Vector-enabled chunks
│   ├── dataset.py            # Training dataset models
│   ├── queries.py            # Query / EmbeddedQuery
│   ├── prompt.py             # Prompt / GenerateDatasetSamplesPrompt
│   └── exceptions.py         # LLMTwinException, ImproperlyConfigured
│
├── application/              # Business logic
│   ├── crawlers/             # Selenium / LangChain / git scraping
│   ├── preprocessing/        # dispatchers, handlers, operations
│   ├── dataset/              # dataset generation
│   ├── rag/                  # retriever, query expansion, reranking
│   ├── networks/             # embedding + cross-encoder singletons
│   └── utils/                # split_user_full_name, misc.batch
│
├── model/                    # ML logic
│   ├── finetuning/           # SFT and DPO training
│   ├── inference/            # inference logic + run scripts
│   └── evaluation/           # model evaluation
│
└── infrastructure/           # External services
    ├── aws/                  # SageMaker + IAM
    ├── db/                   # mongo.py, qdrant.py connectors
    ├── files_io.py           # JsonFileManager
    ├── opik_utils.py
    └── inference_pipeline_api.py  # FastAPI app
```

> The exact filenames above are the ones present in this checkout. Confirm any file with `ls` or `git ls-files` before relying on it in a script.

---

## 🔍 Key Files Explained

### 1. `settings.py` - Centralized Configuration

**Purpose**: A single typed `Settings` object for the entire codebase, loaded from `.env` or the ZenML secret store.

The **real** file (this checkout) is shown below. Note the field names and defaults; do not invent extra fields.

```python
# llm_engineering/settings.py  (verbatim, abridged comments)
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from zenml.client import Client
from zenml.exceptions import EntityExistsError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Required settings even when working locally. ---
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "twin"

    # --- Required settings when deploying the code. ---
    DATABASE_HOST: str = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "twin"

    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None

    # --- Optional settings used to tweak the code. ---
    HF_MODEL_ID: str = "mlabonne/TwinLlama-3.1-8B-DPO"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    COPIES: int = 1
    GPUS: int = 1
    CPUS: int = 2

    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "twin"
    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin"
    TEMPERATURE_INFERENCE: float = 0.01
    TOP_P_INFERENCE: float = 0.9
    MAX_NEW_TOKENS_INFERENCE: int = 150

    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        return int(official_max_token_window * 0.90)

    @classmethod
    def load_settings(cls) -> "Settings":
        try:
            logger.info("Loading settings from the ZenML secret store.")
            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError):
            logger.warning(
                "Failed to load settings from the ZenML secret store. "
                "Defaulting to loading the settings from the '.env' file."
            )
            settings = Settings()
        return settings

    def export(self) -> None:
        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)
        client = Client()
        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning("Secret 'settings' already exists. Delete it manually ...")


settings = Settings.load_settings()
```

**Why this shape**:
- **Pydantic validation at startup**: a typo in `QDRANT_DATABASE_PORT` fails immediately, not three layers deep during a search.
- **`str | None = None` for secrets**: local development works without every credential, but a missing required-at-runtime key surfaces when the feature is actually used.
- **90% token window**: reserves headroom for prompt scaffolding so context building never exceeds the model window.
- **ZenML secret precedence**: `settings = Settings.load_settings()` means the ZenML secret store wins, then `.env`, then defaults. This lets cloud runs read the same config without shipping a `.env`.

**Usage**:
```python
from llm_engineering.settings import settings

db_name = settings.DATABASE_NAME          # "twin"
qcloud = settings.USE_QDRANT_CLOUD        # False locally
```

> **Corrected from an earlier draft**: `COMET_PROJECT` defaults to `"twin"` (not `"llm-engineering"`), and `DATABASE_NAME` exists and defaults to `"twin"`. There is no `SAGEMAKER_ENDPOINT_TRAINING` field. Always trust `llm_engineering/settings.py` over any doc.

---

### 2. `domain/types.py` - Data Categories

**Purpose**: A single `StrEnum` that tags every entity and collection in the system.

```python
# llm_engineering/domain/types.py  (verbatim)
from enum import StrEnum


class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"

    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
```

**Why `StrEnum`**: members *are* `str`, so they serialize cleanly into Mongo documents and Qdrant payloads and can be used directly as dict keys and collection names. The values span the whole lifecycle: raw data (`POSTS`), generated prompts (`PROMPT`, `QUERIES`), and datasets (`INSTRUCT_*`, `PREFERENCE_*`).

---

### 3. `llm_engineering/__init__.py` - Package Structure

```python
# llm_engineering/__init__.py  (verbatim)
from llm_engineering import application, domain, infrastructure
from llm_engineering.settings import settings

__all__ = ["settings", "application", "domain", "infrastructure"]
```

Note that `settings` is re-exported, so both `from llm_engineering import settings` and `from llm_engineering.settings import settings` work. `tools/run.py` uses the former.

---

### 4. `tools/run.py` - The CLI Entry Point

All pipelines are launched through one Click CLI. This is what the Poe tasks call.

```python
# tools/run.py  (excerpt)
from pipelines import (
    digital_data_etl, end_to_end_data, evaluating,
    export_artifact_to_json, feature_engineering,
    generate_datasets, training,
)

@click.command()
@click.option("--no-cache", is_flag=True, default=False)
@click.option("--run-etl", is_flag=True, default=False)
@click.option("--etl-config-filename", default="digital_data_etl_paul_iusztin.yaml")
# ... more flags ...
def main(no_cache, run_etl, etl_config_filename, ...):
    assert (run_end_to_end_data or run_etl or ... ), "Please specify an action to run."

    pipeline_args = {"enable_cache": not no_cache}
    root_dir = Path(__file__).resolve().parent.parent

    if run_etl:
        pipeline_args["config_path"] = root_dir / "configs" / etl_config_filename
        assert pipeline_args["config_path"].exists()
        pipeline_args["run_name"] = f"digital_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        digital_data_etl.with_options(**pipeline_args)(**run_args_etl)
```

**Why a single CLI**: one entry point, one place to discover every runnable pipeline. The `--etl-config-filename` flag injects a YAML config at runtime, so crawling for a different author needs no code change.

---

### 5. `pipelines/digital_data_etl.py` and `steps/etl/*` - A Pipeline End to End

The smallest complete example of the pipeline/step split:

```python
# pipelines/digital_data_etl.py  (verbatim)
from zenml import pipeline
from steps.etl import crawl_links, get_or_create_user


@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    user = get_or_create_user(user_full_name)
    last_step = crawl_links(user=user, links=links)

    return last_step.invocation_id
```

The step wrapper resolves the user; the pipeline just composes it:

```python
# steps/etl/get_or_create_user.py  (excerpt)
from typing_extensions import Annotated
from zenml import get_step_context, step
from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument


@step
def get_or_create_user(user_full_name: str) -> Annotated[UserDocument, "user"]:
    first_name, last_name = utils.split_user_full_name(user_full_name)
    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="user", metadata=_get_metadata(user_full_name, user))
    return user
```

**Why `Annotated[UserDocument, "user"]`**: ZenML turns the returned value into a versioned *artifact*. The `Annotated` name controls the artifact name in the dashboard. The step also attaches metadata (the query and the retrieved user) so you can inspect a run without downloading anything.

**Serialization caveat**: ZenML can serialize most values reducible to primitives, but UUIDs are not natively supported. The repo extends ZenML's materializer for UUIDs; if you add new step return types, verify they serialize.

---

## 🧠 Design Patterns Used

### 1. **Singleton Pattern**

Used for expensive, shared resources: database clients and embedding/cross-encoder models.

```python
# llm_engineering/application/networks/base.py  (verbatim)
from threading import Lock
from typing import ClassVar


class SingletonMeta(type):
    """This is a thread-safe implementation of Singleton."""

    _instances: ClassVar = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
```

**Why the lock**: two threads can race on first access. The lock makes initialization safe; the second thread sees the instance already created.

> **Important nuance**: the DB connectors (`MongoDatabaseConnector`, `QdrantDatabaseConnector`) use a *different* singleton idiom - overriding `__new__` and returning the raw client. See Session 1.3. `SingletonMeta` is used for models.

### 2. **Generic Types**

Used for type-safe document operations so `UserDocument.find(...)` returns `UserDocument | None`:

```python
# llm_engineering/domain/base/nosql.py  (excerpt)
T = TypeVar("T", bound="NoSQLBaseDocument")

class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)
```

**Why**: static analyzers (mypy) and IDEs infer the concrete return type; you catch mistakes before runtime.

### 3. **Strategy Pattern**

Used for crawler dispatching. A URL's netloc selects the crawler; new platforms register without editing existing code.

### 4. **Template Method (ODM / OVM)**

The persistence skeleton lives on the base class (`save`, `find`, `bulk_insert`), while subclasses override only the translation steps (`to_mongo` / `to_point`) and declare a `Settings`/`Config` inner class. Covered fully in Session 1.2.

---

## 🔌 The FTI Pipeline Architecture

The book maps the LLM Twin onto FTI plus a data pipeline. This checkout implements it as ZenML pipelines:

| FTI stage | Book responsibility | This repo |
|-----------|---------------------|-----------|
| Data collection | Crawl LinkedIn/Medium/Substack/GitHub into a warehouse | `pipelines/digital_data_etl.py` -> `steps/etl/*` -> MongoDB |
| Feature pipeline | Clean, chunk, embed into the feature store | `pipelines/feature_engineering.py` -> `steps/feature_engineering/*` -> Qdrant |
| Training pipeline | Fine-tune an LLM, store to registry | `pipelines/training.py`, `pipelines/generate_datasets.py` -> SageMaker |
| Inference pipeline | Serve RAG over REST | `llm_engineering/infrastructure/inference_pipeline_api.py`, `model/inference/*` |

The feature pipeline is the clearest expression of the split (verbatim):

```python
# pipelines/feature_engineering.py  (verbatim)
from zenml import pipeline
from steps import feature_engineering as fe_steps


@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    return [last_step_1.invocation_id, last_step_2.invocation_id]
```

Read it as: query raw -> clean -> store cleaned (fine-tuning snapshot) -> chunk+embed -> store embedded (RAG snapshot). Two snapshots, exactly as the book describes.

**Tradeoff of the "logical feature store"**: instead of a dedicated feature store, the project uses Qdrant as both a NoSQL-style store and a vector index, plus ZenML artifacts for training. Pros: fewer moving parts, reuse of the vector DB already required for RAG, cheap for an MVP. Cons: no point-in-time correctness or automatic feature lineage beyond what you attach manually.

---

## 🧰 The Full LLMOps Tooling Stack

The book dedicates Chapter 2 to the tools. Knowing which tool owns which responsibility prevents you from reaching for the wrong one later.

| Layer | Tool | Responsibility | Why this choice |
|-------|------|----------------|-----------------|
| Dependency + env | Poetry | Pin dependencies, isolate the venv | `pyproject.toml` + `poetry.lock` give reproducible installs |
| Task runner | Poe the Poet | Alias every CLI command | Commands live in `pyproject.toml`, not a stale README |
| Orchestrator | ZenML | Run pipelines, store artifacts, attach metadata | The "stack" abstraction avoids cloud vendor lock-in |
| Model registry | Hugging Face Hub | Version and share models | Open-source ecosystem integration (Unsloth, SageMaker) |
| Experiment tracker | Comet ML | Log metrics, hyperparameters, system stats | Free online tier; intuitive UI |
| Prompt monitoring | Opik | Trace prompt chains for debugging | Open-source, Comet-made, simple |
| NoSQL DB | MongoDB | Raw data warehouse | Flexible schema for unstructured text |
| Vector DB | Qdrant | Cleaned/embedded data + RAG search | Strong RPS/latency/index-time tradeoff |
| Cloud compute | AWS SageMaker | Fine-tune and serve the LLM | Full customizability, unlike Bedrock |

> **Cost note (from the book)**: every tool except AWS has a freemium tier. If you use a personal AWS account you pay for SageMaker. The book estimates **$50-$100** for the testing described, so set billing alarms before running training.

The alternative stack the book acknowledges: Airflow/Prefect/Metaflow/Dagster instead of ZenML; W&B/MLflow/Neptune instead of Comet; Langfuse/LangSmith instead of Opik; Milvus/Weaviate/Pinecone/Chroma/pgvector instead of Qdrant; GCP/Azure instead of AWS. The architecture does not depend on these choices.

---

## ⚖️ Architecture Tradeoffs

### FTI vs a monolithic batch pipeline

| Criterion | Monolithic batch | FTI pipelines |
|-----------|------------------|---------------|
| Training-serving skew | Solved by shared feature code | Solved by a versioned feature store |
| Feature reuse | None | Shared between train and serve |
| Scaling | Scale everything together | Scale each pipeline independently |
| Team parallelism | Hard (one codebase) | Each pipeline can be a different team |
| Streaming | Very hard to retrofit | Natural extension |
| Complexity for an MVP | Low | Slightly higher upfront |

### Dedicated feature store vs "logical" feature store

| Criterion | Dedicated feature store | Logical (vector DB + artifacts) |
|-----------|--------------------------|--------------------------------|
| Setup cost | High (another service) | Low (reuses Qdrant) |
| Point-in-time correctness | Built in | Not provided |
| Lineage | Built in | Manual (ZenML artifact metadata) |
| Offline training | Native | Artifacts |
| Online serving | Native | Vector DB |
| Best for | Large teams, strict ML governance | A 3-person MVP |

The book's verdict: implement the *properties* a feature store needs (versioned, sharable, reusable training dataset) without paying for a dedicated product until the system demands it.

### Layered (DDD) code vs framework-coupled code

Putting ZenML decorators directly on business logic is faster to start but couples your domain to the orchestrator. This repo pays a small structural tax (the `pipelines/` + `steps/` + `llm_engineering/` split) to keep `llm_engineering/` importable from a plain REST API or a test file. The measurable payoff: `tests/` can exercise the domain without a ZenML server.

### Worked example - the same logic, two callers

```
Orchestrated:  poetry poe run-feature-engineering-pipeline
               -> tools/run.py -> pipelines/feature_engineering.py
               -> steps/feature_engineering/*.py
               -> llm_engineering.application.preprocessing.*

Direct:        from llm_engineering.application.preprocessing import CleaningDispatcher
               cleaned = CleaningDispatcher.dispatch(raw_document)

Both paths reach the same dispatcher. ZenML is optional.
```

---

## 🛠️ Hands-On: Environment Setup

The book tests with **Python 3.11.8**, **Poetry 1.8.3**, **Poe the Poet 0.29.0**, and **ZenML 0.74.0**. This repo's `.python-version` contains `3.11.8`.

### Step 1: Install Python with pyenv

```bash
pyenv install 3.11.8
pyenv local 3.11.8     # writes .python-version in the repo
python --version       # Python 3.11.8
```

### Step 2: Install dependencies with Poetry

```bash
poetry install --without aws     # skip the AWS group for local work
poetry self add 'poethepoet[poetry_plugin]'
```

`--without aws` skips `sagemaker`, `s3fs`, and related packages until you deploy.

### Step 3: Configure the environment

```bash
cp .env.example .env
```

Fill in at least `OPENAI_API_KEY`, `HUGGINGFACE_ACCESS_TOKEN`, and `COMET_API_KEY`. Database defaults already work locally.

### Step 4: Stand up local infrastructure

```bash
poetry poe local-infrastructure-up
```

This runs `docker compose up -d` (MongoDB + Qdrant) and starts the local ZenML server (default `http://127.0.0.1:8237/`).

### Step 5: Test configuration

```python
# scripts/check_settings.py
from llm_engineering.settings import settings

print("OpenAI model:", settings.OPENAI_MODEL_ID)
print("Mongo URI:   ", settings.DATABASE_HOST)
print("Mongo DB:    ", settings.DATABASE_NAME)
print("Qdrant cloud:", settings.USE_QDRANT_CLOUD)
print("HF model id: ", settings.HF_MODEL_ID)
```

---

## 📝 Exercise 1: Explore the Codebase

### Task 1: Trace the Dependency Flow

1. Start at `llm_engineering/infrastructure/inference_pipeline_api.py`.
2. Find what it imports from `model/`.
3. Find what `model/` imports from `application/`.
4. Find what `application/` imports from `domain/`.

**Goal**: confirm the arrows only point downward.

### Task 2: Identify All Singleton Classes

Find every class using `SingletonMeta` and every connector using `__new__`.

**Hint**: search for `metaclass=SingletonMeta` and `_instance`.

### Task 3: Map Data Categories

List every place `DataCategory` is used.

**Hint**: `rg "DataCategory" llm_engineering`.

---

## 📝 Exercise 2: Trace a Pipeline End to End

### Task

Follow the `digital_data_etl` pipeline from the CLI to MongoDB.

1. Run `poetry poe run-digital-data-etl-maxime` and open the local ZenML dashboard.
2. In `pipelines/digital_data_etl.py`, read which two steps compose it.
3. In `steps/etl/crawl_links.py`, identify how the dispatcher is built and how metadata is accumulated.
4. In `steps/etl/get_or_create_user.py`, identify which ODM method persists the user.
5. Open `llm_engineering/domain/documents.py` and find the `Settings.name` for `UserDocument`.

**Deliverable**: a one-paragraph trace from "CLI flag" to "document in the `users` collection", naming each file and function.

**Goal**: internalize the pipeline -> step -> application -> domain -> infrastructure chain.

---

## ⚠️ Common Pitfalls

1. **Importing `domain` pulls in a live DB connection.** `nosql.py` runs `_database = connection.get_database(...)` at import time. Unit-testing an entity therefore needs importable settings and a reachable (or mocked) Mongo client. Prefer testing pure helpers, or monkeypatch the connection.
2. **`connection` is a client, not a connector.** `MongoDatabaseConnector()` returns a `MongoClient`. Do not expect connector methods on it.
3. **`.env` values are strings.** `USE_QDRANT_CLOUD=false` is parsed to a Python `bool` by pydantic-settings, but a typo like `False ` with a space can behave unexpectedly. Keep the exact spelling.
4. **ZenML secret wins silently.** If `settings` exists in the ZenML secret store, your `.env` edits are ignored. Run `poetry poe delete-settings-zenml` when confused.
5. **`--no-cache` is not the default.** ZenML caches step outputs. If you changed code and see stale artifacts, add `--no-cache`.
6. **Step outputs must serialize.** Returning a bare object ZenML cannot materialize fails the run. UUIDs needed a custom materializer - keep new return types primitive-friendly.
7. **Poe tasks chain.** `run-digital-data-etl` runs both authors sequentially; a Selenium browser issue on either author fails that task.
8. **`python --version` can lie.** If pyenv is not active, you may be on the system interpreter. Verify `.python-version` is being honored.
9. **Don't commit `.env`.** It holds API keys. `.env.example` is the committable template.

---

## 🎓 Knowledge Check

1. **What is the dependency flow direction?**
   - Answer: `infrastructure -> model -> application -> domain` (higher imports lower; never the reverse).

2. **Why use Singleton for database connections?**
   - Answer: To avoid creating multiple expensive connection pools; reuse one client per process.

3. **What does `Generic[T]` provide?**
   - Answer: Type safety and precise return types across document/chunk subclasses.

4. **Where is configuration stored, and in what precedence?**
   - Answer: `llm_engineering/settings.py` via pydantic-settings; ZenML secret store first, then `.env`, then typed defaults.

5. **What are the three FTI pipelines?**
   - Answer: Feature, training, and inference. The LLM Twin adds a data collection pipeline.

6. **What problem does FTI solve that a monolithic batch pipeline does not?**
   - Answer: Feature reusability, independent scaling/deployment, and team parallelism, while still avoiding training-serving skew via a versioned feature store.

7. **What is the role of `steps/` versus `pipelines/`?**
   - Answer: `steps/` are ZenML `@step` wrappers around `llm_engineering` logic; `pipelines/` compose steps into DAGs.

8. **Why is `llm_engineering` kept free of ZenML imports in `domain/`?**
   - Answer: So the business logic is reusable and testable independent of the orchestrator.

9. **What does `COMET_PROJECT` default to in this repo?**
   - Answer: `"twin"`.

10. **Why does `OPENAI_MAX_TOKEN_WINDOW` multiply by 0.90?**
    - Answer: To reserve headroom for prompt scaffolding and avoid exceeding the model's window.

11. **What does `poetry poe local-infrastructure-up` start?**
    - Answer: MongoDB and Qdrant via Docker Compose, plus the local ZenML server.

12. **Why must step return values be serializable?**
    - Answer: ZenML stores each step output as a versioned artifact; non-serializable types fail materialization.

13. **Which model registry does the book use, and why?**
    - Answer: Hugging Face, for shareability and integration with the open-source LLM ecosystem.

14. **Does the project fit on an RTX 5000 (Turing, sm_75, 16 GB)?**
    - Answer: The data and feature pipelines are CPU-only. Fine-tuning an 8B model is done on SageMaker `ml.g5.2xlarge`; local GPU work is limited to embeddings and small inference. Turing has no bfloat16 and no efficient FlashAttention-2, so use fp16 locally where relevant.

---

## 📖 Glossary

- **LLM Twin**: an LLM fine-tuned on your own data to mimic your style, voice, and knowledge.
- **FTI**: Feature/Training/Inference; a three-pipeline decomposition of an ML system with stable interfaces.
- **Feature store**: the versioned storage of features/labels shared by training and inference. Here it is "logical": a vector DB plus artifacts.
- **Model registry**: versioned storage of trained models and their metadata.
- **Training-serving skew**: a mismatch between features computed at training versus inference time.
- **DDD (Domain-Driven Design)**: organizing code around business entities; here the innermost `domain` layer.
- **ODM**: Object-Document Mapping; maps Python objects to MongoDB documents (Session 1.2).
- **OVM**: Object-Vector Mapping; maps Python objects to Qdrant points (Session 1.2).
- **Artifact**: any versioned, shareable file produced by a pipeline (dataset, model, log) with attached metadata.
- **Orchestrator**: a system that schedules and coordinates pipeline steps; here ZenML.
- **DAG**: Directed Acyclic Graph; the execution graph of a pipeline's steps.
- **CT/CI/CD**: Continuous Training / Continuous Integration / Continuous Delivery.

---

## 🔗 Next Session

**Session 1.2**: [Domain Layer - Data Modeling](./session_1.2_domain_layer.md)

We'll dive deep into:
- MongoDB document design with `NoSQLBaseDocument` (the ODM)
- Vector database design with `VectorBaseDocument` (the OVM)
- Every concrete entity in `llm_engineering/domain/`
- CRUD operations and the `_id`/`id` translation seam

---

## 📚 Additional Resources

- [Domain-Driven Design (Fowler)](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [From MLOps to ML Systems with FTI Pipelines (Hopsworks)](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)
- [ZenML Starter Guide](https://docs.zenml.io/user-guide/starter-guide)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Poetry Documentation](https://python-poetry.org/docs)
- [Poe the Poet](https://github.com/nat-n/poethepoet)
- [pyenv](https://github.com/pyenv/pyenv)
- [The LLM Engineer's Handbook repository](https://github.com/PacktPublishing/LLM-Engineers-Handbook)
- [Getting Started guide](../GETTING_STARTED.md)
- [Curriculum map](../CURRICULUM.md)

---

## 📎 References

- Iusztin, P., & Labonne, M. *LLM Engineer's Handbook*. Packt. Chapter 1 (pages 30-52), Chapter 2 (pages 54-82).
- Dowling, J. (2024). *From MLOps to ML Systems with Feature/Training/Inference Pipelines*. Hopsworks.
- Google Cloud. *MLOps: Continuous delivery and automation pipelines in machine learning*.
- Repository source files read for this session: `llm_engineering/settings.py`, `llm_engineering/__init__.py`, `llm_engineering/domain/types.py`, `llm_engineering/application/networks/base.py`, `pipelines/digital_data_etl.py`, `pipelines/feature_engineering.py`, `steps/etl/get_or_create_user.py`, `steps/etl/crawl_links.py`, `tools/run.py`, `pyproject.toml`, `.python-version`, `docker-compose.yml`.

---

**Estimated Time**: 2-3 hours

**Prerequisites**: Basic Python knowledge

**Outcome**: You understand the LLM Twin's layered architecture and FTI mapping, can navigate the codebase, and can run a pipeline locally.
