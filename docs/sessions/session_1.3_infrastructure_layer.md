# Session 1.3: Infrastructure Layer - Database Connections

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how MongoDB and Qdrant connections are created as singletons
- Know how `settings` is loaded from `.env`, defaults, and the ZenML secret store
- Read and write JSON artifacts with `JsonFileManager`
- Start the local database stack with Docker Compose
- Verify both stores with real diagnostic queries
- Understand the failure modes of import-time connection and environment switching

---

## 🏗️ Architecture Overview

The infrastructure layer is the outermost layer. It owns everything that talks to the outside world: databases, AWS, HTTP APIs, and tracing. The domain layer imports connections from here, but infrastructure never imports pipelines.

```
┌──────────────────────────────────────────────────────────────┐
│                  infrastructure/ (outermost)                  │
│                                                               │
│   db/                    aws/              files_io.py         │
│   ├── mongo.py           ├── deploy/       opik_utils.py       │
│   └── qdrant.py          └── roles/        inference_pipeline_api.py
│        │                      │                   │            │
│        ▼                      ▼                   ▼            │
│   MongoClient            SageMaker           FastAPI app       │
│   QdrantClient           IAM roles                             │
└──────────────────────────────────────────────────────────────┘
                 ▲                       ▲
                 │ imports               │ imports
          domain/base/*.py        model/ + application/
```

### Connection Flow

```
.env  ──►  Settings (pydantic-settings)  ──►  settings singleton
                                                    │
                     ┌──────────────────────────────┴──────────────────────┐
                     ▼                                                      ▼
          MongoDatabaseConnector()                              QdrantDatabaseConnector()
          returns MongoClient                                   returns QdrantClient
                     │                                                      │
          connection.get_database(settings.DATABASE_NAME)       connection.search(...)
                     │                                                      │
          domain/base/nosql.py                                  domain/base/vector.py
```

### Who imports what (verified)

| From | To | What | Why |
|------|----|------|-----|
| `domain/base/nosql.py` | `infrastructure/db/mongo.py` | `connection` | ODM persistence |
| `domain/base/nosql.py` | `settings.py` | `settings` | DB name |
| `domain/base/vector.py` | `infrastructure/db/qdrant.py` | `connection` | OVM persistence |
| `domain/base/vector.py` | `application/networks/embeddings.py` | `EmbeddingModelSingleton` | collection vector size |
| `infrastructure/db/mongo.py` | `settings.py` | `settings` | Mongo URI |
| `infrastructure/db/qdrant.py` | `settings.py` | `settings` | Qdrant host/port/cloud |

Note the domain-to-application edge: `vector.py` imports the embedding singleton only to read `embedding_size`. It does not load the model for persistence, but constructing the singleton *does* instantiate `SentenceTransformer` on first use. See Common Pitfalls.

---

## 📁 Key Files Explained

### 1. `llm_engineering/settings.py` - Central Configuration

**Purpose**: One typed `Settings` object for the entire codebase.

```python
# llm_engineering/settings.py  (verbatim, abridged)
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
```

**Derived property**:

```python
    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)

        return max_token_window
```

The 90% factor reserves headroom for prompt overhead, so context building never exceeds the real model window.

**ZenML secret-store integration**:

```python
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
            logger.warning(
                "Secret 'settings' already exists. Delete it manually by running "
                "'zenml secret delete settings', before trying to recreate it."
            )


settings = Settings.load_settings()
```

**Key Concepts**:
- **Priority order**: ZenML secret store → `.env` → typed defaults. Locally you only need `.env`.
- **`settings` is a module-level singleton**, imported as `from llm_engineering.settings import settings`.
- **Secrets never hard-coded**: `OPENAI_API_KEY`, `DATABASE_HOST`, and AWS keys are all `str | None` with safe local defaults.
- **`export()`** pushes the current settings into ZenML so remote pipeline runs (CI, cloud) read the same config without shipping `.env`. The Poe task `export-settings-to-zenml` calls it.
- **`QDRANT_CLOUD_URL` defaults to the literal `"str"`.** This is a placeholder, not a valid URL. It only matters when `USE_QDRANT_CLOUD=True`; if you flip that flag without setting a real URL, the client will fail at connect time. This is a real footgun.

**Worked example - resolving a setting**:

```
.env:            DATABASE_NAME=my_twin
ZenML secret:    (does not exist)
result:          settings.DATABASE_NAME == "my_twin"

.env:            DATABASE_NAME=my_twin
ZenML secret:    {"DATABASE_NAME": "cloud_twin"}
result:          settings.DATABASE_NAME == "cloud_twin"   # secret wins, silently
```

---

### 2. `llm_engineering/infrastructure/db/mongo.py` - MongoDB Connector

**Purpose**: A lazy, process-wide `MongoClient`.

```python
# llm_engineering/infrastructure/db/mongo.py  (verbatim)
from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from llm_engineering.settings import settings


class MongoDatabaseConnector:
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.DATABASE_HOST)
            except ConnectionFailure as e:
                logger.error(f"Couldn't connect to the database: {e!s}")

                raise

        logger.info(f"Connection to MongoDB with URI successful: {settings.DATABASE_HOST}")

        return cls._instance


connection = MongoDatabaseConnector()
```

**Key Concepts**:
- **Singleton via `__new__`**: `MongoDatabaseConnector()` always returns the same `MongoClient`. Note that `MongoClient` itself is already thread-safe and pools connections.
- **`connection` is actually a `MongoClient`**, not a connector instance. The module-level name reads like a connection, which is how `domain/base/nosql.py` uses it.
- **Lazy** because the module is only imported when first needed; the client connects on first operation.
- **`ConnectionFailure` is raised on construction only in some cases.** PyMongo's `MongoClient(...)` is lazy by default: it does *not* immediately connect, so a down server usually does **not** raise here. The first real operation raises later. Treat this `try/except` as best-effort, not a guaranteed health check.

**How the domain layer consumes it**:

```python
# llm_engineering/domain/base/nosql.py
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings

_database = connection.get_database(settings.DATABASE_NAME)
```

Every document class shares one `_database` handle and selects its collection with `get_collection_name()`.

---

### 3. `llm_engineering/infrastructure/db/qdrant.py` - Qdrant Connector

**Purpose**: The same singleton pattern, with a cloud/local switch.

```python
# llm_engineering/infrastructure/db/qdrant.py  (verbatim)
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from llm_engineering.settings import settings


class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __new__(cls, *args, **kwargs) -> QdrantClient:
        if cls._instance is None:
            try:
                if settings.USE_QDRANT_CLOUD:
                    cls._instance = QdrantClient(
                        url=settings.QDRANT_CLOUD_URL,
                        api_key=settings.QDRANT_APIKEY,
                    )

                    uri = settings.QDRANT_CLOUD_URL
                else:
                    cls._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT,
                    )

                    uri = f"{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"

                logger.info(f"Connection to Qdrant DB with URI successful: {uri}")
            except UnexpectedResponse:
                logger.exception(
                    "Couldn't connect to Qdrant.",
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                    url=settings.QDRANT_CLOUD_URL,
                )

                raise

        return cls._instance


connection = QdrantDatabaseConnector()
```

**Key Concepts**:
- **One flag switches environments**: `USE_QDRANT_CLOUD=false` → local Docker at `localhost:6333`; `true` → Qdrant Cloud with `QDRANT_APIKEY`.
- Local Qdrant needs no auth; the Docker network exposes ports 6333 (HTTP/REST) and 6334 (gRPC).
- The connector is reused by `VectorBaseDocument` through `connection.search`, `connection.scroll`, `connection.upsert`, `connection.create_collection`.

**Local vs cloud comparison**:

| Concern | Local (`USE_QDRANT_CLOUD=false`) | Cloud (`USE_QDRANT_CLOUD=true`) |
|---------|----------------------------------|----------------------------------|
| Host | `QDRANT_DATABASE_HOST` (`localhost`) | `QDRANT_CLOUD_URL` |
| Port | `QDRANT_DATABASE_PORT` (`6333`) | implied by URL (HTTPS) |
| Auth | none | `QDRANT_APIKEY` |
| Persistence | Docker volume `qdrant_data` | managed by Qdrant |
| Failure mode | connection refused | auth/URL error |

---

### 4. `llm_engineering/infrastructure/files_io.py` - JSON Artifacts

**Purpose**: A small, explicit reader/writer for the JSON files under `data/artifacts/`.

```python
# llm_engineering/infrastructure/files_io.py  (verbatim)
import json
from pathlib import Path


class JsonFileManager:
    @classmethod
    def read(cls, filename: str | Path) -> list:
        file_path: Path = Path(filename)

        try:
            with file_path.open("r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path=}' does not exist.") from None
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                msg=f"File '{file_path=}' is not properly formatted as JSON.",
                doc=e.doc,
                pos=e.pos,
            ) from None

    @classmethod
    def write(cls, filename: str | Path, data: list | dict) -> Path:
        file_path: Path = Path(filename)
        file_path = file_path.resolve().absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

        return file_path
```

**Key Concepts**:
- **Error messages carry the path** (`{file_path=}`), so failures point at the exact file.
- **`write` creates parent directories** (`mkdir(parents=True, exist_ok=True)`), which is why `data/artifacts/*.json` can be written on first run.
- Used by the export steps (`steps/export/to_json.py`) and the training/inference scripts that bridge ZenML artifacts and Hugging Face datasets.
- **`read` is annotated to return `list`** but `json.load` can return a dict. The annotation is a convention, not enforced. If you store a dict, the caller must not assume a list.
- **No atomic write**: a crash mid-write can leave a truncated JSON file. For critical artifacts, write to a temp file then rename.

---

### 5. `docker-compose.yml` - Local Database Stack

**Purpose**: Bring up MongoDB and Qdrant locally with one command.

```yaml
# docker-compose.yml  (verbatim)
services:
  mongo:
    image: mongo:latest
    container_name: "llm_engineering_mongo"
    logging:
      options:
        max-size: 1g
    environment:
      MONGO_INITDB_ROOT_USERNAME: "llm_engineering"
      MONGO_INITDB_ROOT_PASSWORD: "llm_engineering"
    ports:
      - 27017:27017
    volumes:
      - mongo_data:/data/db
    networks:
      - local
    restart: always

  qdrant:
    image: qdrant/qdrant:latest
    container_name: "llm_engineering_qdrant"
    ports:
      - 6333:6333
      - 6334:6334
    expose:
      - 6333
      - 6334
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - local
    restart: always

volumes:
  mongo_data:
  qdrant_data:

networks:
  local:
    driver: bridge
```

**Key Concepts**:
- **Named volumes** (`mongo_data`, `qdrant_data`) persist data across `docker compose down` (without `-v`).
- MongoDB uses `mongo:latest` with root credentials matching the default `DATABASE_HOST`.
- Qdrant has no credentials locally, matching `USE_QDRANT_CLOUD=false`.
- **`logging.max-size: 1g`** caps Mongo's logs so a long-running local stack does not fill the disk.
- **`expose`** documents the container-internal ports; `ports` publishes them to the host. Both 6333 and 6334 are published, so an external gRPC client can reach Qdrant.
- **`restart: always`** means the containers come back after a host reboot or Docker Desktop restart.

> **Windows/WSL note**: This project runs on a Windows 11 host with Docker Desktop using the WSL2 backend. If `docker compose up` fails to bind ports, ensure WSL2 integration is enabled and that host ports 27017, 6333, and 6334 are not already taken (`netstat -ano | findstr 6333`).

**Port reference**:

| Service | Host port | Container port | Protocol | Purpose |
|---------|-----------|----------------|----------|---------|
| MongoDB | 27017 | 27017 | TCP | Mongo wire protocol |
| Qdrant | 6333 | 6333 | HTTP/REST | client default |
| Qdrant | 6334 | 6334 | gRPC | high-throughput clients |

---

## 🧭 Bootstrapping the Whole Stack (worked sequence)

The infrastructure is only useful once the pieces come up in the right order. Follow this sequence:

```
1. pyenv local 3.11.8              # honor .python-version
2. poetry install --without aws    # install deps, skip SageMaker for now
3. poetry self add 'poethepoet[poetry_plugin]'
4. cp .env.example .env            # then fill credentials
5. poetry poe local-infrastructure-up
       -> docker compose up -d     # Mongo + Qdrant
       -> zenml logout --local     # clear any stale local server
       -> zenml login --local      # start the local ZenML server
6. python -m scripts.check_mongo   # ping Mongo
   python -m scripts.check_qdrant  # list Qdrant collections
7. poetry poe run-digital-data-etl-maxime
```

Step 5 is the project's own `local-infrastructure-up` Poe chain, not a raw `docker compose up`. Steps 6 and 7 are smoke tests before you trust the stack.

### Environment switching matrix

| Intent | `USE_QDRANT_CLOUD` | `DATABASE_HOST` | `QDRANT_CLOUD_URL` | `QDRANT_APIKEY` |
|--------|--------------------|-----------------|--------------------|-----------------|
| All local | `false` | default localhost | ignored | ignored |
| Local Mongo, cloud Qdrant | `true` | default localhost | real URL | real key |
| All cloud | `true` | Atlas/self-hosted URI | real URL | real key |

Only `USE_QDRANT_CLOUD` changes *code paths*. Mongo has no equivalent flag; you switch it purely by changing `DATABASE_HOST`.

### Worked example - connection string anatomy

```
mongodb://llm_engineering:llm_engineering@127.0.0.1:27017
└─scheme─┘ └───user────┘ └───password──┘ └──host──┘ └port┘
```

`settings.DATABASE_NAME` (`"twin"`) is **not** in the URI. The database is selected later by `connection.get_database(settings.DATABASE_NAME)`. Changing the database means changing that setting, not the URI.

---

## 🧪 Tooling Notes

- **Poetry vs uv.** The book and this repo standardize on Poetry (1.8.3) with Poe the Poet tasks. A `uv.lock` also exists in the checkout, reflecting the wider ecosystem's move to uv (Rust, much faster). Use Poetry to follow the book; uv is worth benchmarking separately.
- **ZenML is a Python package, not a compose service.** That is why `docker-compose.yml` has only Mongo and Qdrant, while `local-infrastructure-up` also starts a ZenML server. Do not look for a `zenml` service in the compose file.
- **The `Dockerfile` builds the pipeline image** (`poetry poe build-docker-image`), and `run-docker-end-to-end-data-pipeline` runs the E2E data pipeline inside it with `--network host` so it can reach the local DBs.
- **GPU is irrelevant to this session.** MongoDB and Qdrant are CPU services. The RTX 5000 (Turing sm_75, 16 GB) matters later for embeddings and inference, not for standing up the stores.

---

## 🩺 Troubleshooting Quick Reference

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `pymongo.errors.ServerSelectionTimeoutError` | Mongo container down or wrong URI | `docker ps`, check `DATABASE_HOST` |
| Qdrant `UnexpectedResponse` on first insert | Collection missing with a vector config mismatch | Let `bulk_insert` create it, or recreate after changing the model |
| Empty Qdrant search results | Query vector dimension ≠ collection size | Verify `EmbeddingModelSingleton().embedding_size` |
| Settings edits ignored | A ZenML `settings` secret exists | `poetry poe delete-settings-zenml`, restart |
| Port already in use on `docker compose up` | Another service on 27017/6333/6334 | `netstat -ano \| findstr 6333` and free it |
| Data vanished after `down` | You used `down -v` | Named volumes are deleted by `-v`; back up first |
| `QdrantDatabaseConnector` fails in cloud mode | `QDRANT_CLOUD_URL` still the placeholder `"str"` | Set a real URL in `.env` |

---

## 🔧 Local Setup

### Step 1: Start databases

```bash
docker compose up -d
```

Or the project's wrapper, which also starts ZenML:

```bash
poetry poe local-infrastructure-up
```

That Poe task is a chain: `local-docker-infrastructure-up` (`docker compose up -d`) → `local-zenml-server-down` → `local-zenml-server-up` (`zenml login --local`, blocking on Windows).

### Step 2: Confirm containers

```bash
docker ps --filter "name=llm_engineering"
```

Expected: `llm_engineering_mongo` on `0.0.0.0:27017` and `llm_engineering_qdrant` on `0.0.0.0:6333-6334`.

### Step 3: Create the environment file

```bash
cp .env.example .env
```

Fill in at least `OPENAI_API_KEY`, `HUGGINGFACE_ACCESS_TOKEN`, and `COMET_API_KEY`. The database defaults already work.

---

## 🛠️ Hands-On: Verify Both Connections

### Test MongoDB

```python
# scripts/check_mongo.py
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings

db = connection.get_database(settings.DATABASE_NAME)
print("Mongo ping:", db.command("ping"))
print("Collections:", db.list_collection_names())
```

```bash
python -m scripts.check_mongo
```

### Test Qdrant

```python
# scripts/check_qdrant.py
from llm_engineering.infrastructure.db.qdrant import connection

print("Qdrant collections:", [c.name for c in connection.get_collections().collections])
```

### Test JSON artifacts

```python
from llm_engineering.infrastructure.files_io import JsonFileManager

data = JsonFileManager.read("data/artifacts/cleaned_documents.json")
print(f"Read {len(data)} records")

path = JsonFileManager.write("data/artifacts/_smoke_test.json", {"ok": True})
print("Wrote:", path)
```

### Worked example - verify the connection is a singleton

```python
from llm_engineering.infrastructure.db.mongo import connection as c1
from llm_engineering.infrastructure.db.mongo import MongoDatabaseConnector

c2 = MongoDatabaseConnector()
print(c1 is c2)                     # True - same client
print(type(c2).__name__)            # MongoClient, not MongoDatabaseConnector
print(c2 is c2.get_database("twin").client)   # True
```

Expected output: `True`, `MongoClient`, `True`. If you see a fresh client each call, the singleton is not being used (for example, the module was re-imported under a different path).

---

## 📝 Exercise 1: Add a New Infrastructure Connector

### Task

Add a thin connector for a Redis cache using the same singleton pattern.

```python
# llm_engineering/infrastructure/db/redis.py
import redis
from llm_engineering.settings import settings


class RedisDatabaseConnector:
    _instance: "redis.Redis | None" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = redis.Redis(host="localhost", port=6379, decode_responses=True)
        return cls._instance


connection = RedisDatabaseConnector()
```

Then:
1. Add `REDIS_HOST` / `REDIS_PORT` to `Settings`.
2. Add a `redis` service to `docker-compose.yml`.
3. Write a `check_redis.py` script that sets and reads a key.

**Goal**: Practice the singleton + settings + compose triad the project already uses.

---

## 📝 Exercise 2: Diagnose a Silent Connection Failure

### Task

Reproduce and explain the two most common "it looks connected but nothing works" scenarios.

**Scenario A - Mongo is down but import succeeds.**

```python
from llm_engineering.infrastructure.db.mongo import connection
print("constructed")                     # prints even with Mongo down
try:
    connection.get_database("twin").command("ping")
except Exception as e:
    print("failed on first operation:", type(e).__name__)
```

Explain why the `try/except ConnectionFailure` in `mongo.py` did not fire.

**Scenario B - Qdrant cloud flag set without a URL.**

```python
import os
# simulate: USE_QDRANT_CLOUD=true in .env, QDRANT_CLOUD_URL left as default "str"
# what happens when QdrantDatabaseConnector() runs?
```

Explain which setting is the placeholder and what error you expect.

**Deliverable**: a short note for each scenario naming the file/line that behaves unexpectedly, plus the exact `.env` change to fix it.

**Goal**: internalize that lazy clients defer connection errors to first use, and that `QDRANT_CLOUD_URL`'s default is a placeholder.

---

## ⚠️ Common Pitfalls

1. **`connection` is a client, not a connector.** In both `mongo.py` and `qdrant.py`, `connection = SomeConnector()` evaluates to the *client* returned by `__new__`. Calling `connection.get_database(...)` is correct; don't look for `MongoDatabaseConnector` methods.
2. **Import-time connection side effect.** `domain/base/nosql.py` runs `connection.get_database(settings.DATABASE_NAME)` at import. Importing the domain layer therefore requires importable settings. Tests should mock `connection` or set `DATABASE_NAME`.
3. **The Mongo `try/except ConnectionFailure` rarely fires.** PyMongo connects lazily, so a down server surfaces on the first query, not at construction.
4. **ZenML secret beats `.env` silently.** If a `settings` secret exists, `.env` edits are ignored. Delete it with `poetry poe delete-settings-zenml` when debugging.
5. **`QDRANT_CLOUD_URL` default is `"str"`.** Flipping `USE_QDRANT_CLOUD=true` without a real URL fails at connect time.
6. **Qdrant local has no auth.** Do not set `QDRANT_APIKEY` expecting it to be used locally; the local branch ignores it.
7. **`JsonFileManager.read` can return a dict.** The return type says `list`, but `json.load` returns whatever is in the file. Guard for dicts.
8. **`JsonFileManager.write` is not atomic.** A crash can truncate the file. For critical artifacts, write-then-rename.
9. **`docker compose down` keeps data; `-v` deletes it.** Named volumes survive a plain `down`. `docker compose down -v` wipes MongoDB and Qdrant.
10. **`restart: always` can mask a crash loop.** If Mongo fails to start, it retries forever; check `docker logs llm_engineering_mongo`.
11. **Port conflicts.** 27017/6333/6334 may already be in use by another local service. Check before assuming the compose file is wrong.
12. **`settings` is evaluated once at import.** Changing `.env` after the process starts has no effect; restart Python.
13. **`RAG_MODEL_DEVICE` defaults to `"cpu"`.** On the RTX 5000 (Turing sm_75, 16 GB), the embedding and cross-encoder models run on CPU by default, which is fine for ingestion. Set it to `"cuda"` only if you have verified the CUDA build; Turing has no bfloat16, so keep fp16 when moving models to GPU.
14. **The `settings` singleton is shared with the embedding singleton.** `VectorBaseDocument._create_collection` constructs `EmbeddingModelSingleton()` to read `embedding_size`, which loads the sentence-transformer model. This can be slow on first collection creation and needs the HF cache or network access.
15. **Windows: `zenml login --local` uses `--blocking`.** The Poe task branches per platform (`win32` uses `--blocking`). If you launch ZenML manually, match that behavior.

---

## 🔬 Deep Dive: Why `__new__` Instead of `SingletonMeta`

The repo uses two different singleton mechanisms, and the difference is deliberate:

| Mechanism | Where | Returns | Thread-safe? | Fits |
|-----------|-------|---------|--------------|------|
| `__new__` override | `db/mongo.py`, `db/qdrant.py` | the raw client (`MongoClient`, `QdrantClient`) | relies on the client's own thread safety | connection factories |
| `SingletonMeta` | `application/networks/base.py` | an instance of the class itself | yes (explicit `Lock`) | model wrappers |

Why not use `SingletonMeta` for the clients too? Because `MongoDatabaseConnector()` is meant to *be* a client, not to expose a connector API. The caller wants `connection.get_database(...)`, not `connection.client.get_database(...)`. Overriding `__new__` to return the client directly keeps call sites short. `SingletonMeta` additionally guards against a first-access race with an explicit lock, which matters when multiple threads may instantiate an embedding model at once.

The tradeoff: `__new__` does **not** lock. Two threads calling `MongoDatabaseConnector()` simultaneously at startup could both see `_instance is None`. In practice PyMongo's own construction is cheap and the connector is usually imported once at module load, so the race is benign. If you add a connector that is expensive to construct and may be hit concurrently, prefer `SingletonMeta`.

### Worked example: how many connections do we actually open?

`MongoClient` is a connection *pool*, not a single socket. Calling the constructor once and reusing `connection` means every document class shares one pool.

```
UserDocument.save()      ─┐
ArticleDocument.bulk_find()├─► connection (one MongoClient) ─► pool ─► mongod
PostDocument.bulk_insert()┘
```

If instead each document created its own `MongoClient`, you would open a new pool per entity type. The singleton prevents that. Qdrant behaves the same way: one `QdrantClient` serves `search`, `scroll`, `upsert`, and `create_collection` for every collection.

### Edge case: reloading modules

If code re-imports `llm_engineering.infrastructure.db.mongo` under a different module path (for example via a symlink or a duplicated package on `sys.path`), Python treats it as a new module with a fresh `_instance`. The "singleton" is per module object, not per process. This is rare but explains confusing "multiple clients" behavior in notebooks that manipulate `sys.path`.

---

## 🎓 Knowledge Check

1. **Why is the connector a singleton, and where is the instance stored?**
   - Answer: To reuse one connection pool; stored on the class as `_instance`, returned by `__new__`.

2. **What is `connection` in `mongo.py` - a connector or a client?**
   - Answer: A `MongoClient`; the class constructor returns the client, not an instance of the class.

3. **How does the code switch between local and Qdrant Cloud?**
   - Answer: Via the `USE_QDRANT_CLOUD` boolean in `Settings`.

4. **Which settings source wins if ZenML has a `settings` secret?**
   - Answer: The ZenML secret store; `.env` and defaults are the fallback.

5. **What does `JsonFileManager.write` do that `json.dump` alone does not?**
   - Answer: Resolves the path and creates parent directories before writing.

6. **Does dropping the compose stack delete data?**
   - Answer: No, named volumes persist; only `docker compose down -v` removes them.

7. **What is the default value of `QDRANT_CLOUD_URL`, and why is it dangerous?**
   - Answer: The literal `"str"`; it is a placeholder and fails if you enable cloud mode without overriding it.

8. **Why might `MongoClient` not raise `ConnectionFailure` at construction?**
   - Answer: PyMongo connects lazily; errors surface on the first DB operation.

9. **Which ports does Qdrant expose locally, and what are they for?**
   - Answer: 6333 (HTTP/REST, client default) and 6334 (gRPC).

10. **Why is `_database` defined at module import in `nosql.py`?**
    - Answer: So every document class shares one database handle; the tradeoff is an import-time dependency on settings and a client.

11. **What does `settings.export()` enable?**
    - Answer: Pushing current settings into the ZenML secret store so remote runs need no `.env`.

12. **How do you verify both stores quickly?**
    - Answer: Mongo `db.command("ping")` plus `list_collection_names()`; Qdrant `connection.get_collections()`.

13. **Where does the Qdrant collection dimension come from?**
    - Answer: `EmbeddingModelSingleton().embedding_size` (384 for `all-MiniLM-L6-v2`), not a literal.

14. **What happens if you import the domain layer with an unreachable Mongo?**
    - Answer: Import may succeed (lazy client), but the first DB call fails; the import-time `get_database` itself does not connect.

15. **Should the embedding model run on the RTX 5000 for ingestion?**
    - Answer: By default no (`RAG_MODEL_DEVICE="cpu"`). CPU is adequate for ingestion; use CUDA only with a verified build and fp16, since Turing lacks bfloat16.

---

## 📖 Glossary

- **Singleton**: a class that yields one shared instance; here via `__new__` for clients, `SingletonMeta` for models.
- **Docker Compose**: declarative multi-container orchestration; here MongoDB + Qdrant.
- **Named volume**: Docker-managed persistent storage that survives `down` but not `down -v`.
- **Lazy client**: a client that defers the actual connection until first use.
- **Wire protocol**: Mongo's binary protocol on port 27017.
- **REST / gRPC**: Qdrant's two client transports, ports 6333 and 6334.
- **pydantic-settings**: loads and validates `Settings` from `.env` and environment variables.
- **ZenML secret store**: centralized secret/config storage; overrides `.env` when a `settings` secret exists.
- **Placeholder default**: a value (like `QDRANT_CLOUD_URL="str"`) that must be replaced before the feature it guards is used.
- **`.env`**: local, uncommitted credentials file; `.env.example` is the template.
- **`_instance`**: the class attribute holding the shared client; set once by `__new__`.
- **Connection pool**: a set of reusable sockets managed by the client; one pool per client.
- **Import-time side effect**: code that runs when a module is first imported, such as `_database = connection.get_database(...)`.
- **Placeholder guard**: a sentinel default (here `QDRANT_CLOUD_URL="str"`) that signals "must be configured" without crashing local runs.
- **Fail-soft read**: a helper that logs and returns an empty result (`None`/`[]`) instead of raising, leaving the decision to the caller.
- **Secret precedence**: the resolution order ZenML secret > `.env` > typed default.
- **Singleton per module**: the instance is shared per imported module object, not per process.
- **Structured setting**: one typed Pydantic field, validated at startup rather than at use.
- **Lazy connect**: deferring the network handshake until the first real operation.
- **Vector size**: the fixed dimension of a Qdrant collection, read from the embedding model.

---

## 🔗 Next Session

**Session 2.1**: [Web Crawling with Selenium](./session_2.1_web_crawling.md)

We move up into the application layer and build the data collection crawlers.

---

## 📚 Additional Resources

- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [PyMongo Tutorial](https://pymongo.readthedocs.io/en/stable/tutorial.html)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/)
- [ZenML Secrets](https://docs.zenml.io/user-guide/advanced-guide/secret-management)
- [Docker Compose file reference](https://docs.docker.com/compose/compose-file/)
- [Curriculum map](../CURRICULUM.md)

---

## 📎 References

- Iusztin, P., & Labonne, M. *LLM Engineer's Handbook*. Packt. Chapter 2 "Tooling and Installation" (pages 54-82).
- Repository source files read for this session: `llm_engineering/settings.py`, `llm_engineering/infrastructure/db/mongo.py`, `llm_engineering/infrastructure/db/qdrant.py`, `llm_engineering/infrastructure/files_io.py`, `llm_engineering/domain/base/nosql.py`, `llm_engineering/domain/base/vector.py`, `llm_engineering/application/networks/embeddings.py`, `docker-compose.yml`, `pyproject.toml`.

---

**Estimated Time**: 2-3 hours

**Prerequisites**: Sessions 1.1, 1.2

**Outcome**: You can stand up the local database stack, verify both stores, add new connectors using the project's conventions, and diagnose lazy-connection and environment-switch failures.
