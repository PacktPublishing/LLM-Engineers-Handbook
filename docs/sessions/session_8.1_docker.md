# Session 8.1: Docker & Local Infrastructure

## 🎯 Learning Objectives

By the end of this session, you will:
- Read the production `Dockerfile` and explain **why** it installs Chrome and layers its instructions the way it does
- Run the local MongoDB + Qdrant stack with Docker Compose
- Build and run the application image for `linux/amd64`
- Use Poe the Poet tasks to drive infrastructure and pipelines
- Understand the two-layer local topology (services vs application)
- Reason about image size, layer caching, and the `--network host` tradeoff
- Diagnose the common failure modes of containerized crawling and database connectivity
- Place the local workflow on an RTX 5000 / WSL2 / Windows workstation

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        Local development                            │
│                                                                      │
│  docker compose (services)                                          │
│    ├── mongo    :27017   (llm_engineering / llm_engineering)        │
│    └── qdrant   :6333/6334                                          │
│                                                                      │
│  application container  (optional)                                  │
│    └── llmtwin image  ──docker run --network host──►  services      │
│                                                                      │
│  ZenML server (local)  :8237                                        │
└────────────────────────────────────────────────────────────────────┘
```

Two distinct things run in Docker: **infrastructure services** (long-lived databases) and an optional **application image** (the code, run per pipeline invocation).

### Why two layers?

| Layer | Lifetime | State | Scaling model | Rebuild frequency |
|-------|----------|-------|---------------|-------------------|
| Infrastructure services (Mongo, Qdrant) | Long-lived, `restart: always` | Stateful, named volumes | Horizontal in cloud | Rarely |
| Application container | Per pipeline run, `--rm` | Stateless (reads env, writes to services) | Vertical, one per job | Every code change |

The design principle is **separation of state from compute**. The databases own the volumes; the application image is disposable. This lets you kill and rebuild the app container freely without losing a single document or vector. In the cloud (Chapter 11, pages 444-461) the same split maps onto managed MongoDB/Qdrant serverless plus an ECR image that SageMaker pulls per step.

### The three environments the same code must run in

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  local host  │     │ docker run   │     │  SageMaker job   │
│  .venv       │     │ llmtwin      │     │  (ECR image)     │
│  localhost   │     │ --network    │     │  cloud Mongo +   │
│  Mongo/Qdrant│     │   host       │     │  Qdrant + S3     │
└──────────────┘     └──────────────┘     └──────────────────┘
         all driven by the SAME python -m tools.run commands
```

The Poe tasks are the abstraction that hides which environment you are in. `poetry poe run-end-to-end-data-pipeline` runs on your laptop; `poetry poe run-docker-end-to-end-data-pipeline` runs the same code inside the image; the CI/CD `parent_image` (Session 8.2) runs it on SageMaker.

---

## 📁 Key Files Explained

### 1. `Dockerfile` - The Application Image

The real file, at the repo root, is 47 lines. Read it in full:

```dockerfile
# Dockerfile
FROM python:3.11-slim-bullseye AS release

ENV WORKSPACE_ROOT=/app/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.8.3
ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_NO_INTERACTION=1

# Install Google Chrome
RUN apt-get update -y && \
    apt-get install -y gnupg wget curl --no-install-recommends && \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-linux-signing-key.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/google-linux-signing-key.gpg] https://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update -y && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Install other system dependencies.
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends build-essential \
    gcc \
    python3-dev \
    build-essential \
    libglib2.0-dev \
    libnss3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using pip and clear cache
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"
RUN poetry config installer.max-workers 20

WORKDIR $WORKSPACE_ROOT

# Copy the poetry lock file and pyproject.toml file to install dependencies
COPY pyproject.toml poetry.lock $WORKSPACE_ROOT

# Install the dependencies and clear cache
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-cache --without dev && \
    poetry self add 'poethepoet[poetry_plugin]' && \
    rm -rf ~/.cache/pypoetry/cache/ && \
    rm -rf ~/.cache/pypoetry/artifacts/

# Copy the rest of the code.
COPY . $WORKSPACE_ROOT
```

**Key Concepts**:

- **Chrome is required** because the crawlers use Selenium (Session 2.1). The Google-signed apt repo installs `google-chrome-stable` so `chromedriver-autoinstaller` can pair a matching driver at runtime. The crawler dispatcher registers LinkedIn, Medium, and GitHub crawlers, all Selenium-driven.
- **`PYTHONDONTWRITEBYTECODE=1`** stops `.pyc` files from being written. Inside a container that is recreated on every run, bytecode caches are pure waste; they also bloat the final image and can create confusing stale-`__pycache__` behavior.
- **`PYTHONUNBUFFERED=1`** forces stdout/stderr to be unbuffered. Without it, `loguru` and `print` output can be trapped in the Python buffer and lost when a container exits, so `docker logs` looks empty. This is why container logs "stream in real time".
- **Layer caching**: `pyproject.toml` and `poetry.lock` are copied **before** the source. Docker caches each `RUN`/`COPY` layer; when only source files change, the expensive `poetry install` layer is reused. The book calls this "decouple your installation steps from copying the rest of the files" (page 455).
- **`virtualenvs.create false`**: dependencies install into the system Python at `/usr/local`. Inside a container there is exactly one process tree, so a venv adds a layer of indirection for no isolation benefit.
- **`--without dev`**: no test/lint tools (ruff, pytest, pre-commit) in the production image. Smaller and closer to the runtime surface.
- **`poetry self add 'poethepoet[poetry_plugin]'`**: Poe tasks become available inside the image, so `poetry poe ...` works in the container and the CI runner uses the identical command surface.
- **`rm -rf ~/.cache/pypoetry/...`**: trims the Poetry cache and artifacts after install. Every megabyte saved here is a megabyte not pushed to ECR on each CD run.

> **Accuracy note — the duplicate `build-essential`.** The real `Dockerfile` list on lines 21-28 contains `build-essential` **twice**: once at the top of the list and again after `python3-dev`. This is harmless (`apt-get` deduplicates) but is a real quirk of the file. If you are tightening the image, deleting the second occurrence is a safe, zero-risk edit. The existing prose in some docs omits the duplicate; the file is the source of truth.

**What each system dependency is for**:

| Package | Why it is needed |
|---------|------------------|
| `gnupg`, `wget`, `curl` | Fetch and verify the Google signing key; add the Chrome apt repo |
| `google-chrome-stable` | Selenium crawler browser |
| `build-essential`, `gcc` | Compile wheels that have no prebuilt manylinux binary (e.g. some C extensions) |
| `python3-dev` | Python C headers for building C extensions |
| `libglib2.0-dev`, `libnss3-dev` | Shared libraries Chrome/Chromium links against headless |

Chrome is installed first, before the build tools, because it changes least often; putting the most stable layer earliest maximizes the number of downstream layers that stay cached.

**The layer-cache mental model**:

```
FROM python:3.11-slim-bullseye  ── immutable base ──────────────┐
ENV ...                          ── cached ──────────────────┐  │
RUN install Chrome               ── cached (rarely changes)  │  │
RUN install build tools          ── cached                   │  │
RUN pip install poetry           ── cached                   │  │
COPY pyproject.toml poetry.lock  ── invalidated on dep change│  │
RUN poetry install               ── re-run ONLY if lock changed│ │
COPY . /app/                     ── invalidated on ANY code edit│
```

Because the source is copied last, editing `steps/feature_engineering/clean.py` re-runs only the final `COPY`. That turns a five-minute rebuild into a two-second rebuild.

**Multi-arch note**: The CD workflow builds with `--platform linux/amd64` (Session 8.2) because SageMaker containers are amd64. On Apple Silicon (or any ARM host) you must build for amd64 explicitly, and Docker uses QEMU emulation, which is slow. The book explains the constraint directly: "We must build it on a Linux platform as the Google Chrome installer we used inside Docker works only on a Linux machine" (page 455).

On this workstation (Windows 11 + WSL2 + Docker Desktop with the WSL2 GPU backend) the same rule holds. Build amd64 for parity with SageMaker; build natively only if you want a fast local iteration loop and never push that tag.

---

### 2. `docker-compose.yml` - Local Services

The real file is 40 lines:

```yaml
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

- **No application service** in Compose. The databases run as services; the app runs via `docker run` (or the local `.venv`) and connects over the host network. This keeps the app image free of database coupling and lets you rebuild it constantly.
- **Credentials** match the default `DATABASE_HOST` in `settings.py` (`llm_engineering:llm_engineering`). They are hard-coded *only for local development*; the cloud path uses `DATABASE_HOST=mongodb+srv://...` from `.env` (Chapter 11, page 448).
- **Qdrant exposes 6333 (REST) and 6334 (gRPC)**; the client uses 6333. The `expose` block documents the ports for the container network without publishing new ones.
- **`restart: always`** keeps databases up across reboots and crashes. For a stateful service you almost always want this.
- **Named volumes** (`mongo_data`, `qdrant_data`) are what make the stack stateful. `docker compose stop` preserves them; `docker compose down -v` deletes them.
- **`logging.options.max-size: 1g`** on Mongo caps the json-file log at 1 GB. Without a cap, a long-lived Mongo container can fill the host disk with logs. Qdrant, oddly, has no cap here; if you run the stack for weeks, add one.

**Compose networking, in one picture**:

```
host (Windows/WSL2)
  │  localhost:27017 ───┐
  │  localhost:6333  ──┤   published ports
  ▼                     ▼
┌─────────────┐   ┌──────────────┐
│  mongo      │   │  qdrant      │   ← both on the `local` bridge network
│  :27017     │   │  :6333/:6334 │
│  mongo_data │   │  qdrant_data │   ← named volumes (state lives here)
└─────────────┘   └──────────────┘
```

A container on the bridge network can reach these by **service name** (`mongo`, `qdrant`), while the host reaches them by `localhost:<published-port>`. The application image uses `--network host`, which collapses this distinction and makes `localhost` work inside the container exactly as it does on the host.

---

### 3. Poe tasks - Docker Orchestration

```toml
# pyproject.toml (excerpt, lines 128-131)
build-docker-image = "docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile ."
run-docker-end-to-end-data-pipeline = "docker run --rm --network host --shm-size=2g --env-file .env llmtwin poetry poe --no-cache --run-end-to-end-data"
bash-docker-container = "docker run --rm -it --network host --env-file .env llmtwin bash"

local-docker-infrastructure-up = "docker compose up -d"
local-docker-infrastructure-down = "docker compose stop"
```

**Key Concepts**:

- **`--network host`** lets the container reach `mongo`/`qdrant` on `localhost` exactly as the local code expects. This is a Linux/WSL2 idiom. On Docker Desktop for Windows, `--network host` does not give you the Linux host network; use the host's reachable address or run the databases and app on the same bridge network instead.
- **`--shm-size=2g`** raises `/dev/shm`. Chrome/headless Chromium allocates shared memory for its renderer processes; the Docker default of 64 MB is too small and causes the browser to crash with `DevToolsActivePort` or `session deleted because of page crash` errors. This flag is the fix.
- **`--env-file .env`** injects configuration and secrets at runtime without baking them into the image. Secrets never enter an image layer, which keeps them out of `docker history` and ECR.
- **`--rm`** deletes the container after it exits, keeping the host clean. State lives in the services, not the container.
- **`poetry poe --no-cache --run-end-to-end-data`** runs the whole data pipeline inside the container. `--no-cache` is passed through to `tools.run`, disabling ZenML step caching for a fresh run.
- **`bash-docker-container`** is the debugging escape hatch: get a shell in the exact image, then run one task at a time.

**The full local-task surface** (from `pyproject.toml`):

| Task | Command | Purpose |
|------|---------|---------|
| `build-docker-image` | `docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile .` | Build the app image |
| `local-docker-infrastructure-up` | `docker compose up -d` | Start Mongo + Qdrant detached |
| `local-docker-infrastructure-down` | `docker compose stop` | Stop (keep volumes) |
| `local-zenml-server-up` | `zenml login --local` (win32: `--blocking`) | Start local ZenML server |
| `local-zenml-server-down` | `zenml logout --local` | Disconnect |
| `run-docker-end-to-end-data-pipeline` | `docker run ...` | Run pipeline in-container |
| `bash-docker-container` | `docker run -it ... bash` | Interactive shell |

> **Windows note.** The `local-zenml-server-up` task is a Poe *switch* on `sys.platform`; on `win32` it runs `zenml login --local --blocking` because Windows needs the server process to hold the terminal. This is why `poetry poe local-infrastructure-up` behaves differently on your workstation than on macOS/Linux.

---

## 🛠️ Hands-On: Build and Run Locally

### Step 1: Start the databases

```bash
docker compose up -d
docker ps --filter name=llm_engineering
```

Expected: two containers, `llm_engineering_mongo` and `llm_engineering_qdrant`, both `Up`.

### Step 2: Verify connectivity from Python

```bash
python -c "from llm_engineering.infrastructure.db.mongo import connection; print(connection.get_database('twin').command('ping'))"
python -c "from llm_engineering.infrastructure.db.qdrant import connection; print([c.name for c in connection.get_collections().collections])"
```

The Mongo `ping` returns `{'ok': 1.0}`. The Qdrant call returns a (possibly empty) list of collection names. If Mongo asks for auth, your `DATABASE_HOST` in `.env` is missing the `llm_engineering:llm_engineering@` credentials.

### Step 3: Build the image

```bash
docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile .
# or:
poetry poe build-docker-image
```

Watch the layer cache: the first build is slow (Chrome + wheels). Edit a step file and rebuild; only the final `COPY` runs.

### Step 4: Open a shell in the image

```bash
docker run --rm -it --network host --env-file .env llmtwin bash
# inside:
poetry poe --no-cache --run-feature-engineering
```

### Step 5: Tear down services

```bash
docker compose stop       # keeps volumes
# docker compose down -v  # also deletes data (dangerous)
```

### Step 6: Measure the image and the layers

```bash
docker images llmtwin --format "{{.Size}}"
docker history llmtwin --no-trunc | head -20
```

Use this to justify each `rm -rf` cleanup. If Chrome is not the largest layer, something unexpected is in the image (usually a wheel or a stray `COPY`).

### Step 7: Follow the databases' logs

```bash
docker compose logs -f mongo
docker compose logs -f qdrant
```

The Mongo log confirms `Waiting for connections`; if it shows authentication failures, the app's `DATABASE_HOST` and the `MONGO_INITDB_*` values disagree. The Qdrant log confirms the REST and gRPC listeners on 6333 and 6334.

---

## 📝 Exercise: Add a Redis Service

### Task

Extend the local stack with a cache.

1. Add a `redis` service to `docker-compose.yml` (image `redis:7`, port 6379) with a named volume and `restart: always`.
2. Add `REDIS_HOST` / `REDIS_PORT` to `Settings` (see `llm_engineering/settings.py`).
3. Add a Poe task `local-infrastructure-status = "docker compose ps"`.
4. Verify a Python `redis` client can `PING`.

**Goal**: Practice extending both Compose and the settings layer consistently, and see how the same credential/host pattern the databases use applies to any new service.

---

## 📝 Exercise 2: Shrink the Image and Prove It

### Task

The image is large because of Chrome and the build tools. Make it smaller without breaking the crawlers.

1. Record the baseline: `docker images llmtwin --format "{{.Size}}"`.
2. Remove the duplicate `build-essential` from the system-dependency `RUN` block.
3. Combine the two `apt-get update ... rm -rf /var/lib/apt/lists/*` blocks so the apt lists are downloaded once, not twice.
4. Add `--no-install-recommends` consistently and confirm Chrome still launches.
5. Rebuild and compare sizes; open a shell and run one crawler step to prove Selenium still works.

**Goal**: Internalize that image size is a sequence of deliberate, testable decisions, not an accident. Never ship a size claim you have not measured with `docker images`.

---

## 🔬 Deep Dive: What Actually Runs Where

```
┌────────────────────────────────────────────────────────────────────┐
│ .env (host)                                                        │
│   DATABASE_HOST, QDRANT_*, HF_TOKEN, ...                           │
└──────────────┬───────────────────────────────────────────────────┘
               │ --env-file
               ▼
┌────────────────────────┐        ┌─────────────────────────────────┐
│ llmtwin container      │        │ mongo container :27017          │
│  poetry poe ...        │ ─────► │  llm_engineering / ...          │
│  --network host        │        └─────────────────────────────────┘
│  localhost:27017       │        ┌─────────────────────────────────┐
│  localhost:6333        │ ─────► │ qdrant container :6333/:6334    │
└────────────────────────┘        └─────────────────────────────────┘
               │
               ▼
┌────────────────────────┐
│ ZenML server :8237     │  ← artifacts/metadata; local by default,
│  (host process)        │    or the cloud tenant in Chapter 11
└────────────────────────┘
```

Three observations that matter for debugging:

1. **The container has no DNS entry for `mongo`** when using `--network host`. That is fine because the code uses `localhost`. If you switch to a bridge network, you must change `DATABASE_HOST` to `mongo:27017`.
2. **The ZenML server is not a Compose service.** It runs on the host (or in the cloud). The container only reaches it if `docker run` includes `--network host`; otherwise remote metadata logging fails.
3. **The `.env` file is shared** between host and container. This is convenient but means a cloud `DATABASE_HOST` left in `.env` will make the local container try to reach MongoDB Atlas. Keep a `.env.local` if you switch often.

---

## 🔬 Deep Dive: WSL2, Docker Desktop, and the RTX 5000

This workstation is Windows 11 with WSL2 (Ubuntu 24.04), Docker Desktop using the WSL2 backend, an RTX 5000 (Turing, sm_75, 16 GB), and `uv` for Python. None of that changes the `Dockerfile`, but it changes how you run the container locally.

**Facts about this environment:**

- Docker Desktop runs containers inside a WSL2 distribution, not on Windows directly. The "Linux host" that `--network host` refers to is that WSL2 VM.
- The RTX 5000 is visible to containers only if the NVIDIA Container Toolkit (bundled with recent Docker Desktop) is enabled. `docker run --gpus all ...` then exposes the GPU.
- Turing sm_75 means: FP16 yes, BF16 no, FlashAttention-2 no (see Session 8.4).
- The `.wslconfig` caps WSL2 at 16 GB RAM, 6 processors, 4 GB swap. Building a Chrome image and running it both inside a 16 GB WSL2 VM can contend for memory.

**Consequences for this session:**

| Concern | What to do here |
|---------|-----------------|
| `--network host` | On Docker Desktop it does not give the Linux host network. Run databases in Compose and connect via published `localhost` ports, or put the app on the same bridge network and use service-name DNS (`mongo`, `qdrant`). |
| GPU in container | Only needed for training/inference images, not this app image. If you add one, use `--gpus all`. |
| Memory pressure | The app image installs Chrome; a Selenium crawl plus Mongo + Qdrant can push WSL2 near its 16 GB cap. Raise `memory=` in `.wslconfig` if crawls get OOM-killed. |
| Build speed | amd64 builds native on an amd64 machine. On an ARM host Docker would emulate via QEMU and the build would be far slower. |
| File I/O | Keeping the repo inside the WSL2 filesystem (`~/projects/`) is much faster for bind mounts than crossing the `/mnt/c` boundary. |
| Python package manager | `uv` manages the host env; the image still uses Poetry (the pinned `POETRY_VERSION`). They are separate; do not assume `uv` is present inside the image. |

**A safe local default for Docker Desktop:** run only `mongo` and `qdrant` in Compose, and run the application from the host `.venv` (or a `uv run`). Reach the databases at `localhost:27017` and `localhost:6333`. This avoids the `--network host` caveat entirely and keeps GPU access on the host. Use the containerized app path only when you are validating the image itself.

### Reproducibility and pinning

```
Layer that floats        Consequence                    Mitigation
────────────────────     ─────────────────────────      ──────────────────────────
python:3.11-slim-...     minor/patch drift over time     pin by digest
mongo:latest             schema/behavior drift           pin a major (mongo:7)
qdrant/qdrant:latest     API drift                       pin a version
POETRY_VERSION=1.8.3     already pinned (good)           keep
poetry.lock              already locked (good)           commit it
```

The one deliberate non-pin is `latest` on the databases, which is fine for a learning stack and wrong for production. The rule of thumb: pin anything whose change would force you to rewrite code; allow drift only for things you re-test every run.

---

## 🔬 Deep Dive: The Settings Contract Between Compose, `.env`, and the Code

The container, the host process, and the cloud job all read the same `Settings` object. The `.env` file is the contract that makes that true. Understanding the resolution order removes most "works locally, fails in Docker" surprises.

```
Resolution order (first hit wins):
  1. Environment variables already set in the process
  2. The file named by ENV_FILE (default .env)
  3. Field defaults declared on the Settings model

Compose service env (MONGO_INITDB_*) ──► the mongo container only
.env DATABASE_HOST                  ──► the app (host, container, or cloud job)
```

Key points:

- **The Mongo init variables and the app's connection string are different things.** `MONGO_INITDB_ROOT_USERNAME/PASSWORD` only create the root user on first start. The app connects using `DATABASE_HOST`, which must embed the same credentials: `mongodb://llm_engineering:llm_engineering@localhost:27017`.
- **`.env` is loaded at process start.** A container started with `--env-file .env` gets those values; change `.env` and you must restart the container. There is no live reload.
- **`ENV_FILE` overrides the filename.** The Poe `test` task sets `ENV_FILE=.env.testing` so tests never touch production values. The container does not set it, so it uses `.env`.
- **The cloud overrides `.env` via ZenML secrets.** After `export-settings-to-zenml`, remote steps read settings from the ZenML secret store; the local `.env` is the fallback. This is why forgetting the export step causes cloud runs to use local defaults.
- **Secrets must not be committed.** `.gitignore` ignores `.env`. The example `.env.example` documents the keys without values. Docker never copies `.env` into the image; it is injected at runtime.

### Decision tree: where should this process run?

```
Do you need to validate the Docker image itself?   ── yes ──► docker run llmtwin ...
        │ no
        ▼
Do you need GPU (training/eval)?                   ── yes ──► host .venv (GPU visible)
        │ no                                                        (or --gpus all)
        ▼
Just running pipelines against local DBs?          ─────────► host .venv with .env
                                                                (simplest, fastest)
```

The default for day-to-day development is the host `.venv`; the container is for image validation and for reproducing the exact production environment.

---

## 🐛 Common Pitfalls

- **Chrome missing at runtime**: a crawler fails with "chromedriver not found". The Dockerfile installs Chrome; if you run outside Docker, install Chrome locally and let `chromedriver-autoinstaller` fetch a matching driver.
- **`--network host` on Windows/Docker Desktop**: the flag does not behave like Linux. Prefer running the app on the host with only the databases in Compose, or put both on the same bridge network.
- **`down -v` deletes data**: only use it when you intend to wipe MongoDB and Qdrant. There is no confirmation prompt.
- **Image size**: installing Chrome and build tools makes the image large; that is the cost of crawling in-container. Do not "optimize" by removing Chrome unless you have moved crawling out of the pipeline.
- **Architecture**: building on ARM without `--platform linux/amd64` produces an image SageMaker cannot run. The failure appears late, at deploy time.
- **Uncapped container logs**: without `max-size`, a long-running service can fill the host disk. Mongo has a 1 GB cap here; Qdrant does not.
- **Stale `__pycache__` in the build context**: `PYTHONDONTWRITEBYTECODE=1` prevents new files, but an old `__pycache__` copied by `COPY . /app/` is still copied. `.gitignore` lists `__pycache__/`; use a `.dockerignore` to actually exclude it from the build context.
- **Secrets baked into the image**: if you `COPY .env` or set an `ENV` with a token, it persists in a layer and can be recovered from `docker history`. Always use `--env-file` / secrets managers.
- **`latest` base images**: `python:3.11-slim-bullseye` is pinned to a minor line but the image tag floats. Reproducible builds eventually need a digest pin.
- **Poetry install before `WORKDIR`**: ordering matters. If `WORKDIR` is set after `COPY`, the copy targets the wrong directory and install fails.

---

## 🎓 Knowledge Check

1. **Why does the Dockerfile install Google Chrome?**
   - Answer: The Selenium crawlers (Session 2.1) need a browser and a matching driver.

2. **Why are `pyproject.toml` and `poetry.lock` copied before the source?**
   - Answer: To cache the dependency-install layer so a code-only edit re-runs only the final `COPY`.

3. **What does `--shm-size=2g` prevent?**
   - Answer: Headless-Chrome crashes caused by the default 64 MB `/dev/shm` being too small.

4. **Which service ports does Compose expose for Qdrant?**
   - Answer: 6333 (REST) and 6334 (gRPC).

5. **What is `--without dev` for?**
   - Answer: Excluding dev-only tools (ruff, pytest, pre-commit) from the production image.

6. **How does the app container reach the databases?**
   - Answer: Over the host network (`--network host`) using `localhost` addresses.

7. **Why is the image built for `linux/amd64`?**
   - Answer: SageMaker runs amd64; the Chrome installer is Linux-only; `--platform` also makes ARM hosts emulate amd64.

8. **What is the duplicate `build-essential` and does it break the build?**
   - Answer: It appears twice in the system-dependency list; `apt-get` deduplicates, so it is harmless but redundant.

9. **What keeps your data alive across `docker compose stop`?**
   - Answer: The named volumes `mongo_data` and `qdrant_data`.

10. **What does `PYTHONUNBUFFERED=1` change?**
    - Answer: It makes stdout/stderr unbuffered so container logs stream live and are not lost on exit.

11. **Why does the local ZenML server task differ on Windows?**
    - Answer: The Poe `sys.platform` switch runs `zenml login --local --blocking` on `win32` so the server holds the terminal.

12. **How do secrets reach the container?**
    - Answer: Via `--env-file .env` at runtime, not baked into image layers.

13. **What is the difference between `docker compose stop` and `docker compose down -v`?**
    - Answer: `stop` halts containers and keeps volumes; `down -v` removes containers and deletes volumes (data loss).

14. **Why might `docker logs` be empty after a pipeline run?**
    - Answer: Python buffering, or logs went to the ZenML server rather than stdout; `PYTHONUNBUFFERED=1` addresses the first.

15. **What is the cheapest way to debug a failing pipeline step inside the image?**
    - Answer: `poetry poe bash-docker-container`, then run the individual task by hand.

---

## 📖 Glossary

| Term | Meaning |
|------|---------|
| **Image** | Immutable, layered filesystem + metadata used to create containers. |
| **Container** | A running (or stopped) instance of an image, with its own writable layer. |
| **Layer** | A cached filesystem diff produced by one Dockerfile instruction. |
| **Volume** | Docker-managed persistent storage; survives container removal. |
| **Buildx** | Docker's extended build client; required for `--platform` and multi-arch. |
| **`/dev/shm`** | Shared memory segment; Chrome renderers need more than the 64 MB default. |
| **`--network host`** | Container shares the host network namespace; `localhost` works directly (Linux/WSL2). |
| **`--env-file`** | Reads KEY=VALUE pairs into container environment at runtime. |
| **Selenium** | Browser automation library driving Chrome for the crawlers. |
| **chromedriver** | The WebDriver binary Chrome accepts; version must match Chrome. |
| **`.dockerignore`** | Files excluded from the build context before `COPY`. |
| **Bind mount** | Host path mounted into a container; slow across the Windows/WSL2 boundary. |
| **Digest** | Content hash of an image; the only truly immutable reference. |

---

## 📚 References

- [Docker Compose](https://docs.docker.com/compose/)
- [Dockerfile best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Buildx and multi-platform builds](https://docs.docker.com/build/building/multi-platform/)
- [Poetry](https://python-poetry.org/docs/)
- [Poe the Poet](https://poethepoet.natn.io/)
- [Chrome for Linux](https://www.google.com/chrome/)
- [Selenium with Chrome](https://www.selenium.dev/documentation/webdriver/browsers/chrome/)
- Book: *LLM Engineer's Handbook*, Chapter 11 (pages 444-461) - infrastructure, containerization, ECR push.

---

## 🔗 Next Session

**Session 8.2**: CI/CD with GitHub Actions

We automate lint, tests, secret scanning, and image publication.

---

## 📚 Additional Resources

- [Docker Compose](https://docs.docker.com/compose/)
- [Poetry](https://python-poetry.org/docs/)
- [Poe the Poet](https://poethepoet.natn.io/)
- [Chrome for Linux](https://www.google.com/chrome/)
- Cross-links: [Session 1.3 Infrastructure Layer](session_1.3_infrastructure_layer.md), [Session 2.1 Web Crawling](session_2.1_web_crawling.md), [Session 8.2 CI/CD](session_8.2_cicd.md), [Session 8.3 ZenML](session_8.3_zenml.md)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 1.3, 2.1

**Outcome**: You can build the application image, run the local service stack, explain every Dockerfile decision and its tradeoff, and drive pipelines through Poe tasks.
