# Session 8.3: ZenML Orchestration

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how `@pipeline` and `@step` compose a DAG and how edges arise
- Read the stack concept and switch between local and AWS stacks
- Control runs through YAML configs and Poe tasks
- Use artifacts, caching, and step invocation IDs for chaining
- Read the full `end_to_end_data` pipeline, including nested child pipelines
- Reason about caching correctness, artifact versioning, and `wait_for` ordering
- Diagnose the two signature mismatches between steps and their callers
- Extend the pipeline graph with a new step and register it correctly

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ZenML stack                                    │
│                                                                        │
│  orchestrator        → where steps run (local / SageMaker)            │
│  artifact store      → where outputs live (local / S3)                │
│  container registry  → image for step containers (local / ECR)        │
│  secrets manager     → settings + credentials                         │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │  end_to_end_data (pipeline)                                        │
 │                                                                    │
 │  for each author:                                                  │
 │    digital_data_etl ──► invocation_id ──────────┐                  │
 │                                                  │ wait_for         │
 │  feature_engineering(author_full_names, wait_for) ◄┘                │
 │    query_data_warehouse → clean → load_to_vector_db                │
 │                        → chunk_and_embed → load_to_vector_db       │
 │                                    │ invocation_ids                 │
 │  generate_datasets(test_split_size, ..., wait_for) ◄───────────────┘
 └────────────────────────────────────────────────────────────────────┘
```

Pipelines are Python functions decorated with `@pipeline`; they call `@step` functions. ZenML records every run, artifact, and parameter.

### Why an orchestrator at all?

A script that calls functions in sequence works until it does not: a mid-run crash leaves you unable to tell what completed, what output it produced, or how to rerun only the missing part. ZenML adds four properties that a plain script lacks:

| Property | Without ZenML | With ZenML |
|----------|---------------|------------|
| **Observability** | print logs | dashboard, per-step status and metadata |
| **Caching** | none | content-addressed step reuse |
| **Artifacts** | files on disk | typed, versioned, named outputs |
| **Portability** | hard-coded paths | same code, swap stack (local ↔ AWS) |

The cost is indirection: steps run in containers or subprocesses, state crosses process boundaries, and configuration lives in YAML. This session is about paying that cost deliberately.

### The two extension axes

```
        code (Python)                    config (YAML)              stack (ZenML)
 ┌───────────────────────┐        ┌──────────────────────┐    ┌──────────────────┐
 │ @step def ...          │        │ parameters:          │    │ orchestrator     │
 │ @pipeline def ...      │  +     │ settings:            │ +  │ artifact store   │
 │ pipelines/__init__.py  │        │   docker.parent_image│    │ container registry│
 └───────────────────────┘        └──────────────────────┘    └──────────────────┘
   what runs                        with what inputs            on which infra
```

The same pipeline code runs on a laptop or on SageMaker with **no code change**; only the stack and config change. That is the whole point of the abstraction.

---

## 📁 Key Files Explained

### 1. The Step/Pipeline Primitives

```python
# pipelines/digital_data_etl.py (full file, 11 lines)
from zenml import pipeline

from steps.etl import crawl_links, get_or_create_user


@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    user = get_or_create_user(user_full_name)
    last_step = crawl_links(user=user, links=links)

    return last_step.invocation_id
```

```python
# steps/feature_engineering/clean.py (excerpt)
from typing_extensions import Annotated
from zenml import get_step_context, step


@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    ...
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))
    return cleaned_documents
```

**Key Concepts**:

- **`@pipeline`** defines a DAG. Calling a `@step` inside it adds a node; passing one step's return value to another creates an edge.
- **`Annotated[type, "name"]`** labels step inputs/outputs. The label appears in the ZenML dashboard and names the artifact. `Annotated[list[str], "crawled_links"]` means the artifact is called `crawled_links`.
- **`last_step.invocation_id`** is a handle to a specific step run, used to express `wait_for` dependencies **without passing data**. This is how pipelines compose without forcing a serialized payload through the graph.
- **`get_step_context().add_output_metadata(...)`** attaches structured metadata to a step output for observability. Metadata is cheap; it is stored with the run, not passed between steps.
- **Return values are artifacts.** `crawl_links` returns the original `links` list (not the crawled content, which was written to MongoDB). The return value exists so the pipeline can capture `invocation_id` and so the artifact graph has a node.

### 2. Named Artifacts

```python
# steps/generate_datasets/generate_intruction_dataset.py (excerpt)
@step
def generate_intruction_dataset(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"],
    test_split_size: Annotated[float, "test_split_size"],
    mock: Annotated[bool, "mock_generation"] = False,
) -> Annotated[
    InstructTrainTestSplit,
    ArtifactConfig(
        name="instruct_datasets",
        tags=["dataset", "instruct", "cleaned"],
    ),
]:
    ...
```

**Key Concepts**:

- **`ArtifactConfig(name=...)`** gives the output a stable name (`instruct_datasets`), so `export_artifact_to_json` and training steps can reference it by name rather than by run. The `configs/export_artifact_to_json.yaml` lists exactly these names.
- **`tags=[...]`** make artifacts searchable in the ZenML UI (`dataset`, `instruct`, `cleaned`).
- Note the misspelling **`generate_intruction_dataset`** (missing "s"). It is the real function name, exported through `steps/generate_datasets/__init__.py` and imported in `pipelines/generate_datasets.py`. Do not "fix" it without a rename across all three sites; the artifact name `instruct_datasets` is spelled correctly and is what external code references.

### 3. The Full Data Pipeline

```python
# pipelines/end_to_end_data.py (full file, 33 lines)
from zenml import pipeline

from .digital_data_etl import digital_data_etl
from .feature_engineering import feature_engineering
from .generate_datasets import generate_datasets


@pipeline
def end_to_end_data(
    author_links: list[dict[str, str | list[str]]],
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
) -> None:
    wait_for_ids = []
    for author_data in author_links:
        last_step_invocation_id = digital_data_etl(
            user_full_name=author_data["user_full_name"], links=author_data["links"]
        )

        wait_for_ids.append(last_step_invocation_id)

    author_full_names = [author_data["user_full_name"] for author_data in author_links]
    wait_for_ids = feature_engineering(author_full_names=author_full_names, wait_for=wait_for_ids)

    generate_datasets(
        test_split_size=test_split_size,
        push_to_huggingface=push_to_huggingface,
        dataset_id=dataset_id,
        mock=mock,
        wait_for=wait_for_ids,
    )
```

**Key Concepts**:

- **A pipeline calling another pipeline** nests runs: `digital_data_etl` and `feature_engineering` become **child pipelines** of `end_to_end_data`. The dashboard shows the nesting.
- **`wait_for`** encodes ordering without data dependencies. `digital_data_etl` returns its `crawl_links` invocation id; `feature_engineering` waits on those before querying the warehouse.
- **`wait_for_ids` is reassigned**: the ETL ids are replaced by the feature-engineering ids, which `generate_datasets` consumes. This is deliberate - dataset generation only needs both vector-DB loads to be done, not the raw ETL steps.
- **The loop runs ETL once per author** but feature engineering once for all authors. That is a design choice: crawling is naturally per-author, while cleaning/embedding benefits from a single run.

### 4. Feature Engineering DAG

```python
# pipelines/feature_engineering.py (full file, 16 lines)
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

The DAG fans out after `clean_documents`: one branch stores cleaned documents, the other chunks+embeds then stores. Both branches feed the returned ids.

```
                       ┌──────────────────────┐
   author_full_names ─►│ query_data_warehouse │
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │   clean_documents    │
                       └──────────┬───────────┘
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌──────────────────────┐    ┌──────────────────────┐
        │  load_to_vector_db   │    │   chunk_and_embed    │
        │   (cleaned_documents)│    └──────────┬───────────┘
        └──────────┬───────────┘               ▼
                   │                ┌──────────────────────┐
                   │                │  load_to_vector_db   │
                   │                │  (embedded_documents)│
                   │                └──────────┬───────────┘
                   └────────┬──────────────────┘
                            ▼
                  [last_step_1.id, last_step_2.id]
```

> **Accuracy note — the `after` argument.** In `feature_engineering` the step is invoked as `query_data_warehouse(author_full_names, after=wait_for)`, but the step definition is `def query_data_warehouse(author_full_names: list[str])` (`steps/feature_engineering/query_data_warehouse.py:12-15`) and declares **no `after` parameter**. The same pattern appears in `generate_datasets` where `query_feature_store(after=wait_for)` is called against `def query_feature_store()` (`steps/generate_datasets/query_feature_store.py:17-18`), which also declares no `after` parameter. In ZenML, step execution order can be influenced by an invocation-level `after` argument, so this may be an intentional use of the framework rather than a plain Python bug - but the parameter does **not** appear in either step signature, so it cannot be discovered by reading the step. Treat it as a real inconsistency: if you run these paths and see an unexpected-kwarg error, this is the line to look at; if they run, `after` is being consumed by ZenML's invocation layer. Either way, the step bodies never read `wait_for`, so ordering is the only effect.

### 5. Stack Configuration

Local default stack (MongoDB, Qdrant, local artifact store, local orchestrator):

```toml
# pyproject.toml (lines 99-118)
local-docker-infrastructure-up = "docker compose up -d"
local-docker-infrastructure-down = "docker compose stop"
local-zenml-server-down = "poetry run zenml logout --local"
local-infrastructure-up = [
    "local-docker-infrastructure-up",
    "local-zenml-server-down",
    "local-zenml-server-up",
]
local-infrastructure-down = [
    "local-docker-infrastructure-down",
    "local-zenml-server-down",
]
set-local-stack = "poetry run zenml stack set default"
set-aws-stack = "poetry run zenml stack set aws-stack"
set-asynchronous-runs = "poetry run zenml orchestrator update aws-stack --synchronous=False"
zenml-server-disconnect = "poetry run zenml disconnect"
export-settings-to-zenml = "poetry run python -m tools.run --export-settings"
```

AWS stack (SageMaker orchestrator, S3 artifact store, ECR registry):

```toml
# pyproject.toml
set-aws-stack = "poetry run zenml stack set aws-stack"
set-asynchronous-runs = "poetry run zenml orchestrator update aws-stack --synchronous=False"
export-settings-to-zenml = "poetry run python -m tools.run --export-settings"
delete-settings-zenml = "poetry run zenml secret delete settings"
```

**Key Concepts**:

- **`zenml stack set <name>`** switches the active stack; the same pipeline code runs locally or on SageMaker with no code change.
- **`set-asynchronous-runs`** flips the SageMaker orchestrator to async: the pipeline submits and returns, rather than blocking the CLI. This is what makes large training runs practical (the book notes synchronous runs can time out on long jobs).
- **`export-settings-to-zenml`** pushes `Settings` into the ZenML secret store (`settings.export()` from Session 1.3), so remote steps read the same config. `delete-settings-zenml` removes it.
- **`local-infrastructure-up` is a composite task** - a list of sub-tasks that run in order. This is Poe's orchestration feature; it fails the whole task if any sub-task fails.

### 6. YAML Configs

```yaml
# configs/feature_engineering.yaml (full file, 11 lines)
settings:
  docker:
    parent_image: 992382797823.dkr.ecr.eu-central-1.amazonaws.com/zenml-rlwlcs:latest
    skip_build: True
  orchestrator.sagemaker:
    synchronous: false

parameters:
  author_full_names:
    - Maxime Labonne
    - Paul Iusztin
```

**Key Concepts**:

- **`settings.docker.parent_image`** pins the base image for step containers (the ECR image built by CD in Session 8.2).
- **`skip_build: True`** uses the existing image instead of rebuilding per run. Without it, ZenML tries to build a container from the working directory on every run.
- **`settings.orchestrator.sagemaker.synchronous: false`** enables async submission for SageMaker.
- **`parameters`** are the pipeline's runtime arguments. Editing YAML changes a run without touching code.
- The identical `settings:` block is repeated in **all nine** configs. It is boilerplate; a future refactor could factor it into a shared base config, but ZenML configs do not support YAML anchors cleanly, so the duplication is accepted.

```yaml
# configs/end_to_end_data.yaml (excerpt, 87 lines total)
settings:
  docker:
    parent_image: 992382797823.dkr.ecr.eu-central-1.amazonaws.com/zenml-rlwlcs:latest
    skip_build: True
  orchestrator.sagemaker:
    synchronous: false

parameters:
  author_links:
    - user_full_name: Paul Iusztin
      links:
        - https://medium.com/decodingml/an-end-to-end-framework-...
        # ... 40+ links ...
    - user_full_name: Maxime Labonne
      links:
        # ... 24 links ...
  test_split_size: 0.1
  push_to_huggingface: false
  dataset_id: pauliusztin/llmtwin
  mock: false
```

**Key Concepts**:

- **`author_links` is a list of dicts**, each with a `user_full_name` and `links`. This drives one ETL run per author and one combined feature-engineering run.
- **`dataset_id`** is the Hugging Face repo the dataset is pushed to (when `push_to_huggingface: true`).
- The two per-author ETL configs (`digital_data_etl_paul_iusztin.yaml`, `digital_data_etl_maxime_labonne.yaml`) are **different link lists** from `end_to_end_data.yaml`; they are used when running ETL alone.

**The config matrix** (all nine files share the same `settings:` block):

| Config | Pipeline | Key parameters |
|--------|----------|----------------|
| `digital_data_etl_paul_iusztin.yaml` | `digital_data_etl` | `user_full_name`, `links` |
| `digital_data_etl_maxime_labonne.yaml` | `digital_data_etl` | `user_full_name`, `links` |
| `feature_engineering.yaml` | `feature_engineering` | `author_full_names` |
| `generate_instruct_datasets.yaml` | `generate_datasets` | `dataset_type: instruction`, `test_split_size: 0.1`, `dataset_id: pauliusztin/llmtwin` |
| `generate_preference_datasets.yaml` | `generate_datasets` | `dataset_type: preference`, `test_split_size: 0.05`, `dataset_id: pauliusztin/llmtwin-dpo` |
| `training.yaml` | `training` | `finetuning_type: sft`, `is_dummy: true` |
| `evaluating.yaml` | `evaluating` | `is_dummy: true` |
| `export_artifact_to_json.yaml` | `export_artifact_to_json` | `artifact_names: [...]` |
| `end_to_end_data.yaml` | `end_to_end_data` | `author_links`, `test_split_size`, ... |

Note `dataset_type` is passed as the string `"instruction"` / `"preference"`; the `generate_datasets` pipeline compares it against the `DatasetType` enum (`DatasetType.INSTRUCTION`), so string-to-enum coercion must occur (Pydantic handles this on the typed parameter).

### 7. The CLI Entry Point

```python
# tools/run.py (excerpt, 200 lines)
    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent

    if run_end_to_end_data:
        run_args_end_to_end = {}
        pipeline_args["config_path"] = root_dir / "configs" / "end_to_end_data.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"end_to_end_data_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        end_to_end_data.with_options(**pipeline_args)(**run_args_end_to_end)
```

**Key Concepts**:

- **`--no-cache` maps to `enable_cache=False`** in `pipeline_args`. By default ZenML caches steps whose inputs are unchanged.
- **Config paths are resolved from the repo root** (`Path(__file__).resolve().parent.parent`), so the CLI can run from anywhere.
- **Run names embed a timestamp**, making runs easy to find.
- **`assert` guards** ensure at least one action flag is passed; running with no flags raises `Please specify an action to run.`
- **One command per action flag**: the file has a distinct `if run_*:` block for each pipeline, each setting `config_path` and `run_name`.
- **`export_settings`** is handled before the pipeline blocks: `settings.export()` pushes credentials to the ZenML secret store.

Example commands:

```bash
python -m tools.run --run-etl --no-cache
python -m tools.run --run-feature-engineering --no-cache
python -m tools.run --run-end-to-end-data --no-cache
python -m tools.run --run-generate-instruct-datasets
python -m tools.run --run-training
python -m tools.run --run-evaluation
python -m tools.run --run-export-artifact-to-json
python -m tools.run --export-settings
```

The matching Poe tasks wrap these (see the earlier list); the Poe `run-*` tasks always pass `--no-cache`, while raw `python -m tools.run` lets you choose.

### 8. Export Pipeline and Step Registration

```python
# pipelines/export_artifact_to_json.py (full file, 16 lines)
from pathlib import Path

from zenml import pipeline
from zenml.client import Client

from steps import export as export_steps


@pipeline
def export_artifact_to_json(artifact_names: list[str], output_dir: Path = Path("output")) -> None:
    for artifact_name in artifact_names:
        artifact = Client().get_artifact_version(name_id_or_prefix=artifact_name)

        data = export_steps.serialize_artifact(artifact=artifact, artifact_name=artifact_name)

        export_steps.to_json(data=data, to_file=output_dir / f"{artifact_name}.json")
```

```python
# pipelines/__init__.py (full file, 17 lines)
from .digital_data_etl import digital_data_etl
from .end_to_end_data import end_to_end_data
from .evaluating import evaluating
from .export_artifact_to_json import export_artifact_to_json
from .feature_engineering import feature_engineering
from .generate_datasets import generate_datasets
from .training import training

__all__ = [
    "generate_datasets",
    "end_to_end_data",
    "evaluating",
    "export_artifact_to_json",
    "digital_data_etl",
    "feature_engineering",
    "training",
]
```

**Key Concepts**:

- **`Client().get_artifact_version(name_id_or_prefix=...)`** retrieves an artifact by name, decoupling export from the producing run. `raw_documents`, `cleaned_documents`, `instruct_datasets`, `preference_datasets` are the names listed in the config.
- **`pipelines/__init__.py` is the registration surface**: `tools/run.py` imports pipelines from it, so a new pipeline must be added there and to `__all__` to be reachable from the CLI. This is the step the exercise requires.
- The export writes to `output/<artifact_name>.json`; `.gitignore` ignores `output/`.

---

## 🔬 Deep Dive: Caching and Artifacts

```
Step input unchanged?  ──yes──►  reuse cached output (skip execution)
       │no
       ▼
Run step  ──►  store output as an artifact  ──►  record in the metadata store
```

- **Caching is content-addressed**: a step reruns only if its inputs, parameters, or code change. The cache key is a hash of the step function's source, its parameters, and its input artifact versions. During development, `--no-cache` forces fresh execution.
- **Caching is a correctness lever, not just speed.** If a step has a side effect that the hash does not capture (for example, writing to an external API whose data changed), a cached "success" can hide stale state. Always use `--no-cache` when the input world changed but the code did not.
- **Artifacts are typed**: `list`, `dict`, `TrainTestSplit`, etc. ZenML serializes them to the artifact store (local or S3).
- **Artifact names are stable IDs.** `export_artifact_to_json.yaml` references `raw_documents`, `cleaned_documents`, `instruct_datasets`, `preference_datasets` by name; those names come from `Annotated[..., "..."]` and `ArtifactConfig(name=...)`.

**Why artifacts matter**: training and evaluation read datasets by name from the artifact store, so a rerun of dataset generation automatically flows into training without passing files by hand.

### The caching decision tree

```
Did you change the step's Python source?  ── yes ──► cache miss, rerun
Did you change any step parameter?        ── yes ──► cache miss, rerun
Did an upstream artifact change?          ── yes ──► cache miss, rerun
Did only an unrelated file change?        ── no  ──► cache hit, reuse
Are you unsure / need a guarantee?               ──► use --no-cache
```

### Stack switching, in practice

```
                ┌───────────────────────────────┐
 zenml stack set│ default  →  local orchestrator │  localhost Mongo/Qdrant
 ───────────────┤ aws-stack→  SageMaker + S3+ECR │  cloud services
                └───────────────────────────────┘
 Same @pipeline code. Only the components behind the stack change.
```

The danger: the same command that is free locally can spin up SageMaker jobs and cost money. Check `zenml stack set --help` or the dashboard before large runs.

---

## 🛠️ Hands-On: Run a Local Pipeline

### Step 1: Bring up the local stack

```bash
poetry poe local-docker-infrastructure-up
poetry poe local-zenml-server-up
poetry poe set-local-stack
```

### Step 2: Export settings and run ETL

```bash
poetry poe export-settings-to-zenml
python -m tools.run --run-etl --no-cache
```

### Step 3: Inspect the dashboard

Open `http://localhost:8237`. Click the run and inspect:
- The DAG of steps.
- Each step's inputs/outputs and attached metadata (for example `num_documents`, `authors`).
- Artifact versions.

### Step 4: Run the end-to-end data pipeline

```bash
python -m tools.run --run-end-to-end-data --no-cache
```

Observe the nested child pipelines and the `wait_for` ordering.

### Step 5: Export an artifact

```bash
python -m tools.run --run-export-artifact-to-json
# writes output/<artifact_name>.json
```

### Step 6: Prove caching works

```bash
python -m tools.run --run-feature-engineering          # full run
python -m tools.run --run-feature-engineering          # all steps cached
python -m tools.run --run-feature-engineering --no-cache  # forced rerun
```

Compare run durations in the dashboard. The middle run should be near-instant.

---

## 📝 Exercise: Create a Custom Pipeline

### Task

Add a pipeline that computes simple corpus statistics.

1. Add a step `steps/stats/count_documents.py`:

```python
from typing_extensions import Annotated
from zenml import step


@step
def count_documents(documents: Annotated[list, "documents"]) -> Annotated[dict, "stats"]:
    stats = {}
    for doc in documents:
        key = doc.get_collection_name()
        stats[key] = stats.get(key, 0) + 1
    return stats
```

2. Add a pipeline `pipelines/stats.py` that queries the warehouse, then counts.
3. Register it in `pipelines/__init__.py` (`from .stats import stats` and add to `__all__`).
4. Add a Poe task and a `configs/stats.yaml`, then run it.

**Goal**: Practice the pipeline/step/artifact pattern end to end, including export registration in `__init__.py`, which is the step new users always forget.

---

## 📝 Exercise 2: Instrument a Step with Custom Metadata

### Task

Metadata is what makes the dashboard useful during debugging. Add domain-specific metadata to a step.

1. Pick `steps/feature_engineering/rag.py`'s `chunk_and_embed` (which already aggregates `num_chunks` and per-category counts).
2. Add the average tokens-per-chunk and the total number of chunks to the metadata dict.
3. Run `--run-feature-engineering --no-cache` and confirm the new keys appear on the `embedded_documents` artifact in the dashboard.
4. Remove one metadata key and observe the dashboard change on the next run.

**Goal**: Learn that `add_output_metadata` is free, versioned with the run, and the fastest path from "I wonder how big this dataset is" to a visible number - without adding logging code.

---

## 🐛 Common Pitfalls

- **Wrong stack active**: running on the `aws-stack` by accident incurs cloud cost. Check the dashboard or CLI before big runs.
- **Cache confusion**: an unexpectedly fast run usually means steps were cached. Use `--no-cache` when you need certainty, especially after external data changed.
- **`wait_for` mismatches**: passing the wrong invocation id (for example the ETL step instead of the last ETL step) changes ordering. The code returns `last_step.invocation_id` deliberately.
- **The `after` signature mismatch**: `feature_engineering` calls `query_data_warehouse(..., after=wait_for)` and `generate_datasets` calls `query_feature_store(after=wait_for)`, but neither step declares `after`. ZenML can consume `after` at the invocation layer, but the step definitions do not show it. If these paths error with an unexpected keyword, start here.
- **Secrets not exported**: remote runs fail to read settings if `export-settings-to-zenml` was never run. `Settings.load_settings()` then falls back to `.env` defaults, which may be wrong in the cloud.
- **`skip_build: False` by omission**: a config missing `skip_build: True` makes ZenML attempt a container build per run, which is slow and can fail without Docker.
- **`parent_image` points at `:latest`**: reproducible runs need a SHA-pinned image (Session 8.2).
- **Artifact name drift**: renaming an `Annotated` label or `ArtifactConfig(name=...)` breaks `export_artifact_to_json.yaml`, which references the old names.
- **`enable_cache` and side effects**: steps that crawl, push to Hugging Face, or train must not silently reuse stale outputs. Use `--no-cache` and be explicit.
- **Forgetting `__init__.py` registration**: a new pipeline is invisible to `tools/run.py` until added to `pipelines/__init__.py`.
- **`is_dummy: true` left on**: `configs/training.yaml` and `configs/evaluating.yaml` ship with `is_dummy: true`. A "successful" training run may have trained on a tiny dummy slice; flip to `false` for real runs.

---

## 🎓 Knowledge Check

1. **What is the difference between `@pipeline` and `@step`?**
   - Answer: A pipeline defines a DAG; a step is one node that runs a single unit of work.

2. **How are two pipelines chained without passing data?**
   - Answer: Via `wait_for` using step `invocation_id`s.

3. **What does `Annotated[type, "name"]` do on a step output?**
   - Answer: Names the artifact so it is identifiable in the UI and addressable by name.

4. **What does `skip_build: True` in a config mean?**
   - Answer: Use the pinned `parent_image` instead of rebuilding a container per run.

5. **What does `synchronous: false` change?**
   - Answer: The SageMaker orchestrator submits runs asynchronously instead of blocking.

6. **How does `enable_cache` behave by default?**
   - Answer: It reuses outputs for steps whose inputs/parameters/code are unchanged; `--no-cache` disables it.

7. **How does `end_to_end_data` order its three stages without passing data between ETL and feature engineering?**
   - Answer: It collects each `digital_data_etl` return (`crawl_links.invocation_id`) and passes them as `wait_for` to `feature_engineering`, then passes that return to `generate_datasets`.

8. **Where are new pipelines registered so `tools/run.py` can call them?**
   - Answer: `pipelines/__init__.py` (import plus `__all__`).

9. **What are the four artifacts listed in `export_artifact_to_json.yaml`?**
   - Answer: `raw_documents`, `cleaned_documents`, `instruct_datasets`, `preference_datasets`.

10. **What is the real name of the instruct-dataset step and why does the spelling matter?**
    - Answer: `generate_intruction_dataset` (a real typo); it is the import name, while the artifact name `instruct_datasets` is spelled correctly and is what export/training reference.

11. **Which two step calls pass an `after` argument the step signature does not declare?**
    - Answer: `query_data_warehouse(author_full_names, after=wait_for)` and `query_feature_store(after=wait_for)`.

12. **Why can switching stacks be dangerous?**
    - Answer: The same command that runs free locally can launch paid SageMaker jobs and write to cloud S3/ECR.

13. **What does `settings.export()` do?**
    - Answer: Pushes local `Settings` into the ZenML secret store so remote steps read the same configuration.

14. **Why is a content-addressed cache a correctness concern, not only a speed feature?**
    - Answer: A cached step can hide stale external state; if the world changed but inputs/hash did not, it reuses an outdated output.

15. **How do the training and evaluation pipelines reach their data?**
    - Answer: `steps/training/train.py` and `steps/evaluating/evaluate.py` delegate to SageMaker helpers (`run_finetuning_on_sagemaker`, `run_evaluation_on_sagemaker`) that read datasets by Hugging Face workspace/id from config parameters.

---

## 📖 Glossary

| Term | Meaning |
|------|---------|
| **Pipeline** | A `@pipeline`-decorated function defining a DAG of steps. |
| **Step** | A `@step`-decorated function; one node of compute. |
| **Artifact** | A typed, versioned output of a step, stored in the artifact store. |
| **invocation_id** | A handle to a specific step run; used for `wait_for` ordering. |
| **`wait_for`** | A directive that delays a step until named invocations finish, without passing data. |
| **Stack** | The set of infrastructure components (orchestrator, artifact store, registry, secrets). |
| **Orchestrator** | The component that decides where/how steps execute (local or SageMaker). |
| **Artifact store** | Where artifacts live (local filesystem or S3). |
| **Container registry** | Where step images live (local or ECR). |
| **Cache key** | A hash of step source, parameters, and input artifact versions. |
| **Child pipeline** | A pipeline invoked by another pipeline; appears nested in the run. |

---

## 📚 References

- [ZenML Pipelines and Steps](https://docs.zenml.io/user-guide/starter-guide)
- [ZenML Stacks](https://docs.zenml.io/user-guide/production-guide/understand-stacks)
- [ZenML Caching](https://docs.zenml.io/user-guide/advanced-guide/caching)
- [ZenML Artifacts](https://docs.zenml.io/user-guide/starter-guide/manage-artifacts)
- [ZenML step execution order](https://docs.zenml.io/user-guide/starter-guide)
- Book: *LLM Engineer's Handbook*, Chapter 2 (pages 61-73) and Chapter 11 (pages 444-461).

---

## 🔗 Next Session

**Session 9.1**: Data Warehouse Operations

We back up and restore the MongoDB warehouse as JSON.

---

## 📚 Additional Resources

- [ZenML documentation](https://docs.zenml.io/)
- [Poe the Poet](https://poethepoet.natn.io/)
- Cross-links: [Session 1.3 Infrastructure Layer](session_1.3_infrastructure_layer.md), [Session 8.1 Docker](session_8.1_docker.md), [Session 8.2 CI/CD](session_8.2_cicd.md), [Session 4.3 Streaming CDC](session_4.3_streaming_cdc.md)

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Sessions 1.3, 8.1, 8.2

**Outcome**: You can read and extend ZenML pipelines, manage stacks and configs, reason about artifacts and caching, and diagnose the step/caller signature mismatches.
