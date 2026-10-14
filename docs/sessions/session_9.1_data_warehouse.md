# Session 9.1: Data Warehouse Operations

## 🎯 Learning Objectives

By the end of this session, you will:
- Back up the MongoDB warehouse to JSON and restore it
- Use `NoSQLBaseDocument.to_mongo` / `from_mongo` for a lossless round trip
- Export ZenML artifacts to JSON
- Reason about incremental backups and versioning
- Understand the difference between raw data and derived artifacts
- Trace a document's full lifecycle from a Pydantic object, through BSON, to a JSON file, and back
- Explain why `bulk_find` is not paginated in this codebase and what that costs at scale
- Identify the schema changes required before incremental or differential backup is possible

---

## 🏗️ Architecture Overview

The project has **two separate backup surfaces** that are easy to confuse. One backs up the *source of truth* (raw documents in MongoDB). The other exports *derived artifacts* (datasets, models, embeddings) that ZenML has already versioned.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Two backup surfaces                                │
│                                                                            │
│  1. Raw data warehouse (MongoDB)  ── the SOURCE OF TRUTH                  │
│     tools.data_warehouse --export-raw-data                                 │
│        ArticleDocument / PostDocument / RepositoryDocument / UserDocument  │
│        → data/data_warehouse_raw_data/<ClassName>.json                     │
│     tools.data_warehouse --import-raw-data                                 │
│        <ClassName>.json → MongoDB (bulk_insert)                            │
│                                                                            │
│  2. Derived artifacts (ZenML)  ── REBUILDABLE, NOT SOURCE OF TRUTH        │
│     pipelines/export_artifact_to_json.py                                   │
│        Client().get_artifact_version(name) → serialize_artifact → to_json  │
│        → output/<artifact_name>.json                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

Both surfaces write JSON, but they serve opposite purposes:

| Surface | What it protects | Can be regenerated? | Where the truth lives |
|---------|------------------|---------------------|-----------------------|
| Raw warehouse | The crawled corpus | No — crawling is expensive and non-deterministic | MongoDB collections |
| ZenML artifacts | Datasets, embeddings, model refs | Yes — by re-running pipelines (given raw data) | ZenML metadata/artifact store |

**The central rule of this session**: raw data is the source of truth; feature stores and datasets are derived and can be rebuilt *from* it. If you can restore the raw warehouse, you can reconstruct everything downstream. If you only back up derived artifacts, you are backing up a cache.

### Why two mechanisms instead of one

A single "dump everything" tool would be simpler to explain but wrong to operate. Raw documents change on *every crawl*; derived artifacts change on *every pipeline run*. They have different cadences, different sizes, and different restore semantics. Keeping them separate lets you:

- Back up raw data on a slow schedule (cheap, small).
- Let ZenML handle artifact versioning automatically (it already stores every output).
- Restore a dataset without touching the corpus, or restore the corpus without touching datasets.

The cost of the split is two mental models and two commands. The benefit is that neither system can corrupt the other's truth.

---

## 📁 Key Files Explained

### 1. `tools/data_warehouse.py` - Export and Import

The CLI is a thin `click` wrapper around four module-private helpers. `__` prefix marks them as non-public (name-mangled to `_main__export` etc. inside the module), so they are not part of the CLI surface.

```python
# tools/data_warehouse.py
@click.command()
@click.option(
    "--export-raw-data", is_flag=True, default=False, help="Whether to export your data warehouse to a JSON file."
)
@click.option(
    "--import-raw-data", is_flag=True, default=False, help="Whether to import a JSON file into your data warehouse."
)
@click.option(
    "--data-dir",
    default=Path("data/data_warehouse_raw_data"),
    type=Path,
    help="Path to the directory containing data warehouse raw data JSON files.",
)
def main(export_raw_data, import_raw_data, data_dir: Path) -> None:
    assert export_raw_data or import_raw_data, "Specify at least one operation."

    if export_raw_data:
        __export(data_dir)
    if import_raw_data:
        __import(data_dir)
```

**Key Concepts**:
- **One CLI, two flags**. `--export-raw-data` and `--import-raw-data` can be combined; export runs first, then import, in the same invocation.
- **`--data-dir` defaults to `data/data_warehouse_raw_data`**, the directory already present in the repo. Override it to write into a dated backup folder.
- **The `assert` guards against running with no operation.** Note: `assert` statements are stripped when Python runs with `-O` (optimized mode), so this guard disappears under `python -O`. For a production tool, a raised `click.UsageError` would be more robust. This is a real edge case to be aware of, not a bug you must fix.

#### Export

```python
def __export(data_dir: Path) -> None:
    logger.info(f"Exporting data warehouse to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    __export_data_category(data_dir, ArticleDocument)
    __export_data_category(data_dir, PostDocument)
    __export_data_category(data_dir, RepositoryDocument)
    __export_data_category(data_dir, UserDocument)


def __export_data_category(data_dir: Path, category_class: type[NoSQLBaseDocument]) -> None:
    data = category_class.bulk_find()
    serialized_data = [d.to_mongo() for d in data]
    export_file = data_dir / f"{category_class.__name__}.json"

    logger.info(f"Exporting {len(serialized_data)} items of {category_class.__name__} to {export_file}...")
    with export_file.open("w") as f:
        json.dump(serialized_data, f)
```

**Key Concepts**:
- **`bulk_find()` with no filters returns the whole collection.** There is no `limit`/`offset` at this call site, so the entire collection is materialized in memory *twice*: once as a Mongo cursor of dicts, once as a list of Pydantic objects, then again as a list of dicts for `json.dump`. Memory is O(collection size).
- **`to_mongo()` converts each document** so `_id` is restored as the Mongo key and UUIDs become strings. Using `to_mongo` (not `model_dump`) is what makes the export re-importable. `model_dump` alone would leave the field named `id` and the value as a `uuid.UUID`, which `from_mongo`/`insert_many` would not round-trip correctly.
- **One file per document class**, named after the class (`ArticleDocument.json`, etc.), which the importer uses to recover the type. The filename *is* the type tag.

#### Import

```python
def __import(data_dir: Path) -> None:
    logger.info(f"Importing data warehouse from {data_dir}...")
    assert data_dir.is_dir(), f"{data_dir} is not a directory or it doesn't exists."

    data_category_classes = {
        "ArticleDocument": ArticleDocument,
        "PostDocument": PostDocument,
        "RepositoryDocument": RepositoryDocument,
        "UserDocument": UserDocument,
    }

    for file in data_dir.iterdir():
        if not file.is_file():
            continue

        category_class_name = file.stem
        category_class = data_category_classes.get(category_class_name)
        if not category_class:
            logger.warning(f"Skipping {file} as it does not match any data category.")
            continue

        __import_data_category(file, category_class)


def __import_data_category(file: Path, category_class: type[NoSQLBaseDocument]) -> None:
    with file.open("r") as f:
        data = json.load(f)

    logger.info(f"Importing {len(data)} items of {category_class.__name__} from {file}...")
    if len(data) > 0:
        deserialized_data = [category_class.from_mongo(d) for d in data]
        category_class.bulk_insert(deserialized_data)
```

**Key Concepts**:
- **The filename stem maps to a class** via the `data_category_classes` dict. Unknown files are skipped with a warning (so a stray `README.md` in the folder is harmless).
- **`from_mongo`** reverses the export: `_id` → `id` (UUID), strings → typed fields, with Pydantic validation. It **mutates** the input dict with `data.pop("_id")` (see the gotcha below).
- **`bulk_insert`** writes in one batch; empty files are no-ops (`if len(data) > 0`).
- **Idempotency caveat**: `bulk_insert` uses `insert_many`, so importing the same file twice **duplicates** documents (same `_id` would raise a duplicate-key error, but re-generated UUIDs would not). For a true restore, drop the target collections first or import into a fresh database.

#### The `from_mongo` mutation gotcha

```python
@classmethod
def from_mongo(cls: Type[T], data: dict) -> T:
    """Convert "_id" (str object) into "id" (UUID object)."""

    if not data:
        raise ValueError("Data is empty.")

    id = data.pop("_id")

    return cls(**dict(data, id=id))
```

`data.pop("_id")` removes the key from the caller's dict *in place*. On the import path this is harmless because each dict is consumed once. It becomes a bug if you ever iterate the same list of dicts twice (for example, to validate and then to import) — the second pass finds no `_id` and raises `KeyError`. If you write tooling on top of this API, copy the dict first: `from_mongo(dict(d))`.

---

### 2. `llm_engineering/domain/base/nosql.py` - The Persistence Base

Every warehouse document ultimately derives from `NoSQLBaseDocument`. Understanding it explains *why* the export works the way it does.

```python
# llm_engineering/domain/base/nosql.py
class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        """Convert "_id" (str object) into "id" (UUID object)."""
        if not data:
            raise ValueError("Data is empty.")
        id = data.pop("_id")
        return cls(**dict(data, id=id))

    def to_mongo(self: T, **kwargs) -> dict:
        """Convert "id" (UUID object) into "_id" (str object)."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        return parsed

    def model_dump(self: T, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)
        for key, value in dict_.items():
            if isinstance(value, uuid.UUID):
                dict_[key] = str(value)
        return dict_
```

The remaining methods are the CRUD surface:

| Method | Kind | Mongo call | Failure behavior |
|--------|------|-----------|------------------|
| `save(**kwargs)` | instance | `insert_one(to_mongo())` | logs exception, returns `None` |
| `get_or_create(**filters)` | class | `find_one`, else construct + `save` | re-raises `OperationFailure` |
| `bulk_insert(docs, **kwargs)` | class | `insert_many` | logs error, returns `False` |
| `find(**filters)` | class | `find_one` → `from_mongo` | logs error, returns `None` |
| `bulk_find(**filters)` | class | `find(filters)` → `from_mongo` each | logs error, returns `[]` |
| `get_collection_name()` | class | reads `cls.Settings.name` | raises `ImproperlyConfigured` |

**Key Concepts**:
- **`id` is a `UUID4` generated client-side**, not a Mongo `ObjectId`. This is a deliberate DDD choice: identity is assigned by the domain, not the database, and it is transportable to Qdrant (which has no ObjectId concept). The tradeoff is that you must translate between `id` and `_id` at every persistence boundary — which is exactly what `to_mongo`/`from_mongo` do.
- **`model_dump` is overridden** so UUIDs stringify even on the plain (non-Mongo) path. This is why `serialize_artifact` in the export pipeline produces JSON-safe dicts without extra conversion.
- **`get_collection_name` requires a nested `Settings` class** with a `name` attribute on each subclass. A document without one raises `ImproperlyConfigured` on first use.
- **`bulk_insert` swallows `WriteError`/`BulkWriteError`** and returns `False`. It does **not** raise on a duplicate `_id`. Callers must check the return value; the data-warehouse importer ignores it.

---

### 3. `llm_engineering/infrastructure/files_io.py` - Safe JSON I/O

```python
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
- **Two readers, two philosophies.** `data_warehouse.py` uses plain `json.dump`/`json.load` for consistency with the raw format (compact output, no indentation). Artifact export uses `JsonFileManager`, which writes `indent=4` (human-readable diffs) and creates parent directories automatically.
- **`write` returns the resolved absolute path**, and the `to_json` step returns it as a ZenML output (`exported_file_path`), so the path is recorded in the run metadata and surfaced in the dashboard.
- **`read` re-raises with context** rather than letting a raw `JSONDecodeError` escape. This makes a malformed file immediately identifiable by path.
- **No atomic write**: `write` opens with `"w"` and dumps directly. A crash mid-write leaves a truncated JSON file. For backups, write to a temp file and `os.replace` (atomic on the same filesystem). This is a production hardening item.

---

### 4. `pipelines/export_artifact_to_json.py` - ZenML Artifact Export

```python
# pipelines/export_artifact_to_json.py
@pipeline
def export_artifact_to_json(artifact_names: list[str], output_dir: Path = Path("output")) -> None:
    for artifact_name in artifact_names:
        artifact = Client().get_artifact_version(name_id_or_prefix=artifact_name)

        data = export_steps.serialize_artifact(artifact=artifact, artifact_name=artifact_name)

        export_steps.to_json(data=data, to_file=output_dir / f"{artifact_name}.json")
```

**Key Concepts**:
- **`Client().get_artifact_version(...)`** fetches an artifact by name from the ZenML metadata store — the bridge from a pipeline run to a file. If the name is ambiguous across runs, pass a full version id or prefix.
- **The pipeline loops in Python**, emitting one `serialize_artifact` step and one `to_json` step per artifact. Each call creates a *distinct* ZenML step invocation, so a failure on artifact N does not corrupt artifact N-1's output.
- **`output_dir` defaults to `Path("output")`**, which is gitignored (`.gitignore` line 171). Exported artifacts are working copies, not committed data.

---

### 5. `steps/export/serialize_artifact.py` and `to_json.py`

```python
# steps/export/serialize_artifact.py
@step
def serialize_artifact(artifact: Any, artifact_name: str) -> Annotated[dict, "serialized_artifact"]:
    serialized_artifact = _serialize_artifact(artifact)

    if serialize_artifact is None:
        raise ValueError("Artifact is None")
    elif not isinstance(serialized_artifact, dict):
        serialized_artifact = {"artifact_data": serialized_artifact}

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="serialized_artifact", metadata={"artifact_name": artifact_name})

    return serialized_artifact


def _serialize_artifact(arfifact: list | dict | BaseModel | str | int | float | bool | None):
    if isinstance(arfifact, list):
        return [_serialize_artifact(item) for item in arfifact]
    elif isinstance(arfifact, dict):
        return {key: _serialize_artifact(value) for key, value in arfifact.items()}
    if isinstance(arfifact, BaseModel):
        return arfifact.model_dump()
    else:
        return arfifact
```

**Key Concepts**:
- **`_serialize_artifact` recursively converts Pydantic models** (documents, datasets) to plain dicts; primitives pass through. Lists and dicts are walked recursively, so a list of `BaseModel` becomes a list of dicts.
- **The `if serialize_artifact is None` check is a known bug.** It tests the *function object* (always truthy), not the local variable `serialized_artifact`. The intended check was `if serialized_artifact is None`. In practice the bug is harmless because `_serialize_artifact(None)` returns `None`, then the next branch wraps `None` as `{"artifact_data": None}`. Being aware of this matters if you extend the step: do not rely on the None guard firing.
- **Non-dict outputs are wrapped** under the key `artifact_data`, so every exported file has a dict at the top level.
- **`to_json` delegates to `JsonFileManager.write`** and returns the absolute path as a typed step output with the alias `exported_file_path`.

```python
# steps/export/to_json.py
@step
def to_json(
    data: Annotated[dict, "serialized_artifact"],
    to_file: Annotated[Path, "to_file"],
) -> Annotated[Path, "exported_file_path"]:
    absolute_file_path = JsonFileManager.write(filename=to_file, data=data)
    return absolute_file_path
```

Run it:

```bash
python -m tools.run --run-export-artifact-to-json
# or, through poetry:
poetry poe run-export-artifact-to-json-pipeline
```

---

## 🔬 Deep Dive: The UUID ↔ BSON Round Trip

This is the single most important mechanism in the session. Trace one document end to end.

```
   Domain object                  Mongo (BSON)                JSON file (string)
 ┌────────────────┐             ┌────────────────┐          ┌────────────────────┐
 │ ArticleDocument│  to_mongo() │ {              │  json.dump│ {                  │
 │ id: UUID4      │────────────►│   "_id": "...."│──────────►│   "_id": "...."    │
 │ content: str   │             │   "content": ..│          │   "content": "..." │
 │ author_id: UUID│             │   "author_id": │          │   "author_id": "..."│
 └────────────────┘             │      "..." }   │          └────────────────────┘
        ▲                       └────────────────┘                   │
        │                                ▲                          │ json.load
        │        from_mongo()            └──────────────────────────┘
        └──────────────────────────────────────────────────────────
             UUID4(id)  ◄──  _id pops back into id, strings re-parse
```

What the conversion actually does, field by field:

| Stage | `_id` / `id` | UUID fields (e.g. `author_id`) | Nested `BaseModel` | Pydantic validation |
|-------|-------------|-------------------------------|--------------------|---------------------|
| Domain object | `id` is `UUID4` | `UUID4` | object | already valid |
| `to_mongo()` | renamed to `_id`, `str()` | stringified | `model_dump` dicts | none |
| `json.dump` | stays str | stays str | nested dicts | none |
| `json.load` | str | str | nested dicts | none |
| `from_mongo()` | becomes `id`, parsed to UUID | parsed to UUID | reconstructed | **yes** |

**Worked example** (abbreviated from a real `ArticleDocument`):

```python
from uuid import UUID
from llm_engineering.domain.documents import ArticleDocument

doc = ArticleDocument(
    content="Retrieval-augmented generation...",
    author_id=UUID("11111111-1111-1111-1111-111111111111"),
    author_full_name="Paul Iusztin",
    link="https://example.com/rag",
    platform="medium",
)

mongo_dict = doc.to_mongo()
# {"_id": "<uuid str>", "content": "...", "author_id": "1111...", "id"? no — popped}

restored = ArticleDocument.from_mongo(mongo_dict)
assert restored == doc            # __eq__ compares id only
assert restored.id == doc.id
assert str(restored.author_id) == str(doc.author_id)
```

**Edge cases**:
1. **`from_mongo({})`** raises `ValueError("Data is empty.")` because of the truthiness guard. An all-empty JSON array is handled one level up (`if len(data) > 0`), so this is only reachable if you call `from_mongo` directly with `{}`.
2. **Missing `_id`** (a hand-edited JSON file) raises `KeyError`. The importer does not catch it, so the whole import aborts. This is the practical argument for validating backups before restoring.
3. **Extra fields** (a JSON file from a newer schema with a field the current model lacks) cause Pydantic to raise unless the model is configured with `extra="ignore"`. Schema drift is the number-one restore failure.
4. **`exclude_unset`**: `to_mongo(exclude_unset=True)` omits fields that were never explicitly set. For a backup you want the default `False` so defaults are materialized. The data-warehouse exporter uses the default.

---

## 🛠️ Hands-On: Back Up and Restore

### Step 1: Export the warehouse

```bash
python -m tools.data_warehouse --export-raw-data
ls data/data_warehouse_raw_data/
# ArticleDocument.json  PostDocument.json  RepositoryDocument.json  UserDocument.json
```

### Step 2: Inspect the format

```python
import json

data = json.load(open("data/data_warehouse_raw_data/ArticleDocument.json"))
print(len(data), "articles")
print(list(data[0].keys()))  # includes "_id"; UUID fields are strings
print(type(data[0]["_id"]))  # <class 'str'>
```

### Step 3: Restore

```bash
python -m tools.data_warehouse --import-raw-data
```

Verify:

```python
from llm_engineering.domain.documents import ArticleDocument

docs = ArticleDocument.bulk_find()
print(len(docs))
```

**If you run this twice you will duplicate the corpus**, because import calls `insert_many` with freshly parsed UUIDs that differ from the existing ones only if the JSON `_id`s differ. To restore cleanly:

```python
from llm_engineering.domain.base.nosql import _database
_database["articles"].delete_many({})  # drop before re-import
```

### Step 4: Export a derived artifact

```bash
python -m tools.run --run-export-artifact-to-json
ls output/
# e.g. instruct_datasets.json
```

### Step 5: Prove the round trip is lossless on a single document

```python
from llm_engineering.domain.documents import PostDocument

original = PostDocument.bulk_find()[0]
round_tripped = PostDocument.from_mongo(original.to_mongo())
assert round_tripped.model_dump() == original.model_dump()
```

This asserts field-for-field equality after a `to_mongo` → `from_mongo` cycle, which is exactly what a backup/restore guarantees.

---

## 📝 Exercise 1: Incremental Backup by Timestamp

### Task

Add a filtered, incremental export.

1. Extend `__export` to accept a `since: str | None` option.
2. When `since` is set, filter with `bulk_find(id=...)` style... but `id` is not a timestamp, so instead add a `created_at` field to documents and filter on it.
3. Write exports to `data/backups/<timestamp>/`.
4. Document how you would detect which documents are new.

**Goal**: Understand that incremental backups need a monotonic marker (a timestamp or version), which the current schema lacks.

> Note: the raw documents have no `created_at` field today, so true incremental backup requires a schema change first. This is a real gap, not a coding mistake.

**Stretch**: `bulk_find` passes its filter straight to `collection.find(...)`, so a range query such as `bulk_find(created_at={"$gt": since})` would work with *zero* changes to the base class — Mongo stores ISO datetime natively. The only work is adding the field at ingestion.

---

## 📝 Exercise 2: A Restore Verifier

### Task

Backups you never test are not backups. Write `tools/verify_backup.py` that:

1. Loads each `<ClassName>.json` in a target directory.
2. Confirms every record has an `_id` and that `from_mongo` succeeds for all of them.
3. Reports the count per class and the earliest/latest `created_at` if present.
4. Exits non-zero if any record fails validation.

```python
import json
import sys
from pathlib import Path

from llm_engineering.domain.documents import ArticleDocument, PostDocument, RepositoryDocument, UserDocument

CLASSES = {
    "ArticleDocument": ArticleDocument,
    "PostDocument": PostDocument,
    "RepositoryDocument": RepositoryDocument,
    "UserDocument": UserDocument,
}


def verify(data_dir: Path) -> int:
    failures = 0
    for file in data_dir.glob("*.json"):
        cls = CLASSES.get(file.stem)
        if cls is None:
            print(f"skip: {file.name} (unknown class)")
            continue
        records = json.loads(file.read_text())
        for i, record in enumerate(records):
            if "_id" not in record or not record["_id"]:
                print(f"FAIL {file.name}[{i}]: missing _id")
                failures += 1
                continue
            try:
                cls.from_mongo(dict(record))  # copy: from_mongo pops _id
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {file.name}[{i}]: {exc}")
                failures += 1
        print(f"ok: {file.name} ({len(records)} records)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if verify(Path(sys.argv[1])) else 0)
```

**Goal**: Turn "we have a backup" into "we have a *verified* backup". Note the `dict(record)` copy — it avoids the `from_mongo` mutation gotcha described above.

---

## 🔬 Deep Dive: Warehouse, Feature Store, and Artifact Store

The book names three storage roles, and it is easy to blur them. Here is the distinction, mapped to this repo.

```
┌───────────────────────────────────────────────────────────────────────┐
│  Data warehouse (MongoDB)                                              │
│    role: system of record for RAW, crawled documents                  │
│    schema: flexible documents, one collection per domain class        │
│    in repo: ArticleDocument / PostDocument / RepositoryDocument /     │
│             UserDocument  → collections articles/posts/repositories/  │
│             users                                                      │
│    truth: YES                                                          │
├───────────────────────────────────────────────────────────────────────┤
│  Feature store (Qdrant + derived cleaned docs)                        │
│    role: embeddings and cleaned features for retrieval                │
│    in repo: EmbeddedArticleChunk / EmbeddedPostChunk /                │
│             EmbeddedRepositoryChunk (VectorBaseDocument subclasses)   │
│    truth: NO — rebuildable from raw                                   │
├───────────────────────────────────────────────────────────────────────┤
│  Artifact store (ZenML)                                                │
│    role: pipeline outputs (datasets, models, metrics) + metadata      │
│    in repo: export_artifact_to_json pulls from here                   │
│    truth: NO — rebuildable by re-running pipelines                    │
└───────────────────────────────────────────────────────────────────────┘
```

| Store | Unit stored | Versioned by | Backup tool | Restore impact |
|-------|-------------|--------------|-------------|----------------|
| Warehouse | documents | your JSON export | `tools.data_warehouse` | recovers the corpus |
| Feature store | vectors + payloads | Qdrant snapshot / rebuild | re-run feature pipeline | regenerable |
| Artifact store | typed artifacts | ZenML automatically | ZenML backend / export | regenerable |

**Why this matters for backup**: you only *must* back up the warehouse. Everything else is derivable, so its backup is a convenience, not a safety net. If you spend your backup budget on artifacts and lose the corpus, you have backed up the wrong layer.

---

## 📊 Capacity and Scale Planning

The current exporter is O(collection size) in memory and does not paginate. That is fine for the LLM Twin's few-thousand documents and wrong for a real corpus. Reason about scale before it bites.

**Memory estimate for export**: the peak is roughly

```
peak ≈ N × (size_of_bson_doc + size_of_pydantic_doc + size_of_dict_copy)
```

For a 2 KB average document and N = 10,000, that is ~10 M BSON + object overhead, tens to low-hundreds of MB in Python — acceptable. At N = 1,000,000 the same growth puts you at many GB and a likely OOM on a 32 GB workstation once Pydantic objects and dict copies coexist.

| Corpus size | Export approach | Risk |
|-------------|-----------------|------|
| < 50k docs | current `bulk_find` | none material |
| 50k-500k docs | batched `find` with skip/limit | slow, memory spikes |
| > 500k docs | `mongodump`/streaming, or cursor iteration to file | current code unsuitable |

**Pagination with the current base class**: `bulk_find(**filter_options)` forwards filters to `collection.find(...)`. Mongo's `find` supports `skip` and `limit` as cursor methods, but the base class does not expose them. Until it does, the practical scale path is `mongodump` (native, streaming, compressed) rather than extending the Python exporter.

```text
Native dump (scales):      mongodump --uri "$DATABASE_HOST" --db twin --out backup/
Native restore:            mongorestore --uri "$DATABASE_HOST" --db twin backup/twin/
```

The Python exporter's value is **portability and schema-awareness** (one JSON file per class, re-importable via the ODM), not raw throughput. Choose the tool by goal: portability vs scale.

---

## 🗂️ Worked Example: Rebuild a Derived Collection from a Raw Backup

This ties the whole session together. You lost the Qdrant instance; the warehouse is intact. Rebuild:

1. **Confirm the warehouse** (or restore it first):

```python
from llm_engineering.domain.documents import ArticleDocument, PostDocument, RepositoryDocument

for cls in (ArticleDocument, PostDocument, RepositoryDocument):
    print(cls.__name__, len(cls.bulk_find()))
```

2. **Re-run feature engineering** against the existing warehouse. The pipeline re-queries Mongo, cleans, chunks, embeds, and loads into Qdrant:

```bash
poetry poe run-feature-engineering-pipeline
```

3. **Verify the vector store** by retrieving:

```python
from llm_engineering.application.rag.retriever import ContextRetriever
docs = ContextRetriever(mock=False).search("How does RAG work?", k=3)
print(len(docs), "chunks")
```

4. **Confirm provenance**: because the rebuild reads the warehouse and not a stale artifact, the new embeddings reflect exactly the backed-up corpus. This is why raw data is the source of truth.

```
raw backup ──► MongoDB ──► feature_engineering pipeline ──► Qdrant
                                    │
                                    └──► ZenML artifacts (versioned)
```

**Edge case**: if the embedding model changed since the last build, the rebuild produces *different* vectors (a different embedding space). The warehouse is unchanged, but the derived store is not comparable to the old one. "Rebuildable" means rebuildable *with the same code and config*; record the embedding model id with the corpus version.

---

## ⚙️ Design Decisions (ADR-style)

**Decision 1: CSV vs JSON for the raw export.**
- Chosen: JSON, because documents are nested (tags, lists, nested objects) and JSON maps directly to BSON.
- Alternative: CSV — smaller but flattens nesting and loses type information.
- Revisit if: the corpus becomes flat and size dominates.

**Decision 2: One file per class vs one file for everything.**
- Chosen: one file per class, so the filename encodes the type and import can dispatch without a manifest.
- Alternative: a single `dump.json` with a `type` field per record — fewer files but a schema you must maintain.
- Revisit if: the number of document classes grows large.

**Decision 3: `insert_many` (append) vs upsert on import.**
- Chosen: append, because a restore normally targets an empty database.
- Alternative: upsert by `_id`, which would make import idempotent and let you merge backups.
- Revisit if: incremental restores become a requirement (which is exactly Exercise 1).

**Decision 4: synchronous CLI vs a ZenML step.**
- Chosen: a standalone `click` CLI (`tools.data_warehouse.py`), independent of the orchestrator.
- Alternative: a ZenML pipeline — versioned and monitored but requiring a stack to run.
- Revisit if: backups must run on a schedule with alerting.

---

## 🐛 Common Pitfalls

- **Re-import duplicates**: `bulk_insert` does not upsert. Back up into a fresh DB or clear collections before restoring.
- **Schema drift**: exporting with an old model and importing with a new one fails validation when required fields changed. Version your exports alongside the code.
- **Large collections in memory**: `bulk_find()` loads everything at once. For big warehouses, paginate (Qdrant's `bulk_find` already supports `limit`/`offset`; Mongo's does not in this code).
- **Raw vs derived confusion**: only `data_warehouse_raw_data` is the source of truth. Qdrant collections and datasets can be rebuilt from it.
- **Assuming `assert` is a guard**: under `python -O`, the "specify an operation" and "is a directory" asserts vanish. Both become no-ops or, worse, an import into a nonexistent path.
- **Trusting an untested backup**: a truncated file from a mid-write crash (`json.dump` is not atomic) loads as `JSONDecodeError` at restore time, when you least want to discover it.
- **`from_mongo` mutates its input**: iterating the same records list twice loses the `_id` on the second pass. Always pass a copy when you might reuse the dict.
- **`serialize_artifact` None-guard is ineffective**: `if serialize_artifact is None` checks the function, not the value. Do not build logic on it.
- **Exporting derived artifacts as if they were source**: `output/*.json` is gitignored and regenerable. Restoring it does not recover lost raw data.

---

## 🎓 Knowledge Check

1. **Why use `to_mongo` / `from_mongo` for backup instead of `model_dump`?**
   - They handle the `_id`/`id` and UUID↔string conversions, making the export re-importable. `model_dump` alone leaves `id` as a `uuid.UUID` and does not produce `_id`.

2. **How does the importer know which class to use for a file?**
   - By the filename stem, mapped through the `data_category_classes` dictionary. The filename is the type tag.

3. **What is the idempotency risk of importing twice?**
   - `bulk_insert` uses `insert_many` and is not an upsert; a second import duplicates documents. Drop the target collection first for a clean restore.

4. **What does `export_artifact_to_json` retrieve and how?**
   - ZenML artifacts by name via `Client().get_artifact_version(name_id_or_prefix=...)`, then serializes and writes each to `output/<name>.json`.

5. **Which directory holds the raw source of truth?**
   - `data/data_warehouse_raw_data/`.

6. **What is missing for incremental backups?**
   - A monotonic field such as `created_at` on documents.

7. **Why is `id` a UUID4 instead of a Mongo ObjectId?**
   - Identity is assigned in the domain and must be transportable to Qdrant, which has no ObjectId. The tradeoff is the `id`↔`_id` translation at every boundary.

8. **What does `bulk_insert` return on a duplicate key, and why does it matter?**
   - It logs and returns `False` rather than raising. Callers that ignore the return value silently lose write errors; the data-warehouse importer ignores it.

9. **How does `from_mongo` differ from a normal constructor?**
   - It expects Mongo BSON with `_id`, pops it into `id`, and passes the rest to Pydantic for validation and UUID parsing.

10. **Why does `data_warehouse.py` not use `JsonFileManager`?**
    - To keep the raw export compact and format-consistent; `JsonFileManager` writes indented, directory-creating output suitable for artifact exports.

11. **What breaks if you run the tool under `python -O`?**
    - The `assert` guards are stripped, so the "at least one operation" and "data_dir is a directory" checks disappear.

12. **What is the memory complexity of `__export_data_category`?**
    - O(collection size): the full collection is held as Pydantic objects and again as dicts before `json.dump`.

13. **Which method would you override to make backups atomic?**
    - `JsonFileManager.write`: write to a temp file in the same directory, then `os.replace` to the final path.

14. **How do you distinguish a raw backup from a derived artifact export at a glance?**
    - Raw backups live in `data/data_warehouse_raw_data/` and are compact, un-indented, one file per document class. Derived exports live in `output/`, are indented, and are named after ZenML artifacts.

---

## 📖 Glossary

- **BSON**: Binary JSON, MongoDB's storage format. Extends JSON with types such as `ObjectId` and `datetime`.
- **Document class**: A Pydantic model (e.g. `ArticleDocument`) deriving from `NoSQLBaseDocument` that maps to one Mongo collection.
- **`_id`**: Mongo's primary-key field. The project stores the domain `id` (UUID4) stringified here.
- **UUID4**: A randomly generated universally unique identifier; the domain's identity type.
- **Round trip**: A `to_mongo` → `from_mongo` cycle that must reproduce the original object field-for-field.
- **Derived artifact**: Any output (dataset, embedding collection, model reference) that can be regenerated from raw data.
- **Source of truth**: The raw warehouse; the only data that cannot be cheaply regenerated.
- **Upsert**: Insert-or-update; `bulk_insert` is *not* an upsert.
- **Schema drift**: Divergence between the schema used to write a backup and the schema used to read it.
- **Idempotent**: Repeated application yields the same result. The importer is not idempotent.
- **Atomic write**: A write that either fully succeeds or leaves the original intact; `json.dump` alone is not atomic.

---

## 🔗 Next Session

**Session 9.2**: Performance Optimization

We profile and tune batching, caching, and quantization across the pipeline.

---

## 📚 Additional Resources

- [PyMongo: Insert and Find](https://pymongo.readthedocs.io/en/stable/tutorial.html)
- [ZenML Artifacts](https://docs.zenml.io/user-guide/starter-guide/manage-artifacts)
- [MongoDB Backup Methods](https://www.mongodb.com/docs/manual/core/backups/)
- [Pydantic: Model export and `model_dump`](https://docs.pydantic.dev/latest/concepts/serialization/)
- [Python `uuid` module](https://docs.python.org/3/library/uuid.html)
- [Click options and flags](https://click.palletsprojects.com/en/stable/options/)

---

## 🔎 References

- **Book**: *LLM Engineer's Handbook* — Chapter 3 (data engineering, MongoDB warehouse) and Chapter 11 (MLOps/LLMOps, versioning raw vs derived data).
- **Repo**: `tools/data_warehouse.py`, `llm_engineering/domain/base/nosql.py`, `llm_engineering/infrastructure/files_io.py`, `pipelines/export_artifact_to_json.py`, `steps/export/*.py`.
- **Related sessions**: [`session_1.2_domain_layer.md`](session_1.2_domain_layer.md) (the document model), [`session_1.3_infrastructure_layer.md`](session_1.3_infrastructure_layer.md) (Mongo connection), [`session_8.3_zenml.md`](session_8.3_zenml.md) (artifact store).
- **Curriculum**: [`../CURRICULUM.md`](../CURRICULUM.md) and [`../BOOK-MAP.md`](../BOOK-MAP.md).

---

**Estimated Time**: 3-4 hours

**Prerequisites**: [Session 1.2](session_1.2_domain_layer.md), [Session 1.3](session_1.3_infrastructure_layer.md), [Session 8.3](session_8.3_zenml.md)

**Outcome**: You can back up and restore the warehouse, verify a backup is restorable, export artifacts, and reason about data provenance and versioning.
