# Session 3.1: Instruction Dataset Creation

## 🎯 Learning Objectives

By the end of this session, you will:
- Generate instruction-answer pairs from cleaned documents with GPT-4o-mini
- Understand the `DatasetGenerator` abstract base and its template isolation
- Control prompt size with `tiktoken` and `OPENAI_MAX_TOKEN_WINDOW`
- Parse a list of structured objects with a custom Pydantic output parser
- Produce the exact `instruction`/`output` schema that SFT training expects
- Explain why the book's standalone script and the repo's ZenML pipeline solve the same problem two ways

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    generate_datasets pipeline                       │
│                                                                      │
│  query_feature_store(after=...)                                     │
│      │  scroll cleaned_articles / cleaned_posts / cleaned_repos     │
│      ▼                                                               │
│  create_prompts(documents, dataset_type)                            │
│      │  extract_substrings()  → 1000..2000 char extracts            │
│      │  group_by_category → GenerateDatasetSamplesPrompt per doc     │
│      ▼                                                               │
│  generate_intruction_dataset(prompts)                                  │
│      │  ChatOpenAI(gpt-4o-mini, temp=0.7, max_tokens=1200)          │
│      │  chain.batch(batch=24)  →  ListPydanticOutputParser           │
│      │  build_dataset → InstructDataset per category                 │
│      ▼                                                               │
│  InstructTrainTestSplit(train, test)                                │
│      │                                                               │
│      ▼  (optional)                                                   │
│  push_to_huggingface(dataset_id)                                    │
└────────────────────────────────────────────────────────────────────┘
```

The instruction dataset teaches the model **what** to answer. The preference dataset (Session 3.2) teaches it **which** of two answers is better.

### Where this fits in the LLM Twin project

The LLM Twin is a digital replica of a specific author. Three data types flow through the project:

| Stage | Input | Output | Consumer |
|-------|-------|--------|----------|
| Crawl + clean (2.2) | Raw web pages | `CleanedArticleDocument` etc. | This session |
| Instruction dataset (this session) | Cleaned documents | `instruction` / `answer` pairs | SFT (5.1) |
| Preference dataset (3.2) | Same cleaned documents | `instruction` / `chosen` / `rejected` | DPO (5.2) |

The key architectural decision: the **same cleaned corpus** feeds both dataset generators. Only the prompt template, the output schema, and the post-processing differ. This is why `DatasetGenerator` is an abstract base with a template-method `generate()`.

---

## 📚 Theory First: What the Book Teaches (pp. 207-235)

Before the repo code, the book defines the problem and the quality bar. These ideas explain **why** the code is shaped the way it is.

### Instruction datasets as pairs

An instruction dataset is pairs of `instruction` (model input) and `answer` (expected output). Templates such as **Alpaca** add two optional fields:

- `system` — a meta-prompt that steers general behavior (e.g. "You are a helpful assistant"). It is a subfield of the instruction.
- `input` — data the task needs, separate from the task itself (e.g. a list of concepts to include in a sentence).

During fine-tuning you can train on instruction+answer, or on the answer only.

### The three quality dimensions

The book judges any instruction sample on three axes. Every design decision in the pipeline maps to one of them:

| Dimension | Definition | How the repo addresses it |
|-----------|------------|---------------------------|
| **Accuracy** | Factually correct and relevant to the instruction | "Only use concepts from the context" prompt rule grounds answers in the source |
| **Diversity** | Covers the topics, contexts, lengths, styles the model will see | `temperature=0.7`; many extracts from many documents |
| **Complexity** | Non-trivial, multi-step, challenging | Weakest axis here; the book defers this to Chapter 6 augmentation |

### Data quantity guidance

| Model type | Typical sample count | Source |
|------------|---------------------|--------|
| Large (~70B) | as few as ~1,000 high-quality samples | LIMA |
| General-purpose | ≥1 million | OpenHermes, Dolphin |
| Llama 3 (full pipeline) | ~10 million (SFT + alignment) | Meta |
| Yi | <10,000 | 01-ai |
| Task-specific | 100 - 100,000 | Book |
| Domain-specific | highly variable (medicine/law can approach general scale) | Book |

**Why this matters for the LLM Twin**: the book's own run produces **3,335 pairs** (p. 234), well below 1,000 per category for a general model. That is deliberate. The Twin is a **domain/style-specific** model, not general-purpose. It later compensates by upsampling with the general-purpose FineTome dataset during SFT (Chapter 5, pp. 251-252).

### Backtranslation and rephrasing

Raw articles have no instructions. The book's transformation is **backtranslation**: provide the expected answer (a text chunk) as if it were the output, and ask the LLM to generate the matching instruction. Because a raw paragraph is not always a well-formed answer, the model **rephrases** it while imitating the author's writing style. That style imitation is what makes the trained model an *LLM Twin*.

> If you want the curation side of this chapter (filtering, dedup, decontamination, quality eval, augmentation), see `session_5.4_data_curation.md`.

---

## 📁 Key Files Explained

### 1. `llm_engineering/application/dataset/generation.py` - The Base Generator

**Purpose**: template-method base class shared by instruction and preference generation.

```python
# llm_engineering/application/dataset/generation.py
class DatasetGenerator(ABC):
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)
    dataset_type: DatasetType | None = None

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
    prompt_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"

        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION else "instruction-answer triples"
        )
        input_variables = {
            "dataset_format": dataset_format,
        }
        system_prompt = cls.system_prompt_template.format(**input_variables)

        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt,
        )
```

**Key Concepts**:
- **`tokenizer` is a class attribute** built once from the configured OpenAI model, so token counting matches the model that will consume the prompt. Building it once avoids re-loading the BPE tables on every call.
- **`dataset_type`** is the only abstract discriminator: subclasses set `DatasetType.INSTRUCTION` or `DatasetType.PREFERENCE`.
- **`system_prompt_template`** uses a single `{dataset_format}` slot, so one template serves both generators. The format is fixed by the class, so there is no injection surface.
- **`get_system_prompt` returns a `Prompt` domain object**, not a raw string. This keeps prompts as first-class, inspectable artifacts (they are `VectorBaseDocument`s).

#### Why a class attribute rather than an instance

`tiktoken.encoding_for_model()` downloads and caches a BPE vocabulary. Making `tokenizer` a class attribute means it is created once when the module is imported, not once per document. For a corpus of thousands of documents this is a large, silent saving. **Tradeoff**: the tokenizer is bound to `settings.OPENAI_MODEL_ID` at import time; changing the model in settings mid-process will not rebuild it.

---

### 2. Prompt Building with Token Truncation

```python
@classmethod
def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
    documents = generation_utils.extract_substrings(documents)

    grouped_prompts = {}
    grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
    for category, category_documents in grouped_cleaned_documents.items():
        category_prompts = [cls.get_prompt(document) for document in category_documents]
        grouped_prompts[category] = category_prompts

    return grouped_prompts


@classmethod
def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
    assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

    data_category = document.get_category()

    prompt_template = PromptTemplate.from_template(
        template=cls.prompt_template_str,
        template_format="jinja2",
    )
    input_variables = {
        "extract": document.content,
    }
    prompt = prompt_template.format(**input_variables)
    prompt_tokens = cls.tokenizer.encode(prompt)
    if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
        prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
        prompt = cls.tokenizer.decode(prompt_tokens)

    prompt = GenerateDatasetSamplesPrompt(
        template=prompt_template.template,
        input_variables=input_variables,
        content=prompt,
        num_tokens=len(prompt_tokens),
        data_category=data_category,
        document=document,
    )

    return prompt
```

**Key Concepts**:
- **`extract_substrings`** (from `dataset/utils.py`) reuses `chunk_document` to slice long cleaned content into 1000-2000 character extracts and returns a `model_copy()` per extract. Each extract becomes one generation prompt.
- **`template_format="jinja2"`** is required because the templates contain literal JSON braces (`{ "instruction": ... }`). Jinja2 uses `{{extract}}` for substitution, leaving `{`/`}` untouched. (Python `str.format`-style would crash on the literal braces.)
- **Hard truncation at `OPENAI_MAX_TOKEN_WINDOW`** guarantees the prompt fits even for extreme documents. `num_tokens` records the final count.
- **`GenerateDatasetSamplesPrompt` carries the source document** so the dataset sample can be traced back to its origin.

#### The `OPENAI_MAX_TOKEN_WINDOW` value

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

For `gpt-4o-mini`, `128000 * 0.90 = 115200`. **Why 90%?** The window must also hold the model's *response* (up to 1200 tokens) and the system prompt. Leaving a 10% headroom is a safety margin for token-count estimation differences. The `extract_substrings` 2000-character cap makes this truncation effectively never fire in practice for this pipeline; it is a defensive backstop.

#### Note on truncation order in `get_prompt`

The prompt string is the template (with instructions) *plus* the extract. Truncating the token list from the end cuts the **extract first**, because the extract is the last part of the prompt. The instruction template survives intact. This is the desired behavior: a slightly clipped extract still yields usable pairs, whereas a clipped instruction would confuse the model.

---

### 3. Generation Loop with Batching and a Custom Parser

```python
    @classmethod
    def generate(
        cls,
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> TrainTestSplit:
        assert cls.dataset_type is not None, "Dataset type must be set before calling generate()"

        def _to_langchain(prompt: GenerateDatasetSamplesPrompt) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]
            return messages

        if mock:
            llm = FakeListLLM(responses=[constants.get_mocked_response(cls.dataset_type)])
        else:
            assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"

            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=2000 if cls.dataset_type == DatasetType.PREFERENCE else 1200,
                temperature=0.7,
            )
        parser = ListPydanticOutputParser(pydantic_object=cls._get_dataset_sample_type())

        chain = llm | parser

        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]
            batches = utils.misc.batch(langchain_category_prompts, size=24)

            flattened_instruct_dataset_samples = []
            for batch in batches:
                try:
                    batched_dataset_samples = chain.batch(batch, stop=None)

                    for instruct_dataset_sample_batch in batched_dataset_samples:
                        flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)
                except OutputParserException:
                    logger.exception(f"Failed to parse the output JSON for a batch for category {category}")

            dataset = domain.dataset.build_dataset(
                dataset_type=cls.dataset_type, category=category, samples=flattened_instruct_dataset_samples
            )
            datasets[category] = dataset
            logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'.")

        processed_datasets = cls.post_process_datasets(datasets, test_size=test_size)

        return processed_datasets
```

**Key Concepts**:
- **`temperature=0.7`** produces varied instructions. A deterministic generator would yield near-duplicate questions across extracts.
- **`max_tokens` differs by task**: 1200 for pairs, 2000 for triples (three strings each).
- **`chain.batch(batch, stop=None)`** sends up to 24 prompts concurrently. This is the throughput dial: `batch` uses LangChain's parallel execution, so higher values increase rate-limit risk.
- **`OutputParserException` is caught per batch**. A single malformed JSON response drops only that batch; the rest of the dataset survives.
- **`mock=True` swaps in `FakeListLLM`** so the whole pipeline can be tested without an API key (used in CI).

#### Why `chain.batch` and not a `ThreadPoolExecutor`

The book (p. 233) uses `ThreadPoolExecutor(max_workers=4)`. The repo uses LangChain's `chain.batch`. Both parallelize; the tradeoff is:

| Approach | Pro | Con |
|----------|-----|-----|
| `ThreadPoolExecutor(max_workers=4)` (book) | explicit; easy to reason about the concurrency cap | manual per-item error handling |
| `chain.batch(batch, stop=None)` (repo) | idiomatic LangChain; returns parsed objects already | concurrency capped at the outer batch size (24); rate-limit mountain is steeper |

The book deliberately uses only **4 workers** because higher values exceed OpenAI rate limits. The repo's 24-per-batch is more aggressive. If you see HTTP 429s, lower the `batch(..., size=24)` value first.

#### Why `stop=None`

`stop=None` explicitly tells the underlying `ChatOpenAI` not to append additional stop tokens to the request. The dataset prompt already ends with the extract and the JSON schema clause; adding a stop sequence could truncate valid JSON mid-array. Passing `None` makes the behavior explicit and matches the non-streaming batch path.

#### The per-batch failure mode

If the model returns five objects where one fails Pydantic validation, the **entire batch of 24 prompts** raises `OutputParserException`, and all 24 responses are discarded — not just the bad one. This is the single biggest quality-vs-yield tradeoff in the generator. Alternatives and their costs:

1. **Current design**: drop the batch. Simple, but loses up to 24 extracts' worth of work.
2. **Per-prompt fallback**: catch and retry the failing prompt individually. More API calls, more code.
3. **Lenient parsing**: repair the JSON or drop only the bad object. Highest yield, but risks injecting malformed samples.

For a first build, the current design is correct: a bad batch is a signal to fix the prompt, not to silently salvage data.

---

### 4. `output_parsers.py` - List of Pydantic Objects

**Purpose**: parse a JSON **array** of sample objects into Pydantic models.

```python
# llm_engineering/application/dataset/output_parsers.py
from langchain.output_parsers import PydanticOutputParser


class ListPydanticOutputParser(PydanticOutputParser):
    def _parse_obj(self, obj: dict | list):
        if isinstance(obj, list):
            return [super(ListPydanticOutputParser, self)._parse_obj(obj_) for obj_ in obj]
        else:
            return super(ListPydanticOutputParser, self)._parse_obj(obj)
```

**Key Concepts**:
- LangChain's `PydanticOutputParser` expects a **single** object. This subclass detects a top-level list and maps `_parse_obj` over its items.
- It reuses all of the base class's validation and format instructions, so no prompt-format string is duplicated.
- If any item fails validation, the whole batch raises `ParserOutputException`, which the generator catches.

#### How the parser is wired

```python
parser = ListPydanticOutputParser(pydantic_object=cls._get_dataset_sample_type())
chain = llm | parser
```

`cls._get_dataset_sample_type()` returns `InstructDatasetSample` for instruction mode and `PreferenceDatasetSample` for preference mode. The parser schema therefore enforces the correct fields per mode: `{instruction, answer}` or `{instruction, rejected, chosen}`.

#### Book vs repo naming (do not conflate)

The book presents a **standalone** implementation with different class and field names. They solve the same problem; keep them straight:

| Concept | Book (pp. 227-232) | Repo |
|---------|--------------------|------|
| Parse JSON array into typed objects | `InstructionAnswerSet.from_json(...)` | `ListPydanticOutputParser` |
| Top-level JSON key | `"instruction_answer_pairs"` | a bare JSON array (no wrapper key) |
| Pair fields | `instruction`, `answer` | `instruction`, `answer` |
| Preference fields | `generated_answer`, `extracted_answer` | `rejected`, `chosen` |
| Concurrency | `ThreadPoolExecutor(max_workers=4)` | `chain.batch(size=24)` |
| Structured output | OpenAI JSON mode (`response_format={"type": "json_object"}`) | LangChain parser + prompt-specified JSON |

The book's `InstructionAnswerSet` takes a `json_str` and builds tuples:

```python
class InstructionAnswerSet:
    def __init__(self, pairs):
        self.pairs = pairs

    @classmethod
    def from_json(cls, json_str: str) -> "InstructionAnswerSet":
        data = json.loads(json_str)
        pairs = [(pair["instruction"], pair["answer"]) for pair in data["instruction_answer_pairs"]]
        return cls(pairs)

    def __iter__(self):
        return iter(self.pairs)
```

The repo instead asks the model for a bare JSON list and wraps it in Pydantic, which gives validation for free. Both are valid; the repo trades a wrapper key for schema checking.

---

### 5. `InstructionDatasetGenerator` - The Prompt and Pipeline Hooks

```python
class InstructionDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.INSTRUCTION

    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \

Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{{extract}}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    ) -> TrainTestSplit:
        train_test_split = generation_utils.create_instruct_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split
```

**Prompt-design principles in this template**:
- **"Only use concepts from the context"** grounds the model and prevents hallucinated facts.
- **"Instructions must never explicitly mention a context / system / course / extract"** makes samples look like real user questions, not dataset artifacts.
- **"Answers must imitate the writing style of the context"** is what makes the trained model an *LLM Twin* — it copies the author's voice.
- **A worked example** anchors the output shape before the strict JSON schema.
- **`{{extract}}`** (Jinja2) is the only substitution.
- **"generate five ... pairs"** multiplies a limited corpus. The book reasons that more samples means better style imitation, so each ~2000-char chunk yields five pairs.

**Key Concepts**:
- **`random_state=42`** makes the train/test split reproducible; reruns produce the same split.
- `post_process_datasets` is the extension hook: instruction generation only splits, preference generation also filters (Session 3.2).

#### How the book's prompt differs

The book's prompt (p. 231) is nearly identical but uses a Pydantic-style wrapper and JSON mode:

- It asks for `"instruction_answer_pairs": [ {...}, ... ]` (a wrapper object), while the repo asks for a bare array.
- The book sets `response_format={"type": "json_object"}` (OpenAI JSON mode). The repo instead specifies the JSON shape in the prompt and relies on the parser, which is why the repo's `ListPydanticOutputParser` must accept a top-level list.

The repo's approach works with any OpenAI-compatible endpoint; JSON mode is OpenAI-specific.

---

### 6. `dataset/utils.py` - Splitting and Sample Typing

```python
def create_instruct_train_test_split(
    data: dict[DataCategory, InstructDataset], test_size=0.2, random_state=42
) -> InstructTrainTestSplit:
    train_data = {}
    test_data = {}

    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )
            train_samples = [InstructDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [InstructDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]
        else:
            train_samples = []
            test_samples = []

        train_dataset = InstructDataset(category=category, samples=train_samples)
        test_dataset = InstructDataset(category=category, samples=test_samples)

        train_data[category] = train_dataset
        test_data[category] = test_dataset

    return InstructTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)
```

**Key Concepts**:
- **The split is per category**, so a small category is not starved by a large one.
- **Empty categories are tolerated** (the `else` branch), which matters when an author has only articles and no posts.
- **Round-trip through `model_dump` -> sklearn -> Pydantic** keeps the domain objects pure and avoids sklearn knowing about Pydantic.

#### Why per-category splitting matters

Imagine 1,000 articles and 10 posts. A global 20% test split would put roughly 200 articles and 2 posts in test — fine on average, but it can leave a whole category empty in a small run. Per-category splitting guarantees each category contributes both train and test samples, which keeps per-category evaluation honest.

**Edge case**: with exactly one sample in a category and `test_size=0.1`, sklearn returns a one-item train set and a zero-item test set (it rounds). The split succeeds but the test set is empty. The pipeline does not guard against this; it only guards against a completely empty input category.

---

### 7. The `generate_datasets` Pipeline

```python
# pipelines/generate_datasets.py
@pipeline
def generate_datasets(
    dataset_type: DatasetType = DatasetType.INSTRUCTION,
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
    wait_for: str | list[str] | None = None,
) -> None:
    cleaned_documents = cd_steps.query_feature_store(after=wait_for)
    prompts = cd_steps.create_prompts(documents=cleaned_documents, dataset_type=dataset_type)
    if dataset_type == DatasetType.INSTRUCTION:
        dataset = cd_steps.generate_intruction_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock)
    elif dataset_type == DatasetType.PREFERENCE:
        dataset = cd_steps.generate_preference_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if push_to_huggingface:
        cd_steps.push_to_huggingface(dataset=dataset, dataset_id=dataset_id)
```

**Key Concepts**:
- **One pipeline, two modes** selected by `dataset_type`.
- **`query_feature_store`** scrolls Qdrant `cleaned_*` collections with pagination (`bulk_find(limit=1)` + `next_offset`), fetching from articles, posts, and repositories in parallel via `ThreadPoolExecutor`.
- **`ArtifactConfig(name="instruct_datasets", tags=[...])`** names the ZenML artifact so later steps (training) can reference it by name.

#### The real signature mismatch (verified)

```python
# pipelines/generate_datasets.py
cleaned_documents = cd_steps.query_feature_store(after=wait_for)

# steps/generate_datasets/query_feature_store.py
@step
def query_feature_store() -> Annotated[list, "queried_cleaned_documents"]:
    ...
```

The pipeline passes `after=wait_for`, but the step function declares **no parameters at all**. There is no `after` argument. Depending on the ZenML version and `wait_for` value, this either raises a `TypeError` at pipeline definition time or is silently absorbed by ZenML's step-parameter injection. If you hit a signature error locally, this is the mismatch. The book's `end_to_end_data.yaml` / `feature_engineering.yaml` pipeline (Chapter 4) has a real `after` parameter; this step appears to be an older or re-exported copy.

**What to do**: if you need it, change the step signature to accept `after: Annotated[str | list[str] | None, "after"] = None` and forward it to ZenML's `after` mechanism. Do not guess — verify against the installed ZenML version. This is exactly the kind of drift the "inspect before changing" rule exists to catch.

#### The step graph

```
query_feature_store  ──►  create_prompts  ──►  generate_intruction_dataset
                                                         │
                                                         ▼
                                             (optional) push_to_huggingface
```

Each `@step` adds its own metadata to the ZenML dashboard:

| Step | Artifact name | Metadata recorded |
|------|---------------|-------------------|
| `create_prompts` | `prompts` | `data_categories`, `data_categories_num_prompts` |
| `generate_intruction_dataset` | `instruct_datasets` | `data_categories`, `test_split_size`, `train_num_samples_per_category`, `test_num_samples_per_category` |
| `push_to_huggingface` | — | pushes `dataset.to_huggingface(flatten=True)` to the Hub |

---

## 🔬 Worked Example: One Extract, End to End

Let's trace a single document through the whole pipeline with realistic values.

**Input** — one cleaned article, content truncated for readability:

```
"Data pipelines are the backbone of any machine learning system. They move data from
sources to models, cleaning and shaping it along the way. A feature store sits between
the pipeline and the model, serving both training and inference. Without a feature
store, training and serving drift apart, and the model sees different data than it was
trained on. ..."
```

**Step 1 — `extract_substrings`** calls `chunk_document(content, 1000, 2000)`, which:

1. Splits on sentence boundaries with the regex `(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s`.
2. Concatenates sentences until the next sentence would exceed `max_length=2000`.
3. Emits a chunk only if its length is `>= min_length=1000`.

Result: the 5,000-character article becomes roughly 3 extracts of ~1,600 chars each.

**Step 2 — `get_prompt`** renders the Jinja2 template with `{{extract}}` = extract #1, giving a prompt whose `num_tokens` is recorded (well under 115200).

**Step 3 — `generate`** groups prompts by category (`ARTICLES`), batches 24 at a time, and calls GPT-4o-mini at `temperature=0.7`, `max_tokens=1200`.

**Step 4 — `ListPydanticOutputParser`** validates the response. A good response:

```json
[
  {"instruction": "Explain the role of a feature store in a machine learning pipeline.",
   "answer": "A feature store sits between the data pipeline and the model..."},
  ...  // 5 total
]
```

A bad response (fenced in ```` ```json ````) raises `OutputParserException`, and that whole 24-prompt batch is dropped.

**Step 5 — `build_dataset`** produces an `InstructDataset` with `category=DataCategory.ARTICLES` and 15 samples (3 extracts × 5 pairs).

**Step 6 — `create_instruct_train_test_split`** with `test_size=0.1` moves ~1-2 samples to test.

**Step 7 — `to_huggingface`** maps the internal schema to the SFT schema:

```
InstructDatasetSample          Hugging Face Dataset
{instruction, answer}   ───►   {instruction, output}
```

---

## 📊 Schema Reference

### Domain models (`llm_engineering/domain/dataset.py`)

```python
class DatasetType(Enum):
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"

class InstructDatasetSample(VectorBaseDocument):
    instruction: str
    answer: str

class InstructDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[InstructDatasetSample]

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]
        return Dataset.from_dict(
            {"instruction": [d["instruction"] for d in data], "output": [d["answer"] for d in data]}
        )
```

| Internal field | Hugging Face column | Why the remap |
|----------------|---------------------|---------------|
| `instruct_signal` -> `instruction` | `instruction` | standard SFT key |
| `answer` | `output` | matches `mlabonne/llmtwin` and TRL conventions |

---

## 🛠️ Hands-On: Generate a Small Instruction Dataset

### Step 1: Fetch cleaned documents

```python
from llm_engineering.domain.cleaned_documents import CleanedArticleDocument

docs, offset = CleanedArticleDocument.bulk_find(limit=1)
while offset:
    more, offset = CleanedArticleDocument.bulk_find(limit=1, offset=offset)
    docs.extend(more)
print("Cleaned articles:", len(docs))
```

### Step 2: Build prompts

```python
from llm_engineering.application.dataset import generation
from llm_engineering.domain.dataset import DatasetType

gen = generation.get_dataset_generator(DatasetType.INSTRUCTION)
prompts = gen.get_prompts(docs[:3])
for category, category_prompts in prompts.items():
    print(category, len(category_prompts), "prompts")
    print(category_prompts[0].content[:400])
```

### Step 3: Generate with mock, then real

```python
split = gen.generate(prompts, test_size=0.2, mock=True)  # no API key needed
for category, ds in split.train.items():
    print(category, ds.num_samples)
    print(ds.samples[0].instruction if ds.samples else "(empty)")
```

### Step 4: Inspect the Hugging Face schema

```python
hf = split.train[list(split.train.keys())[0]].to_huggingface()
print(hf.column_names)  # ['instruction', 'output']
print(hf[0])
```

### Step 5: Drive the full pipeline (ZenML)

```bash
# Edit configs/generate_instruct_datasets.yaml first (set mock: true to avoid spend)
python -m tools.run --run-generate-instruct-datasets --no-cache
```

The config supplies `test_split_size`, `dataset_type`, `push_to_huggingface`, `dataset_id`, and `mock`:

```yaml
parameters:
  test_split_size: 0.1
  dataset_type: "instruction"
  push_to_huggingface: true
  dataset_id: pauliusztin/llmtwin
  mock: false
```

### Step 6: Run with the mock response (safe, offline)

`constants.get_mocked_response(DatasetType.INSTRUCTION)` returns exactly three pairs:

```python
MOCKED_RESPONSE_INSTRUCT = """
[
    {"instruction": "<mocked generated instruction> 1", "answer": "<mocked generated answer> 1"},
    {"instruction": "<mocked generated instruction> 2", "answer": "<mocked generated answer> 2"},
    {"instruction": "<mocked generated instruction> 3", "answer": "<mocked generated answer> 3"}
]
"""
```

Because `FakeListLLM` always returns this string, every category produces exactly 3 samples per batch. This is how CI verifies the parsing and splitting logic without spending money.

---

## 📝 Exercise 1: Tune the Instruction Prompt

### Task

Improve sample quality and measure the difference.

1. Add a constraint that answers must be **between 80 and 150 words**.
2. Add a constraint that instructions must **start with a verb** ("Explain...", "Describe...").
3. Generate 20 samples with the mock, then with the real LLM.
4. Manually inspect 5 samples: are they self-contained and style-matched?

**Goal**: Feel how prompt constraints change the dataset. Strong constraints improve training quality but can cause parse failures if the model over-fits to formatting.

**What to watch**: after tightening constraints, re-check the `OutputParserException` rate. If batches start dropping, the constraint is fighting the model's output format, not improving it.

---

## 📝 Exercise 2: Measure Extract Boundaries

### Task

The `extract_substrings` size band (1000-2000 chars) directly controls sample count and context richness.

1. Call `chunk_document(content, 1000, 2000)` on one article and print each chunk's length.
2. Re-run with `min_length=500, max_length=1000` and with `min_length=2000, max_length=4000`.
3. Record how many extracts each setting produces.

```python
from llm_engineering.application.preprocessing.operations.chunking import chunk_document

content = docs[0].content
for lo, hi in [(500, 1000), (1000, 2000), (2000, 4000)]:
    chunks = chunk_document(content, lo, hi)
    print(f"({lo},{hi}) -> {len(chunks)} chunks, lengths={[len(c) for c in chunks]}")
```

4. Answer: does smaller mean more samples but less grounding? Does larger mean fewer, richer prompts? Which tradeoff fits a style-imitation Twin?

**Goal**: Understand that chunk size is a lever on both dataset size and per-sample context, and that the "right" value depends on how information-dense the corpus is.

---

## 🐛 Common Pitfalls

- **JSON drift**: the model sometimes wraps JSON in ```` ```json ```` fences. The base `PydanticOutputParser` prompt instructions and `temperature=0.7` reduce this, but the `OutputParserException` catch is what keeps the pipeline alive.
- **Whole-batch loss**: one malformed object discards all 24 responses in the batch. Watch the log line "Failed to parse the output JSON for a batch".
- **Cost**: each extract becomes one request, and prompts are sent 24 at a time. Estimate cost before a full run by counting `num_prompts × 5 pairs`. The book reports the full 3,335-pair run costs **less than $0.50** with GPT-4o-mini (p. 233).
- **Rate limits**: reduce the `batch(..., size=24)` value if you see 429 responses.
- **The `query_feature_store(after=...)` signature mismatch**: the pipeline passes `after`; the step declares no parameters. Verify before relying on `wait_for`.
- **Step-name typo**: the file and function are `generate_intruction_dataset` (missing `s`). Import by the exact name; do not assume `generate_instruction_dataset`.
- **Empty category**: a category with zero samples yields empty train and test splits. Downstream SFT must tolerate that.
- **Style leakage in instructions**: if the model writes "In the following extract...", the sample leaks its provenance. The prompt forbids it; spot-check outputs.
- **`temperature=0` mistake**: lowering to 0 yields near-duplicate instructions across extracts, wasting the five-per-extract multiplier.

---

## 🎓 Knowledge Check

1. **Why is `template_format="jinja2"` required?**
   - The templates contain literal JSON braces; Jinja2 substitutes `{{extract}}` without treating `{`/`}` as format fields.

2. **What does `extract_substrings` produce, and why 1000-2000 characters?**
   - Sentence-aware extracts of cleaned documents; the band keeps prompts focused, within token budget, and information-dense.

3. **What is the role of `ListPydanticOutputParser`?**
   - It extends the single-object parser to accept a JSON array of samples.

4. **Why `temperature=0.7` here rather than 0?**
   - To generate diverse instructions; determinism is not required for dataset synthesis.

5. **What two prompt rules make samples realistic?**
   - Instructions never mention the context/system, and answers imitate the source writing style.

6. **What schema does `to_huggingface()` emit for instruction data?**
   - `instruction` and `output` columns.

7. **How does the repo's parser differ from the book's `InstructionAnswerSet`?**
   - The repo validates a bare JSON array of `{instruction, answer}` objects with Pydantic; the book's class reads a wrapper key `"instruction_answer_pairs"` from JSON mode output and returns tuples.

8. **What is `OPENAI_MAX_TOKEN_WINDOW` for `gpt-4o-mini`, and why the 0.90 factor?**
   - 115,200 tokens (128,000 × 0.90); the headroom reserves space for the response and system prompt.

9. **What happens when one object in a batch fails validation?**
   - The whole batch of (up to) 24 responses raises `OutputParserException` and is dropped.

10. **Why is the train/test split done per category rather than globally?**
    - So a small category is not starved by a large one; each category contributes to both splits.

11. **Why does `max_tokens` differ between instruction and preference generation?**
    - Triples (three strings) are longer than pairs (two strings): 2000 vs 1200.

12. **What does `mock=True` change, and why does CI use it?**
    - It swaps `ChatOpenAI` for `FakeListLLM` with a canned response, so the pipeline runs with no API key and no spend.

13. **Name the three quality dimensions the book uses to judge instruction data.**
    - Accuracy, diversity, complexity.

14. **What is backtranslation in this context?**
    - Treating a text chunk as the answer and asking the LLM to generate the matching instruction, then rephrasing to imitate the author's style.

15. **What is the real signature problem in this pipeline?**
    - `generate_datasets` calls `query_feature_store(after=wait_for)`, but the step `def query_feature_store()` declares no parameters.

---

## 📖 Glossary

- **Backtranslation** — generating an instruction from a provided answer (the reverse of the usual direction).
- **BPE tokenizer** — byte-pair-encoding tokenizer; `tiktoken.encoding_for_model` returns the one for a given OpenAI model.
- **Category** — `DataCategory` enum value (`ARTICLES`, `POSTS`, `REPOSITORIES`) used to group documents and split datasets.
- **Extract** — a 1000-2000 character, sentence-aligned slice of a cleaned document.
- **JSON mode** — OpenAI's `response_format={"type": "json_object"}` forcing valid JSON output.
- **LLM Twin** — a model fine-tuned to imitate a specific author's writing style and voice.
- **Template method** — design pattern where a base class fixes the algorithm skeleton and subclasses fill in the varying steps (`DatasetGenerator.generate`).
- **VectorBaseDocument** — the project's Pydantic base for documents that can be stored in the vector DB.

---

## 🔗 Next Session

**Session 3.2**: Preference Dataset for DPO

We generate chosen/rejected triples, filter low-quality samples, and prepare the DPO dataset.

---

## 📚 Additional Resources

- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [tiktoken](https://github.com/openai/tiktoken)
- [sklearn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Open-Orca/OpenOrca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 2.2, 2.3

**Outcome**: You can generate, parse, split, and publish an instruction dataset grounded in your own corpus, and explain every design choice against the book's three quality dimensions.
