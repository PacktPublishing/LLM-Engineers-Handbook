# Session 7.3: Model Evaluation

## 🎯 Learning Objectives

By the end of this session, you will:
- Generate test-set answers with vLLM
- Implement an LLM-as-a-judge evaluation (accuracy + style)
- Run large-scale evaluation in parallel batches
- Compare SFT, DPO, and an Instruct baseline
- Understand pointwise versus pairwise judging and their biases
- Estimate the cost and failure modes of an LLM-judge pipeline
- Trace how the evaluation is orchestrated from ZenML down to the SageMaker processing job

> This session is the hands-on core of **Book Chapter 7: Evaluating LLMs** (pp. 304-316). The wider theory (general/domain/task-specific benchmarks, Ragas, ARES) lives in `session_7.4_rag_evaluation.md`.

---

## 🏗️ Architecture Overview

```
test split of llmtwin dataset
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ generate_answers(model_id, dataset)                               │
│   vLLM: LLM(model_id, max_model_len=2048)                         │
│   SamplingParams(temperature=0.8, top_p=0.95, min_p=0.05, 2048)   │
│   → push "{workspace}/{model}-results" to the Hub                 │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ evaluate_answers(model_id)                                        │
│   batch samples, ThreadPoolExecutor(max_workers=10)               │
│   evaluate_answer(): gpt-4o-mini judge → accuracy + style scores  │
│   → add accuracy/style columns, re-push to the Hub                │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ analyze: mean accuracy / mean style per model                     │
└──────────────────────────────────────────────────────────────────┘
```

### The two-stage split

The pipeline deliberately separates **generation** from **judging**:

1. **Generation is local and GPU-bound.** vLLM loads the candidate model and produces answers. This is the only expensive-VRAM step.
2. **Judging is remote and network-bound.** Each answer is sent to `gpt-4o-mini`. No local GPU is used.

Splitting the stages means a judge failure does not require re-running generation, and generation for all three models can complete before any judge call is made. The intermediate results are pushed to the Hugging Face Hub after each stage, so the pipeline is resumable.

```
Stage 1 (GPU)                  Stage 2 (network)              Stage 3 (CPU)
generate_answers ──push──► evaluate_answers ──push──► analyze (mean scores)
   per model                    per model                     per model
```

**Models compared**:

```python
model_ids = [
    check_if_huggingface_model_exists(f"{MODEL_HUGGINGFACE_WORKSPACE}/TwinLlama-3.1-8B", "mlabonne/TwinLlama-3.1-8B"),
    check_if_huggingface_model_exists(
        f"{MODEL_HUGGINGFACE_WORKSPACE}/TwinLlama-3.1-8B-DPO", "mlabonne/TwinLlama-3.1-8B-DPO"
    ),
    "meta-llama/Llama-3.1-8B-Instruct",
]
```

SFT vs DPO answers the alignment question. The Instruct baseline anchors absolute quality.

### Why these three models

| Model | Question it answers |
|-------|---------------------|
| `TwinLlama-3.1-8B` (SFT) | Did supervised fine-tuning teach the twin the author's knowledge and task format? |
| `TwinLlama-3.1-8B-DPO` | Did preference alignment improve tone/style without hurting accuracy? |
| `Llama-3.1-8B-Instruct` | What is the absolute ceiling a heavily post-trained general model reaches? |

Without the Instruct baseline you cannot tell whether a score of 2.4 is good or bad. Without DPO you cannot isolate the effect of preference alignment. The three-way comparison is the minimum needed to make a claim about fine-tuning.

---

## 📁 Key Files Explained

The evaluation lives in four places that form a call chain:

| Layer | File | Responsibility |
|-------|------|----------------|
| Pipeline | `pipelines/evaluating.py` | ZenML DAG declaration |
| Step | `steps/evaluating/evaluate.py` | ZenML step wrapper |
| Job launcher | `llm_engineering/model/evaluation/sagemaker.py` | Builds and runs the SageMaker processing job |
| Worker | `llm_engineering/model/evaluation/evaluate.py` | Runs inside the container: generate + judge + analyze |

```
tools.run --run-evaluation
        │
        ▼
pipelines/evaluating.py  @pipeline evaluating(is_dummy)
        │
        ▼
steps/evaluating/evaluate.py  @step evaluate(is_dummy)
        │
        ▼
model/evaluation/sagemaker.py  run_evaluation_on_sagemaker(is_dummy)
        │  HuggingFaceProcessor(...).run(code="evaluate.py", source_dir=...)
        ▼
model/evaluation/evaluate.py  (inside the SageMaker container)
        generate_answers() → evaluate_answers() → analyze
```

### 1. `evaluate.py` - Generation with vLLM

```python
# llm_engineering/model/evaluation/evaluate.py
def generate_answers(model_id: str, dataset_name: str):
    def format(sample):
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n".format(
            sample["instruction"]
        )

    dataset = load_dataset(dataset_name, split="test")
    if IS_DUMMY:
        try:
            dataset = dataset.select(range(10))
        except Exception:
            print("Dummy mode active. Failed to trim the dataset to 10 samples.")
    print(f"Dataset size: {len(dataset)}")
    dataset = dataset.map(lambda sample: {"prompt": format(sample)})

    print(f"Generating answers for {model_id}")
    llm = LLM(model=model_id, max_model_len=2048)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_p=0.05, max_tokens=2048)
    outputs = llm.generate(dataset["prompt"], sampling_params)

    answers = [output.outputs[0].text for output in outputs]
    dataset = dataset.add_column("answers", answers)

    print(f"Uploading results for {model_id}")
    dataset.push_to_hub(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results")
    gc.collect()

    return dataset
```

**Key Concepts**:
- **vLLM** gives high-throughput batched generation via PagedAttention. It is the standard tool for evaluating many prompts quickly.
- **The `format` prompt matches the Alpaca inference format** used elsewhere, so the model sees the same interface it was trained and served with.
- **`split="test"`** uses the held-out split created during dataset generation (Session 3.1) - evaluation never touches training data.
- **`gc.collect()`** frees GPU memory before loading the next model, which matters when evaluating three models sequentially.
- **Results are pushed to the Hub** under `{workspace}/{model}-results`, so evaluation is versioned and shareable.

**Sampling parameters**: `temperature=0.8`, `top_p=0.95`, `min_p=0.05` produce varied but coherent generations - appropriate when you want to judge typical model output, not the single most likely answer.

> **Repo vs book note (accuracy)**: this repository uses `max_model_len=2048` and `max_tokens=2048`. The printed book text shows `4096` for both. The docs track the repository because that is the code you actually run. `IS_DUMMY` is read with `os.environ.get("IS_DUMMY", False)`, so the string `"True"` from SageMaker is truthy and the trim can be caught by a `try/except`.

**Why vLLM over `transformers.generate`**:

| Concern | `transformers.generate` | vLLM |
|---------|-------------------------|------|
| Batching | manual, padded | continuous batching, no padding waste |
| KV cache | naive, contiguous | PagedAttention blocks |
| Throughput | baseline | several times higher |
| API surface | full model control | inference-only, simpler |
| Fit for evaluation | fine for 1 prompt | ideal for hundreds of prompts |

The evaluation needs hundreds of prompts across three models. That is exactly the workload vLLM is built for.

### 2. The Judge Prompt

```python
def evaluate_answer(instruction: str, answer: str, client: OpenAI) -> dict:
    prompt = f"""You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:
1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

Accuracy scale:
1 (Poor): Contains factual errors or misleading information
2 (Good): Mostly accurate with minor errors or omissions
3 (Excellent): Highly accurate and comprehensive

Style scale:
1 (Poor): Too formal, uses some overly complex words
2 (Good): Good balance of technical content and accessibility, but still uses formal words and expressions
3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.

Instruction: {instruction}
Answer: {answer}

Provide your evaluation in JSON format with the following structure:
{{
    "accuracy": {{ "analysis": "...", "score": 0 }},
    "style": {{ "analysis": "...", "score": 0 }}
}}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who evaluates answers based on accuracy and style. Provide your response in JSON format with a short analysis and score for each criterion.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        temperature=0.9,
    )
    return json.loads(completion.choices[0].message.content)
```

**Key Concepts**:
- **Two orthogonal criteria**: accuracy (factual correctness) and style (accessible blog/social tone). The project's twin is judged on style as much as correctness.
- **Anchored 1-3 scales with explicit descriptions** reduce judge variance. Each score level has a concrete definition.
- **Worked style examples (bad vs excellent)** are the strongest lever for consistency; the judge copies the distinction.
- **`response_format={"type": "json_object"}`** forces valid JSON, so no fence-stripping is needed.
- **`temperature=0.9`** adds judge variability. This is intentional for diversity but means scores are noisy; report means over the dataset, not single judgments.

> **Repo vs book note (accuracy)**: the repository passes `temperature=0.9`; the printed book shows `0.8`. The repository is the source of truth for the code shipped here.

**The prompt's design anatomy**:

```
┌─────────────────────────────────────────────────────────────┐
│ Role framing      "You are an expert judge..."              │
│ Criteria list     1. Accuracy   2. Style                    │
│ Anchor scales     1/2/3 with a definition per level         │
│ Few-shot examples bad style vs excellent style              │
│ Task payload      Instruction: {...} Answer: {...}          │
│ Output contract   JSON schema with analysis + score         │
└─────────────────────────────────────────────────────────────┘
```

Every one of these blocks earns its place. Remove the anchors and scores drift; remove the examples and "formal" is interpreted inconsistently; remove the JSON contract and parsing fails.

### 3. Parallel Batch Evaluation

```python
def evaluate_batch(batch, start_index):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return [(i, evaluate_answer(instr, ans, client)) for i, (instr, ans) in enumerate(batch, start=start_index)]


def evaluate_answers(model_id: str, num_threads: int = 10, batch_size: int = 5) -> Dataset:
    dataset = load_dataset(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results", split="all")

    batches = [
        (i, list(zip(dataset["instruction"][i : i + batch_size], dataset["answers"][i : i + batch_size], strict=False)))
        for i in range(0, len(dataset), batch_size)
    ]

    evaluations = [None] * len(dataset)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(evaluate_batch, batch, start_index) for start_index, batch in batches]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            for index, evaluation in future.result():
                evaluations[index] = evaluation
    ...
```

**Key Concepts**:
- **Indexed batches**: each batch carries its start index, and results are placed back at the original index. This preserves alignment even though futures complete out of order.
- **`ThreadPoolExecutor(max_workers=10)`** parallelizes the judge API calls. Ten workers balances throughput against OpenAI rate limits.
- **`tqdm`** shows progress, which matters for long evaluations.
- **Error tolerance in post-processing**: any malformed evaluation appends `None` for accuracy and style, keeping the columns aligned with the dataset.

**Why threads, not processes**: the work is I/O-bound (waiting on the OpenAI HTTP API), not CPU-bound. Python threads release the GIL during network waits, so ten threads generate ten overlapping requests. Processes would add pickling overhead for no gain.

**Order preservation, worked example**:

```
dataset:   [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]
batches:   (0, [i0..i4])                (5, [i5..i9])
            |                            |
            ▼ (finishes later)           ▼ (finishes first)
results:   [(0,d0)...(4,d4)]            [(5,d5)...(9,d9)]
            |                            |
            └────────► evaluations[0..9] ◄────────┘
              each tuple writes itself back at its index
```

Without the `start_index` carried in each batch, the second batch's results would overwrite the first batch's slot. The index is the alignment mechanism.

**Score aggregation**:

```python
def evaluate_answers(...):
    ...
    for evaluation in dataset["evaluation"]:
        try:
            eval_dict = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
            accuracy_scores.append(eval_dict["accuracy"]["score"])
            style_scores.append(eval_dict["style"]["score"])
        except (json.JSONDecodeError, KeyError, TypeError):
            accuracy_scores.append(None)
            style_scores.append(None)

    dataset = dataset.add_column("accuracy", accuracy_scores)
    dataset = dataset.add_column("style", style_scores)
    dataset.push_to_hub(...)
```

`None` entries keep alignment; when averaging, skip them or treat missing as excluded.

**Why keep the raw `evaluation` column**: it stores the judge's *analysis* text, not just the score. When a model scores surprisingly low, you can read the reasoning to distinguish a real defect from a judge mistake. Scores are cheap to aggregate; analyses are what you inspect when the aggregate looks wrong.

### 4. Running on SageMaker

```python
# llm_engineering/model/evaluation/sagemaker.py
def run_evaluation_on_sagemaker(is_dummy: bool = True) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.OPENAI_API_KEY, "OpenAI API key is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    env = {
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "DATASET_HUGGINGFACE_WORKSPACE": huggingface_user,
        "MODEL_HUGGINGFACE_WORKSPACE": huggingface_user,
    }
    if is_dummy:
        env["IS_DUMMY"] = "True"

    hfp = HuggingFaceProcessor(
        role=settings.AWS_ARN_ROLE,
        instance_count=1,
        instance_type="ml.g5.2xlarge",
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        base_job_name="evaluate-llm-twin",
        env=env,
    )
    hfp.run(code="evaluate.py", source_dir=str(evaluation_dir))
```

**Key Concepts**:
- **`HuggingFaceProcessor`** runs a SageMaker *processing* job (not training) - the right primitive for batch generation + judging.
- **Env vars carry the three credentials**: HF token (model/dataset access), OpenAI key (judge), and the workspaces.
- **`is_dummy`** sets `IS_DUMMY=True` in the container, which `evaluate.py` reads to trim the dataset to 10 samples.
- **The launcher discovers the HF user** via `HfApi().whoami(token=...)` and uses that name as both workspace variables, so the job reads and writes under the caller's own Hub account.

**Why a processing job and not an endpoint**: evaluation is a batch, offline, fire-and-forget workload. An endpoint is long-lived and billed by the hour; a processing job bills only for the run and tears itself down. The `ml.g5.2xlarge` (A10G) holds an 8B model after quantization comfortably.

**Instance economics**:

| Instance | GPU | VRAM | Typical use |
|----------|-----|------|-------------|
| `ml.g5.2xlarge` | 1x A10G | 24 GB | 8B generation with vLLM |
| `ml.g5.12xlarge` | 4x A10G | 96 GB | larger batch / bigger models |
| `ml.p4d.24xlarge` | 8x A100 | 320 GB | multi-GPU training |

The project chose the smallest single-GPU option that fits the 8B model. Scaling up the instance is unnecessary because the workload is one model at a time.

### 5. The ZenML Pipeline

```python
# pipelines/evaluating.py
from zenml import pipeline

from steps import evaluating as evaluating_steps


@pipeline
def evaluating(is_dummy: bool = False) -> None:
    evaluating_steps.evaluate(is_dummy=is_dummy)
```

```python
# steps/evaluating/evaluate.py
from zenml import step

from llm_engineering.model.evaluation.sagemaker import run_evaluation_on_sagemaker


@step
def evaluate(is_dummy: bool = False) -> None:
    run_evaluation_on_sagemaker(is_dummy=is_dummy)
```

Run with:

```bash
python -m tools.run --run-evaluation
```

**Why wrap a single call in ZenML**: the pipeline gives you run history, caching, artifact tracking, and a consistent CLI (`tools.run`) matching the training and feature pipelines. When the project later adds RAG evaluation as another step, it slots into the same DAG. The indirection costs one file today and buys orchestration consistency tomorrow.

---

## 🔬 Deep Dive: LLM-as-a-Judge

### Pointwise vs pairwise

| Mode | Input | Output | Project use |
|------|-------|--------|-------------|
| Pointwise | one answer | absolute score | yes (accuracy/style 1-3) |
| Pairwise | two answers | which is better | not used here |

Pointwise scales to many models and one-dimensional criteria, and it is what `evaluate_answer` implements. Pairwise is more reliable for close comparisons but requires pairing every model with every other.

### Known biases and mitigations

- **Position bias** (pairwise only): the first answer wins more often. Not applicable to pointwise.
- **Verbosity bias**: longer answers tend to score higher. The project controls for this by using concise style examples and a style criterion.
- **Self-preference**: a judge may favor answers resembling its own outputs. Using `gpt-4o-mini` as judge for Llama models partially avoids this.
- **Noise from `temperature=0.9`**: report the **mean over the dataset**, and treat single-point differences between models as non-significant.
- **Score compression**: judges cluster on 2/3 and rarely emit 1. Treat the scale as ordinal, not interval - a jump from 2.0 to 2.5 is not the same "distance" as 1.0 to 1.5.

### Why accuracy *and* style

An LLM Twin must both be correct and *sound like the author*. A model that is accurate but formal fails the style criterion; one that is stylish but wrong fails accuracy. The two scores together capture the twin objective.

```
        high accuracy
             ▲
             │   Instruct baseline      Ideal (unreached)
             │   (correct, verbose)      (correct, natural)
             │
  ───────────┼──────────────────────────► high style
             │
             │   Bad model
             │   (wrong, rambling)
```

The goal is the top-right quadrant. The Instruct baseline sits top-left (correct but verbose). DPO is meant to push right without falling down.

---

## 📊 Worked Example: Interpreting the Published Results

The book reports these averages over the test set:

```
TwinLlama-3.1-8B        - Accuracy: 2.45   Style: 2.04
TwinLlama-3.1-8B-DPO    - Accuracy: 2.46   Style: 2.12
Llama-3.1-8B-Instruct   - Accuracy: 2.62   Style: 1.86
```

How to read this table:

| Observation | Interpretation | Caveat |
|-------------|----------------|--------|
| Instruct wins accuracy (2.62) | More post-training data (10M+ samples vs 13k) yields broader factual coverage | The gap is 0.16 on a 3-point scale - small |
| DPO wins style (2.12) | Preference alignment successfully taught natural tone | Improvement over SFT is only 0.08 |
| SFT and DPO tie on accuracy | Alignment did not damage factual content | Within judge noise |
| Instruct loses style (1.86) | General models trend formal and verbose | The twin's whole purpose is beating this |

**The takeaway**: fine-tuning did not beat a frontier-quality instruct model on knowledge, but it did beat it on style - which is the specific axis the LLM Twin optimizes. This is an honest, narrow win, and the docs should present it that way rather than claiming overall superiority.

**Reading a single answer**: the book shows the `algorithm bias` instruction. SFT and DPO answers are near-identical and concise; Instruct is much longer and lists many examples. The judge scored Instruct's style 2 with the note that phrases like "unintended or inherent bias" could be simplified. This is the verbosity+formality pattern caught in a single example.

---

## 🔬 Deep Dive: How vLLM Produces Answers

Understanding the generation engine explains the performance numbers and the VRAM constraint.

### PagedAttention in one paragraph

A naive inference server stores each sequence's key/value cache in one contiguous block. Because output lengths are unknown up front, it must reserve worst-case space, wasting 60-80% of VRAM to fragmentation. PagedAttention splits the cache into fixed-size blocks addressed by a page table, exactly like an OS virtual-memory system. Memory is allocated on demand, fragments are eliminated, and sequences that share a prompt prefix can share blocks.

```
Naive KV cache                 PagedAttention
┌───────────────┐              ┌────┬────┬────┬────┐
│ reserved, big │              │ b0 │ b1 │ b2 │ b3 │  ← blocks filled on demand
│ most unused   │              └────┴────┴────┴────┘
└───────────────┘                page table maps tokens → blocks
```

### Continuous batching

The server does not wait for a whole batch to finish. When one sequence emits its stop token, it is evicted and a queued request is admitted into the freed slot. The GPU stays saturated even when request lengths differ wildly.

```
step 1: [A][B][C][D]        all busy
step 2: [A][B][C][D]        C finishes
step 3: [A][B][ ][D]        evict C
step 4: [A][B][E][D]        admit E  ← batch stays full
```

### Why `max_model_len=2048` matters

`max_model_len` bounds prompt + generated tokens. vLLM pre-allocates block tables for the maximum, so raising it raises memory pressure. The repo uses 2048, which fits an 8B model on the `ml.g5.2xlarge` A10G comfortably and matches the project's instruction lengths. A larger window is not free: it trades VRAM that could otherwise hold more concurrent sequences.

### Sampling parameters, decoded

| Parameter | Repo value | Effect | Failure if wrong |
|-----------|-----------|--------|------------------|
| `temperature` | 0.8 | scales the logit distribution; higher = more random | 0 → greedy, repetitive; too high → incoherent |
| `top_p` | 0.95 | nucleus: keep the smallest set of tokens whose probability sums to 0.95 | too low → bland; 1.0 → no truncation |
| `min_p` | 0.05 | drop tokens below 5% of the top token's probability | removes long-tail junk |
| `max_tokens` | 2048 | hard cap on generated length | too low → truncated answers; too high → long tail cost |

The combination (`temperature=0.8`, `top_p=0.95`, `min_p=0.05`) is a standard "varied but coherent" recipe. For evaluation you *want* typicality, so sampling rather than greedy decoding is correct: greedy would measure the model's single best path, not its expected output under deployment sampling.

### Worked VRAM estimate

```
8B params × 2 bytes (FP16)        ≈ 16 GB   weights
KV cache at 2048 tokens           ≈  2 GB   (32 layers, 32 heads, dim 128)
activations + vLLM overhead        ≈  2-4 GB
────────────────────────────────────────────
total                              ≈ 20-22 GB  → overflow of a 16 GB RTX 5000
```

This is why the project evaluates on `ml.g5.2xlarge` (24 GB) rather than locally. On the RTX 5000 you must quantize or shard; the repo's own workflow assumes the cloud GPU.

---

## 🧪 Edge Cases and Failure Modes

| Failure | Symptom | Root cause | Detection | Fix |
|---------|---------|-----------|-----------|-----|
| Wrong model evaluated | One model's scores look like another's | `check_if_huggingface_model_exists` silently fell back to a public default | Read the printed `Defaulting to ...` line | Set the workspace env var to your own account |
| All answers empty | `answers` column is blank strings | Wrong chat template for that model | Inspect raw answers | Use the model's own template |
| Judge returns non-schema JSON | `None` accuracy/style for many rows | Prompt or model changed | Count `None` rate | Pin the judge model; re-check the prompt |
| OOM during generation | Container killed after first model | 8B in FP16 does not fit / KV cache growth | CloudWatch logs | Quantize; reduce `max_model_len`; one model at a time |
| `TypeError` in final analysis | Crash at the end after successful scoring | `sum()` over a column containing `None` | Traceback names `__main__` | Filter `None` before averaging |
| Inflated style for verbose answers | Instruct scores high on style despite verbosity | Judge verbosity bias | Read the `analysis` text | Tighten the style examples; add a conciseness criterion |
| Model contamination | Fine-tuned model beats baseline by a large margin | Test set leaked into training | Compare test vs a fresh held-out set | Re-split; verify no overlap |

### Design alternatives considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| Greedy decoding for generation | Deterministic, reproducible | Measures best-case, not typical | Rejected - evaluation should sample |
| Human evaluation | Gold-standard quality | Slow, expensive, subjective | Rejected as primary; useful for spot checks |
| Pairwise-only judging | Best for close comparisons | O(n^2) model pairs; position bias | Rejected as primary; used in the exercise |
| A dedicated reward model as judge | Cheap at scale, consistent | Needs training; can be gamed | Noted as an ARES-style future option |
| Ragas/ARES for a model-only eval | Measures groundedness | Requires a retriever/context | Reserved for the RAG system (Session 7.4) |

### Cost model

Judge calls are the dominant variable cost. Estimate before running:

```
calls = samples × models
cost  ≈ calls × (avg_prompt_tokens × in_price + avg_output_tokens × out_price)
```

With a 240-sample test set and 3 models, that is 720 calls. At a few cents per thousand calls for `gpt-4o-mini`, the judge is cheap relative to the GPU hours; the real budget risk is a retry loop that re-scores without caching. The Hub persistence after each stage prevents re-generation after a judge failure.

---

## 🛠️ Hands-On: Run a Small Evaluation

### Step 1: Dummy end-to-end

```bash
python -m tools.run --run-evaluation
# or directly:
python llm_engineering/model/evaluation/evaluate.py   # with IS_DUMMY=True
```

### Step 2: In a local environment (GPU)

```bash
export OPENAI_API_KEY=...
export DATASET_HUGGINGFACE_WORKSPACE=<you>
export MODEL_HUGGINGFACE_WORKSPACE=<you>
export IS_DUMMY=True
python llm_engineering/model/evaluation/evaluate.py
```

### Step 3: Inspect results on the Hub

Open `{workspace}/TwinLlama-3.1-8B-DPO-results` and browse the `answers`, `evaluation`, `accuracy`, and `style` columns.

### Step 4: Compare means

```python
from datasets import load_dataset

for m in ["TwinLlama-3.1-8B", "TwinLlama-3.1-8B-DPO"]:
    ds = load_dataset("<workspace>/" + m + "-results", split="all")
    acc = [a for a in ds["accuracy"] if a is not None]
    sty = [s for s in ds["style"] if s is not None]
    print(m, "accuracy", sum(acc) / len(acc), "style", sum(sty) / len(sty))
```

### Step 5: Quantify judge noise

Re-run the judge on the same answers twice and correlate the two score vectors. If the correlation is weak, you need more samples or a lower-temperature judge before comparing models.

---

## 📝 Exercise: Add a Pairwise Comparison

### Task

Implement a pairwise judge between two models' answers to the same instruction.

1. Load both `-results` datasets.
2. For each instruction, ask `gpt-4o-mini` which answer is better, **randomizing which is A/B** and tracking the mapping.
3. Report a win rate for DPO vs SFT.
4. Compare the pairwise result with the pointwise means.

**Goal**: Experience why pairwise judging is preferred for close comparisons and how to defend against position bias by randomizing order.

### Second Exercise: Build a Confidence Interval

The pointwise means above differ by 0.08 on style. Is that real or noise?

1. Collect the per-sample style scores for SFT and DPO into two lists.
2. Bootstrap: resample each list with replacement 1,000 times, compute the mean difference each time.
3. Report the 2.5th and 97.5th percentiles as a 95% confidence interval.
4. State whether the interval excludes zero. If it does not, the DPO style win is not statistically distinguishable from noise.

```python
import random

def bootstrap_diff(a, b, n=1000):
    diffs = []
    for _ in range(n):
        ra = [random.choice(a) for _ in a]
        rb = [random.choice(b) for _ in b]
        diffs.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]
```

**Goal**: Learn to treat judge scores as noisy measurements with error bars, not ground truth.

---

## 🛠️ Case Study: Diagnosing a Regression

Suppose a new SFT checkpoint scores **2.71 accuracy / 1.90 style**, up from **2.45 / 2.04**. Accuracy rose but style collapsed. Walk the diagnosis in order:

```
1. Read the raw answers for 5 low-style instructions.
      → Are they longer? More formal? Different template?
2. Check the chat template.
      → A template mismatch depresses style but can raise apparent "accuracy".
3. Check for contamination.
      → Accuracy +0.26 is plausible; +0.26 with style -0.14 suggests a data shift.
4. Compare token lengths.
      → Verbosity bias means longer answers inflate accuracy and depress style here.
5. Re-run the judge twice.
      → If the second run gives 1.95 style, most of the drop is judge noise.
6. Inspect the DPO data.
      → Preference pairs may have drifted toward formal completions.
```

Only after ruling out template, contamination, and noise should you conclude the model actually regressed on style. The point is procedural: a score is a hypothesis, and the raw answers plus a re-run are the falsification tools.

### Rubric engineering checklist

| Rubric element | Why it matters | Failure if omitted |
|----------------|----------------|--------------------|
| Explicit criteria | Defines what is scored | Judge invents its own axes |
| Per-level anchors | Reduces interpretation variance | Scores drift between runs |
| Worked examples | Teaches the boundary case | "Formal" is applied inconsistently |
| Output schema | Guarantees parseability | Fences and prose break JSON parsing |
| Ground-truth option | Anchors accuracy to a reference | Judge rates confidently-wrong answers high |
| Concise length handling | Counters verbosity bias | Long answers win on length alone |

### Comparing scoring approaches

| Approach | Determinism | Cost | Scale | Best fit |
|----------|-------------|------|-------|----------|
| Exact match / F1 (reference metrics) | high | none | unlimited | extractive QA, classification |
| Pointwise LLM judge | low | per call | high | open-ended quality (this project) |
| Pairwise LLM judge | low | per pair | medium | close model comparisons |
| Reward model | high | training cost | very high | production filtering |
| Human review | medium | highest | low | calibration and spot checks |

The project's pointwise judge sits in the middle: scalable, cheap, and noisy. That is the right trade for an open-ended creative task where no single gold answer exists.

---

## 🐛 Common Pitfalls

- **vLLM VRAM**: loading an 8B model for generation needs substantial VRAM; the batch is token-bounded, but `max_model_len=2048` still needs headroom. On a 16 GB card, evaluate one model at a time and `gc.collect()` between them.
- **Judge cost**: one judge call per sample. Estimate `samples × models` before a full run. At test-set size, this is hundreds of calls per model.
- **Missing model/data**: the `check_if_huggingface_*_exists` fallbacks keep the script running with public defaults, which can silently evaluate the wrong model. Read the printed warnings.
- **Unkilled nulls**: if all evaluations failed for a sample, `accuracy`/`style` are `None`; ensure averages skip them. The final analysis loop in `__main__` does **not** skip `None` - it would raise `TypeError`. Guard it if any judge call fails.
- **Chat-template mismatch**: every model is fed the Alpaca-style template, including `Llama-3.1-8B-Instruct`, which was trained on a different template. The book accepts this for simplicity, but a wrong template can quietly depress scores for one model only. Review raw answers to catch it.
- **Judge JSON drift**: `response_format={"type": "json_object"}` guarantees valid JSON but not the schema. A missing `style` key raises `KeyError`, caught and turned into a `None`. Watch the `None` rate; a high rate means the prompt or model changed.
- **Comparing single judgments**: `temperature=0.9` makes one score unreliable. Never compare models on a single instruction.

---

## 🎓 Knowledge Check

1. **What two criteria does the judge score?**
   - Answer: accuracy (factual correctness) and style (accessible blog/social tone), each 1-3.

2. **Why use vLLM for generation?**
   - Answer: High-throughput batched inference via PagedAttention.

3. **How are out-of-order batch results re-aligned?**
   - Answer: Each batch carries its start index; results are written back at the original index.

4. **Why does the post-processing append `None` on parse errors?**
   - Answer: To keep accuracy/style columns aligned with the dataset.

5. **Which model is the absolute-quality baseline?**
   - Answer: `meta-llama/Llama-3.1-8B-Instruct`.

6. **What is the main bias of pointwise judging, and how is it mitigated here?**
   - Answer: Verbosity/length bias; mitigated with a style criterion and concise style examples - and by averaging over the dataset.

7. **Why threads instead of processes for the judge calls?**
   - Answer: The work is I/O-bound (waiting on the API), so threads overlap the network wait without pickling overhead.

8. **Why is the SageMaker primitive a processing job rather than an endpoint?**
   - Answer: Evaluation is a batch, offline workload; a processing job bills only for the run instead of an always-on endpoint.

9. **What does the `evaluation` column store beyond the score?**
   - Answer: The judge's free-text analysis for each criterion, used to diagnose surprising scores.

10. **Why is `max_model_len=2048` a VRAM concern?**
    - Answer: The KV cache grows with sequence length; a longer window needs more VRAM and competes with the weights.

11. **What is self-preference bias and how is it partly avoided?**
    - Answer: A judge favors outputs resembling its own; using `gpt-4o-mini` for Llama models partially avoids it.

12. **Why can the final analysis loop crash?**
    - Answer: It sums the score column directly; a `None` from a failed judge call raises `TypeError` unless skipped.

13. **What did DPO actually buy, per the published averages?**
    - Answer: A small style improvement (2.04 to 2.12) at equal accuracy - the specific axis the twin targets.

14. **Why report means instead of per-item scores?**
    - Answer: Judge temperature injects per-item noise; only aggregate means over the dataset are stable enough to compare models.

---

## 📖 Glossary

- **LLM-as-a-judge**: using a language model to score another model's outputs against a rubric.
- **Pointwise judging**: scoring each answer independently on an absolute scale.
- **Pairwise judging**: comparing two answers and picking a winner.
- **Position bias**: a pairwise judge's tendency to prefer whichever answer appears first.
- **Verbosity bias**: awarding higher scores to longer answers.
- **Self-preference bias**: a judge favoring text that resembles its own generations.
- **PagedAttention**: vLLM's block-based KV-cache memory management, enabling high-throughput batching.
- **SamplingParams**: vLLM's decoding configuration (temperature, top_p, min_p, max_tokens).
- **Anchored scale**: a rating scale where each level has an explicit written definition.
- **Processing job**: a SageMaker job type for batch, non-training workloads.
- **Bootstrap**: resampling with replacement to estimate a statistic's confidence interval.

---

## 🔗 Next Session

**Session 8.1**: Docker & Local Infrastructure - see `session_8.1_docker.md`.

We containerize the application and study the local service topology.

---

## 📚 Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [LLM-as-a-Judge (Zheng et al.)](https://arxiv.org/abs/2306.05685)
- [SageMaker Hugging Face Processor](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/)
- [Ragas metrics](https://docs.ragas.io/)
- [Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [Length-Controlled AlpacaEval (Dubois et al., 2024)](https://arxiv.org/abs/2404.04475)
- [Using LLM-as-a-judge (Hugging Face Cookbook)](https://huggingface.co/learn/cookbook/en/llm_judge)

---

## 📎 References

- Lianmin Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685, June 2023.
- Aymeric Roucher. "Using LLM-as-a-judge for an automated and versatile evaluation - Hugging Face Open-Source AI Cookbook." huggingface.co.
- LangChain. "Aligning LLM-as-a-Judge with Human Preferences." blog.langchain.dev, June 26, 2024.
- Yann Dubois et al. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." arXiv:2404.04475, April 2024.
- Iusztin, P. and Labonne, M. "LLM Engineer's Handbook." Packt, 2024 - Chapter 7 (pp. 304-316).

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Sessions 3.1, 5.1, 5.2, 5.3

**Outcome**: You can generate, judge, and compare model outputs with an LLM-as-a-judge pipeline, quantify the judge's noise, and reason about its biases.
