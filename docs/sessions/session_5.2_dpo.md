# Session 5.2: Direct Preference Optimization (DPO)

## 🎯 Learning Objectives

By the end of this session, you will:
- Explain the DPO objective and the role of `beta` as a KL anchor
- Derive the DPO loss by hand for a worked numeric example
- Format preference triples for `DPOTrainer`
- Configure DPO to run on top of the SFT checkpoint
- Compare SFT-only output against SFT+DPO output
- Understand why DPO uses a much smaller learning rate than SFT
- Read the DPO metrics Comet ML logs and diagnose a bad run
- Decide between LoRA and QLoRA for DPO on a 16 GB consumer GPU

---

## 🏗️ Architecture Overview

```
SFT checkpoint (Session 5.1)
   {workspace}/TwinLlama-3.1-8B
        │  (this is π_ref, the frozen reference)
        ▼
┌────────────────────────────────────────────────────────────────┐
│                     finetune(finetuning_type="dpo")            │
│                                                                 │
│  load_model(sft_checkpoint, load_in_4bit=...)                  │
│  PatchDPOTrainer()                                             │
│        │                                                        │
│  format_samples_dpo()                                          │
│        │  prompt  = alpaca_template.format(prompt, "")         │
│        │  chosen  = chosen + EOS                               │
│        │  rejected= rejected + EOS                             │
│        ▼                                                        │
│  DPOTrainer(beta=0.5, ref_model=None,                          │
│             max_length=1024, max_prompt_length=1024)           │
│        │  DPOConfig(lr=2e-6, epochs=1, optim=adamw_8bit)       │
│        ▼                                                        │
│  save_pretrained_merged  →  TwinLlama-3.1-8B-DPO               │
└────────────────────────────────────────────────────────────────┘
```

### Where the data comes from

```
Session 3.2 (Preference Dataset)
  PreferenceDatasetSample(instruction, rejected, chosen)
        │  PreferenceDataset.to_huggingface()
        ▼
  HF dataset with columns: prompt, rejected, chosen
        │  pushed to {workspace}/llmtwin-dpo
        ▼
  load_dataset("{workspace}/llmtwin-dpo", split="train")
```

The three-column schema is not arbitrary: it is exactly what
`PreferenceDataset.to_huggingface()` emits in `domain/dataset.py:95`, so the
training script can consume the exported dataset with no renaming step.

### The dependency chain

```
Chapter 5 SFT  ──produces──►  TwinLlama-3.1-8B   (π_ref)
Chapter 3 data ──produces──►  llmtwin-dpo        (preference pairs)
                                        │
                                        ▼
                                   DPO training
                                        │
                                        ▼
                              TwinLlama-3.1-8B-DPO
```

DPO cannot run before SFT. The reference model *is* the SFT model; without it
the preference signal has no anchor.

---

## 🧠 The DPO Objective, Precisely

### The RLHF problem DPO replaces

Classical RLHF (e.g. PPO) maximizes expected reward subject to a KL penalty
against a frozen reference policy:

```
max_π  E_(x,y~π) [ r(x, y) ]  -  β · KL( π(·|x) || π_ref(·|x) )
```

This needs a separately trained reward model `r` and an online RL loop. DPO's
insight (Rafailov et al., 2023) is that the optimal policy of that objective has
a closed form, and it can be inverted to express the reward in terms of the
policy:

```
r(x, y) = β · log( πθ(y|x) / π_ref(y|x) )  +  β · log Z(x)
```

where `Z(x)` is a partition function that depends only on `x` (not on `y`).
Substituting this reward back into the Bradley-Terry preference likelihood makes
`Z(x)` cancel, leaving a loss that trains `πθ` directly on preference pairs.

### The loss

```
L_DPO(πθ; πref) = - E_(x, y_w, y_l) [ log σ( β · Δ ) ]

Δ = log(πθ(y_w|x) / πref(y_w|x)) - log(πθ(y_l|x) / πref(y_l|x))
```

- **`πref`** is the frozen SFT model; **`πθ`** is the policy being trained (it
  starts from the same weights).
- **`β` (beta)** controls how strongly the policy is pulled back toward the
  reference. Low beta allows large deviations; high beta keeps the model close
  to SFT. In the RLHF view, beta *is* the KL coefficient.
- The loss increases the **relative** log-probability of chosen versus rejected,
  normalized by the reference. The normalization subtracts out what the SFT
  model already finds likely, so DPO only learns the *preference* signal instead
  of relearning the language.
- `σ` is the logistic sigmoid. No reward model and no reinforcement loop are
  needed, unlike PPO-based RLHF - it is a single supervised-style loss over
  pairs.

### The gradient, intuitively

The gradient of `L_DPO` with respect to the policy weights is:

```
∇L = -β · σ( -β · Δ ) · [ ∇log πθ(y_w|x) - ∇log πθ(y_l|x) ]
```

The term `σ(-βΔ)` is large when the model currently gets the pair wrong
(`Δ < 0`) and small when it already prefers `y_w` (`Δ` large). So DPO applies a
strong update to pairs it is getting wrong, and a near-zero update once the
margin is established. That is why the loss naturally flattens.

### Worked numeric example

Take `β = 0.5`, prompt `x`, chosen `y_w`, rejected `y_l`, and the reference from
SFT. Suppose at the start of training:

| Quantity | Value |
|----------|-------|
| `πθ(y_w \| x)` | 0.20 |
| `πref(y_w \| x)` | 0.10 |
| `πθ(y_l \| x)` | 0.05 |
| `πref(y_l \| x)` | 0.10 |

Step 1 - log ratios away from the reference:

```
log(πθ(y_w|x)/πref(y_w|x)) = log(0.20/0.10) = log 2  =  0.6931
log(πθ(y_l|x)/πref(y_l|x)) = log(0.05/0.10) = log 0.5 = -0.6931
```

Step 2 - subtract:

```
Δ = 0.6931 - (-0.6931) = 1.3863
```

Step 3 - scale by beta and squash:

```
β·Δ = 0.5 × 1.3863 = 0.6931
σ(0.6931) = 1 / (1 + e^-0.6931) = 0.6667
```

Step 4 - loss:

```
L = -log(0.6667) = 0.4055
```

Now suppose training shifts the policy so it likes `y_w` more and `y_l` less:
`πθ(y_w|x) = 0.30`, `πθ(y_l|x) = 0.02`.

```
log ratios:  log 3 = 1.0986  and  log 0.2 = -1.6094
Δ = 2.7080
β·Δ = 1.3540
σ(1.3540) = 0.7946
L = -log(0.7946) = 0.2299
```

The loss fell from 0.4055 to 0.2299: the same direction of update, but with a
smaller step because the margin is already healthier. This is the mechanism
Comet ML's *margin* metric tracks.

### Why the learning rate is tiny (2e-6 vs 3e-4 for SFT)

DPO starts from a good model and makes a small, targeted adjustment. A large LR
would blow past the reference, collapse the margin, and often destroy SFT
quality. The book trained over 20 models to settle on this value. Note that this
project is an *advanced* DPO case: the goal is to imitate a writing style, but
DPO's natural tendency is to push toward formal language (chosen answers are
often more formal than rejected ones). That conflict is exactly why a low LR and
a single epoch are required.

### SFT vs DPO in one table

| Aspect | SFT | DPO |
|--------|-----|-----|
| Data fields | `instruction` / `output` | `prompt` / `chosen` / `rejected` |
| Objective | Cross-entropy on the target answer | Preference contrast vs reference |
| Reference model | none | the SFT checkpoint (`πref`) |
| Learning rate | `3e-4` | `2e-6` |
| Epochs | 3 | 1 |
| Loss | `-log P(target)` | `-log σ(β·Δ)` |
| Memory | base + adapter + one sequence | base + adapter + two sequences (chosen/rejected) |
| Output | `TwinLlama-3.1-8B` | `TwinLlama-3.1-8B-DPO` |

**What changes in behaviour**: SFT makes the model answer in your style. DPO
makes the model prefer the *style and faithfulness* of the verbatim corpus
excerpts over its own generated attempts. The result is a model that is both
on-style and more faithful to the source.

---

## 📁 Key Files Explained

### 1. Formatting the Preference Triples

```python
# llm_engineering/model/finetuning/finetune.py (dpo branch)
    elif finetuning_type == "dpo":
        PatchDPOTrainer()

        def format_samples_dpo(example):
            example["prompt"] = alpaca_template.format(example["prompt"], "")
            example["chosen"] = example["chosen"] + EOS_TOKEN
            example["rejected"] = example["rejected"] + EOS_TOKEN

            return {"prompt": example["prompt"], "chosen": example["chosen"], "rejected": example["rejected"]}

        dataset = load_dataset(f"{dataset_huggingface_workspace}/llmtwin-dpo", split="train")
```

**Key Concepts**:
- **The prompt uses the Alpaca template with an empty response**:
  `alpaca_template.format(prompt, "")` ends at `### Response:\n`, so the model is
  asked to produce the answer. This matches the SFT prompt format exactly, which
  matters: the reference model must see the same prompt distribution it was
  trained on.
- **Only the instruction gets the chat template.** Chosen and rejected are raw
  continuations; they only need `EOS_TOKEN` appended.
- **Both chosen and rejected get an EOS token**, matching the SFT formatting. The
  loss compares the full continuation probabilities of each.
- **Column names `prompt`/`chosen`/`rejected`** are exactly what the
  `PreferenceDataset.to_huggingface()` export produced in Session 3.2
  (`domain/dataset.py:95`), so the schema is intentionally aligned end to end.
- **`PatchDPOTrainer()` is called twice** in this file: once at module import
  (`finetune.py:5-7`) and again inside the DPO branch (`finetune.py:145`). The
  Unsloth patch fixes DPO log rendering in notebook environments. It is
  idempotent, so the repetition is harmless.

The upstream `alpaca_template` is defined once at module scope
(`finetune.py:20`):

```python
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
```

### 2. Dataset Loading and Dummy Mode

```python
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")
        print(f"Loaded dataset with {len(dataset)} samples.")

        dataset = dataset.map(format_samples_dpo)
        dataset = dataset.train_test_split(test_size=0.05)
```

Same dummy/split pattern as SFT: 400 samples for smoke tests, 5% evaluation
split. In dummy mode the trainer also forces one epoch, set once at the top of
`finetune()` (`finetune.py:86-89`):

```python
    if is_dummy is True:
        num_train_epochs = 1
        print(f"Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'")
```

Note the mapping is **not batched** for DPO (`dataset.map(format_samples_dpo)`
without `batched=True`), because each example is transformed independently. The
SFT branch uses `batched=True` and `remove_columns`, DPO does not.

### 3. `DPOTrainer` Configuration

```python
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    beta=beta,  # default 0.5
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_length=max_seq_length // 2,  # 2048 // 2 = 1024
    max_prompt_length=max_seq_length // 2,
    args=DPOConfig(
        learning_rate=learning_rate,  # 2e-6 from __main__
        num_train_epochs=num_train_epochs,  # 1
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        per_device_eval_batch_size=per_device_train_batch_size,
        warmup_steps=10,
        output_dir=output_dir,
        eval_steps=0.2,
        logging_steps=1,
        report_to="comet_ml",
        seed=0,
    ),
)
```

**Key Concepts**:
- **`ref_model=None`** is the Unsloth optimization: the reference is the
  *disabled adapter* of the same model. With PEFT, disabling the LoRA adapter
  recovers the frozen base, so no second model copy is loaded. This roughly
  halves VRAM versus loading an explicit reference. The book states this
  directly: because only adapters are trained, the base model is never modified,
  so the reference and the trained model can share one set of weights.
- **`beta=0.5`** is a moderately strong KL anchor. Lower values (0.1) allow
  bigger stylistic shifts; higher values (0.9) stay close to SFT. The book's
  default recommendation is 0.1, but 0.5 was chosen here because lower values
  pushed the model toward formal language.
- **`max_length = max_seq_length // 2 = 1024`**: DPO stores both chosen and
  rejected completions, so each sequence budget is halved to fit both in memory.
- **`max_prompt_length`** caps the shared prompt portion; the remaining budget is
  reserved for the two continuations.
- **`eval_steps=0.2`** evaluates 5 times per epoch (fraction-based), which is
  useful given DPO runs only 1 epoch. The repo's `DPOConfig` does not set
  `eval_strategy`; the book snippet adds `eval_strategy="steps"`. Set it
  explicitly if you need evaluation to fire on a schedule.
- **`optim="adamw_8bit"`** keeps optimizer state in 8-bit, which matters when two
  sequences and two forward passes per step are competing for VRAM.

### 4. The `__main__` DPO Entry Point

```python
    elif args.finetuning_type == "dpo":
        sft_base_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
        sft_base_model_repo_id = check_if_huggingface_model_exists(sft_base_model_repo_id)
        print(f"Training from base model '{sft_base_model_repo_id}'")

        output_dir_dpo = Path(args.model_dir) / "output_dpo"
        model, tokenizer = finetune(
            finetuning_type="dpo",
            model_name=sft_base_model_repo_id,
            output_dir=str(output_dir_dpo),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            num_train_epochs=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=2e-6,
            is_dummy=args.is_dummy,
        )
        inference(model, tokenizer)

        dpo_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B-DPO"
        save_model(model, tokenizer, "model_dpo", push_to_hub=True, repo_id=dpo_output_model_repo_id)
```

**Key Concepts**:
- **`check_if_huggingface_model_exists`** (`finetune.py:226`) falls back to
  `mlabonne/TwinLlama-3.1-8B` if your SFT model is not on the Hub:

  ```python
  def check_if_huggingface_model_exists(model_id: str, default_value: str = "mlabonne/TwinLlama-3.1-8B") -> str:
      api = HfApi()
      try:
          api.model_info(model_id)
      except RepositoryNotFoundError:
          model_id = default_value
      return model_id
  ```

  This is a convenience so readers can run DPO without having trained SFT
  themselves.
- **DPO starts from the SFT checkpoint**, not the raw Llama base. This is what
  makes the reference meaningful.
- **`learning_rate=2e-6`, `num_train_epochs=1`** are the production values,
  overriding the general defaults (`3e-4`, `3`).
- **The `__main__` also reads SageMaker environment variables**:
  `SM_OUTPUT_DATA_DIR`, `SM_MODEL_DIR`, `SM_NUM_GPUS` (`finetune.py:257-259`).
  When run locally these must be exported, or the script fails at argparse time.
- **`save_model` always merges weights**:

  ```python
  def save_model(model, tokenizer, output_dir, push_to_hub=False, repo_id=None):
      model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
      if push_to_hub and repo_id:
          model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")
  ```

  The adapter is merged back into a 16-bit base before export, so the Hub
  artifact is a standalone model, not a LoRA adapter.

### 5. `domain/dataset.py` - The Preference Schema

```python
# llm_engineering/domain/dataset.py
class PreferenceDatasetSample(VectorBaseDocument):
    instruction: str
    rejected: str
    chosen: str

    class Config:
        category = DataCategory.PREFERENCE_DATASET_SAMPLES


class PreferenceDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[PreferenceDatasetSample]

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]
        return Dataset.from_dict(
            {
                "prompt": [d["instruction"] for d in data],
                "rejected": [d["rejected"] for d in data],
                "chosen": [d["chosen"] for d in data],
            }
        )
```

**Key Concepts**:
- The domain object stores the field name **`instruction`**, but the Hugging Face
  export renames it to **`prompt`**. That rename is the reason
  `format_samples_dpo` reads `example["prompt"]` and not
  `example["instruction"]`.
- Field order in the export is `prompt`, `rejected`, `chosen` - alphabetical by
  chance, not significance. The trainer keys off names, not order.

---

## 🔬 Deep Dive: DPO Metrics and Debugging

Unlike SFT, DPO logs several preference-specific series. Comet ML shows them in
the experiment for the run (`report_to="comet_ml"`). The book reviews them as
follows.

| Metric | What it means | Healthy shape | Red flag |
|--------|---------------|---------------|----------|
| Training loss | `-log σ(β·Δ)` on train | Decreasing on average | Falls to ~0 instantly |
| Validation loss | Same, on the 5% split | Small gap vs train | Large or rising gap |
| Gradient norm | Magnitude of updates | Small, few spikes | Frequent large spikes |
| Rewards (chosen) | Mean `log πθ/πref` for chosen | Rises | Flat or falls |
| Rewards (rejected) | Mean `log πθ/πref` for rejected | Falls | Rises |
| Margins | chosen reward − rejected reward | Rises then plateaus | Stays near 0 |
| Accuracies | % of pairs where margin > 0 | Gradual rise | 100% almost immediately |

**Reading the signal**:
- A training loss that collapses to zero almost immediately means the model is
  no longer learning; it is not necessarily overfitting, but it warrants a
  closer look at the data difficulty.
- An accuracy of 100%, especially if reached quickly, indicates the preference
  dataset is too easy. The model can still learn, but more challenging examples
  would help.
- A margin that plateaus is expected and good. The margin is the quantity DPO
  is explicitly pushing up; once it is large the gradient `σ(-βΔ)` shrinks.

**Automating style evaluation** (mentioned in the book): compare the word
distribution of SFT vs DPO output against the ground-truth corpus. The SFT model
is expected to over-represent GPT-4o-mini words (for example "delve into"); a
well-aligned DPO model should be closer to the chosen answers' distribution.
This is optional and outside the scope of the training script.

---

## 🔬 Deep Dive: LoRA vs QLoRA on the RTX 5000 (16 GB)

The book's tutorial sets `load_in_4bit=False` and runs LoRA DPO, because the
authors trained on datacenter GPUs. On the local RTX 5000 (16 GB, sm_75) the
arithmetic is tighter.

Rough VRAM budget for an 8B model at `max_seq_length` 1024 (per the workstation
`vram-calc` helper):

| Configuration | Weights | Overhead | Total | Fits 16 GB? |
|---------------|---------|----------|-------|-------------|
| FP16 training (full) | ~16.0 GB | ~15.5 GB | ~31.5 GB | No (exceeds by ~15.5 GB) |
| QLoRA DPO (4-bit) | ~4.0 GB | ~2.7 GB | ~6.7 GB | Yes (~9.3 GB free) |
| 8-bit merged model at inference | ~8.0 GB | ~1.1 GB | ~9.1 GB | Yes (~6.9 GB free) |

**Recommendation for this workstation**:
- Set `load_in_4bit=True` (QLoRA) when you run DPO locally. `load_in_4bit` is a
  parameter of `finetune()` (`finetune.py:67`), forwarded to
  `FastLanguageModel.from_pretrained`.
- Keep `per_device_train_batch_size=1` and rely on
  `gradient_accumulation_steps=8` to preserve the effective batch size.
- The two sequences per example (chosen, rejected) are the memory multiplier;
  `max_length // 2` is what keeps them in budget.
- DPO does two forward passes per step (policy and, implicitly, the disabled
  adapter reference), so it is slower than SFT at the same batch size.

`sentencepiece`/`unsloth` installs are heavy; if the local run OOMs on the
4-bit load, reduce `max_seq_length` from 2048 to 1024 before touching the model
size.

---

## 🛠️ Hands-On: Run and Compare DPO

### Step 1: Confirm the SFT checkpoint

```python
from huggingface_hub import HfApi

api = HfApi()
print(api.model_info("<workspace>/TwinLlama-3.1-8B").id)
```

### Step 2: Inspect a formatted DPO sample

```python
from datasets import load_dataset

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

ds = load_dataset("<workspace>/llmtwin-dpo", split="train")
ex = ds[0]
print("PROMPT:", alpaca_template.format(ex["prompt"], ""))
print("CHOSEN:", ex["chosen"])
print("REJECTED:", ex["rejected"])
```

### Step 3: Dummy DPO run

```bash
python llm_engineering/model/finetuning/finetune.py \
    --finetuning_type dpo \
    --is_dummy True \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-6
```

Export the SageMaker env vars first if you run the script directly:

```bash
export SM_OUTPUT_DATA_DIR=./output_data
export SM_MODEL_DIR=./model
export SM_NUM_GPUS=1
```

### Step 4: Compare generations

Load `TwinLlama-3.1-8B` and `TwinLlama-3.1-8B-DPO`, run the same prompt
through both, and compare tone and faithfulness. The DPO model should more often
reproduce corpus phrasing. A useful reference prompt is the book's own:
"Write a paragraph to introduce supervised fine-tuning."

### Step 5: Read the Comet ML run

Open the run from the `twin` project and inspect the reward and margin curves.
A good run shows chosen reward rising, rejected reward falling, and margin
rising then plateauing.

---

## 📝 Exercise 1: Beta Sweep

### Task

Understand how `beta` balances style and safety.

1. Run DPO three times with `beta=0.1`, `0.5`, and `0.9` (dummy mode).
2. For each, generate the same three prompts.
3. Judge: which is most faithful? Which drifts most from SFT?
4. Explain how beta is acting as a KL anchor.
5. Plot the margin curve for each run side by side.

**Goal**: See the tradeoff between moving toward preferred behaviour and staying
close to the trusted SFT model.

---

## 📝 Exercise 2: Hand-Compute the Loss, Then Verify in Code

### Task

1. On paper, compute `L_DPO` for `β = 0.5` given
   `πθ(y_w)=0.15, πref(y_w)=0.10, πθ(y_l)=0.07, πref(y_l)=0.10`.
2. Now compute it again after the policy moves to `πθ(y_w)=0.25, πθ(y_l)=0.03`.
3. Write a 15-line Python function `dpo_loss(...)` that returns the loss, and
   check it reproduces both numbers.
4. Explain why the loss is largest when `Δ < 0` and smallest when `Δ` is large
   positive.

**Goal**: Internalize that DPO is a scalar function of two log-ratios, not a
mystery RL process.

---

## 🐛 Common Pitfalls

- **Starting DPO from the base model**: without an SFT reference the preferences
  have no anchor; quality degrades. Always start from `TwinLlama-3.1-8B`.
- **Learning rate too high**: `3e-4` (the SFT value) will wreck the model. DPO
  needs `~2e-6`.
- **Sequence budget**: `max_length = max_seq_length // 2`; raising it without
  lowering batch size will OOM because two completions are held per example.
- **Empty `chosen` after filtering**: if Session 3.2 filters removed almost
  everything, DPO trains on too little data. Check the sample count first. The
  book's pipeline kept 1,467 of 2,970 generated samples after filtering.
- **Missing `eval_strategy`**: the repo config sets `eval_steps` but not
  `eval_strategy`, so validation may not trigger. Add `eval_strategy="steps"` if
  you want per-step eval logs.
- **Running the script without SageMaker env vars**: `argparse` reads
  `SM_OUTPUT_DATA_DIR`, `SM_MODEL_DIR`, and `SM_NUM_GPUS`, so a bare
  `python finetune.py` fails unless they are exported.
- **Interpreting a fast-falling loss as success**: a loss that hits ~0 is a
  signal to check the margin and accuracy, not a win.
- **Forgetting EOS on only one branch**: both chosen and rejected need the EOS
  token; an asymmetric append biases the likelihood comparison.
- **Using beta lower than the style needs**: on this dataset, low beta pushed
  the model toward formal language. Style imitation may require beta at the
  higher end.

---

## 🎓 Knowledge Check

1. **What plays the role of `πref` in this project?**
   - Answer: The SFT checkpoint `TwinLlama-3.1-8B`, reused via Unsloth's
     disabled-adapter mechanism (`ref_model=None`).

2. **Why is `beta` called a KL anchor?**
   - Answer: It controls how far the policy may deviate from the reference;
     higher beta keeps it closer. In the RLHF objective, beta is the KL
     coefficient.

3. **Why does DPO use two completions per example in memory?**
   - Answer: The loss compares chosen and rejected, so both sequences must be
     present in the batch.

4. **Why is the DPO learning rate ~150× smaller than SFT's?**
   - Answer: DPO makes a small alignment adjustment from an already-good model;
     a large LR would overshoot the reference and destroy SFT quality.

5. **What does `check_if_huggingface_model_exists` guard against?**
   - Answer: A missing custom SFT model, by falling back to the public
     `mlabonne/TwinLlama-3.1-8B`.

6. **What schema does the DPO dataset use?**
   - Answer: `prompt`, `chosen`, `rejected` - matching the Session 3.2 export.

7. **What is `Δ` in the DPO loss?**
   - Answer: The difference of two reference-normalized log-ratios:
     `log(πθ(y_w)/πref(y_w)) - log(πθ(y_l)/πref(y_l))`.

8. **Why does the `Z(x)` partition function cancel?**
   - Answer: It depends only on the prompt `x`, so it appears identically in the
     chosen and rejected terms of the Bradley-Terry ratio and divides out.

9. **Compute `L_DPO` for `β=1`, `Δ=0`.**
   - Answer: `σ(0) = 0.5`, so `L = -log 0.5 = 0.693`. The model is indifferent
     between chosen and rejected.

10. **What does a margin of zero mean?**
    - Answer: The policy assigns the same reference-normalized reward to chosen
      and rejected; it has not learned the preference yet.

11. **Why does only the prompt get the chat template?**
    - Answer: Chosen and rejected are the model's continuations, so they must be
      raw text plus EOS, not formatted as new instructions.

12. **What does `save_method="merged_16bit"` produce?**
    - Answer: A standalone 16-bit model with the LoRA adapter merged into the
      base weights, suitable for direct loading or SageMaker inference.

13. **Which metric tells you the preference dataset is too easy?**
    - Answer: Accuracy near 100% reached very quickly.

14. **Why can DPO share one model for policy and reference?**
    - Answer: Because only LoRA adapters are trained; disabling the adapter
      recovers the untouched base, so no second weight copy is needed.

---

## 📖 Glossary

- **DPO (Direct Preference Optimization)**: A supervised method that trains a
  policy directly on chosen/rejected pairs, replacing the reward model and RL
  loop of RLHF.
- **πθ (policy)**: The model being trained.
- **πref (reference)**: The frozen SFT model used to normalize the preference
  signal.
- **β (beta)**: The KL-anchor strength; higher means closer to the reference.
- **Δ (delta)**: The preference margin in log-ratio space that the loss pushes
  positive.
- **Bradley-Terry model**: A probabilistic model of pairwise preferences; the
  basis for the DPO derivation.
- **Margin**: `reward(chosen) - reward(rejected)`; the tracked preference gap.
- **LoRA / QLoRA**: Parameter-efficient fine-tuning with low-rank adapters;
  QLoRA additionally quantizes the base to 4-bit.
- **EOS token**: End-of-sequence marker appended to each continuation.
- **KL divergence**: A measure of how far the policy's distribution is from the
  reference; the regularization DPO inherits.

---

## 🔗 Next Session

**Session 5.3**: AWS SageMaker Deployment

We deploy the DPO model to a real-time SageMaker endpoint with autoscaling.

See [Session 5.3: AWS SageMaker Deployment](session_5.3_sagemaker_deployment.md).

---

## 📚 Additional Resources

- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [Unsloth DPO Guide](https://docs.unsloth.ai/basics/dpo)
- [A Survey of RLHF (Kaufmann et al., 2023)](https://arxiv.org/abs/2312.14925)
- [APA preference dataset: mlabonne/llmtwin-dpo](https://huggingface.co/datasets/mlabonne/llmtwin-dpo)
- Related sessions: [Session 3.2 Preference Dataset](session_3.2_preference_dataset.md),
  [Session 5.1 SFT](session_5.1_sft.md)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 3.2, 5.1

**Outcome**: You can align an SFT model with preference data, derive and
hand-compute the DPO loss, read its metrics, and reason about the
beta/learning-rate tradeoffs.
