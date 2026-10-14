# Session 5.1: Supervised Fine-Tuning (SFT)

> Book: Chapter 5, "Supervised Fine-Tuning" (pages 235-255).
> Repo: `llm_engineering/model/finetuning/finetune.py`, `llm_engineering/model/finetuning/sagemaker.py`.
> Companion: [Session 3.1](session_3.1_instruction_dataset.md) builds the instruction dataset this session trains on.

## 🎯 Learning Objectives

By the end of this session, you will:

- Explain what SFT is, when to fine-tune, and when to prefer prompt engineering or RAG.
- Read `finetune.py` end to end: imports, `load_model`, the SFT branch, `inference`, `save_model`, and the `__main__` entry point.
- Distinguish full fine-tuning, LoRA, and QLoRA, and compute their memory costs.
- Configure `SFTTrainer` correctly for the RTX 5000 (Turing sm_75): fp16, not bfloat16.
- Read `sagemaker.py` and understand how the same script runs on `ml.g5.2xlarge`.
- Format the Alpaca template and explain why the EOS token is mandatory.
- Run a dummy smoke test, then an ablation on LoRA rank.

---

## 🏗️ Architecture Overview

### Where SFT sits

```
Instruction dataset (Session 3.1)   {workspace}/llmtwin : (instruction, output)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     finetune(finetuning_type="sft")            │
│                                                                │
│  FastLanguageModel.from_pretrained(base, load_in_4bit=?)       │
│        │  LoRA: fp16/fp32 frozen base  (load_in_4bit=False)     │
│        │  QLoRA: 4-bit NF4 frozen base (load_in_4bit=True)      │
│        ▼                                                       │
│  FastLanguageModel.get_peft_model(r=32, alpha=32, dropout=0.0) │
│        │  trainable adapters on q/k/v/o/gate/up/down_proj       │
│        ▼                                                       │
│  format_samples_sft()  →  alpaca_template + EOS_TOKEN          │
│        ▼                                                       │
│  SFTTrainer(packing=True, fp16 (Turing), adamw_8bit)           │
│        ▼                                                       │
│  save_pretrained_merged(save_method="merged_16bit")            │
│        ▼                                                       │
│  push_to_hub  →  {workspace}/TwinLlama-3.1-8B                  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Output is the base for DPO (Session 5.2) and the RAG generator.
```

### When to fine-tune (book Figure 5.8)

```
Do you have a clear task and eval?
        │
        ▼
Try prompt engineering first (few-shot, RAG)  ──►  meets metrics? ──► ship
        │ no
        ▼
Do you have enough quality instruction data?  ──►  no ──► collect more
        │ yes
        ▼
Fine-tune (SFT), then align (DPO)
```

Start with prompt engineering. It works with open and closed models and lets you build an evaluation pipeline measuring accuracy, cost, and latency. Only when those results miss requirements, and you have enough data, does fine-tuning become an option.

### Why SFT, and its limits

SFT retrains a pre-trained model on (instruction, answer) pairs to turn next-token prediction into instruction following. It can also add a voice, a domain, or a task focus. Limits to know:

- SFT refocuses pre-existing knowledge; it does not teach knowledge far from pre-training (rare languages, novel facts).
- Fine-tuning on new knowledge can *increase* hallucinations.
- Destructive techniques (full fine-tuning, continual pre-training) risk **catastrophic forgetting**.

The three SFT techniques:

| Technique | Trainable params | Base precision | Memory (7-8B) | Notes |
|-----------|------------------|----------------|---------------|-------|
| Full fine-tuning | all | fp16/fp32 | ~112 GB (7B, fp32 baseline) | Best quality, needs many GPUs |
| LoRA | A/B matrices only | frozen fp16/fp32 | 14-18 GB | Fastest, highest quality per GB |
| QLoRA | A/B matrices only | frozen 4-bit NF4 | ~9 GB | ~30% slower, minor quality gap |

---

## 📁 Key Files Explained

### 1. `finetune.py` — the patch-first import order

```python
# llm_engineering/model/finetuning/finetune.py
import argparse
import os
from pathlib import Path

from unsloth import PatchDPOTrainer

PatchDPOTrainer()

from typing import Any, List, Literal, Optional  # noqa: E402

import torch  # noqa
from datasets import concatenate_datasets, load_dataset  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import RepositoryNotFoundError  # noqa: E402
from transformers import TextStreamer, TrainingArguments  # noqa: E402
from trl import DPOConfig, DPOTrainer, SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402
```

**Why `PatchDPOTrainer()` runs before the TRL imports.** It monkey-patches TRL classes for Unsloth compatibility. The patch must be in place *before* TRL is imported, so the import order is load-bearing. The `# noqa: E402` comments silence the "module level import not at top of file" lint that this ordering creates. `PatchDPOTrainer` is imported here but only *called* again inside the DPO branch; the top-level call is what matters.

**Everything trains through Unsloth** (`FastLanguageModel`, `is_bfloat16_supported`, `get_chat_template`), which supplies fused kernels and memory optimizations. The book notes Unsloth is 2-5× faster and uses up to 80% less memory than a vanilla TRL setup.

---

### 2. The Alpaca template

```python
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
```

**Key points.**

- Every sample is `alpaca_template.format(instruction, output) + EOS_TOKEN`.
- The EOS token is the answer terminator. Without it the model never learns to stop and will generate without end.
- The same template is reused at DPO time (with an empty response) and at inference, so train and inference formatting stay identical.
- The book calls Alpaca "less error-prone" than ChatML because it needs no extra special tokens, though it can slightly underperform a proper chat template. The script still calls `get_chat_template(tokenizer, chat_template="chatml")` in `load_model`, so the tokenizer knows the family's special tokens even while the data uses Alpaca formatting.

---

### 3. `load_model` — base model + LoRA adapters

```python
def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    chat_template: str,
) -> tuple:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )

    return model, tokenizer
```

**Key points.**

- **`load_in_4bit`** selects LoRA vs QLoRA. `False` (the default in `finetune`) keeps the base frozen in its loaded precision (LoRA). `True` quantizes the frozen base to 4-bit NF4 (QLoRA). Forward/backward still run in higher precision; only the adapters are stored for training.
- **`r=32, alpha=32`.** `alpha == r` keeps the effective update scale near 1. Rank is the capacity knob; larger ranks fit more but use more memory and risk overfitting.
- **`target_modules`** spans attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`). Adapting the MLP projections is what lets a small rank capture style changes. Targeting one module can be much cheaper but weaker than targeting all seven.
- **`lora_dropout=0.0`:** Unsloth recommends 0 for its fused kernels. The book allows 0-0.1 as optional regularization.
- **`get_chat_template(tokenizer, "chatml")`** installs the ChatML template on the tokenizer.

> **Correction vs earlier drafts:** the project's `__main__` SFT path calls `finetune(...)` **without** `load_in_4bit`, so the shipped default is **LoRA (4-bit off)**, matching the book's example. Pass `load_in_4bit=True` explicitly for QLoRA.

---

### 4. `finetune` — signature and defaults

```python
def finetune(
    finetuning_type: Literal["sft", "dpo"],
    model_name: str,
    output_dir: str,
    dataset_huggingface_workspace: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],  # noqa: B006
    chat_template: str = "chatml",
    learning_rate: float = 3e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    beta: float = 0.5,  # Only for DPO
    is_dummy: bool = True,
) -> tuple:
    model, tokenizer = load_model(
        model_name, max_seq_length, load_in_4bit, lora_rank, lora_alpha, lora_dropout, target_modules, chat_template
    )
    EOS_TOKEN = tokenizer.eos_token
    print(f"Setting EOS_TOKEN to {EOS_TOKEN}")  # noqa

    if is_dummy is True:
        num_train_epochs = 1
        print(f"Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'")  # noqa
        print(f"Training in dummy mode. Reducing dataset size to '400'.")  # noqa
```

`EOS_TOKEN` is taken from the tokenizer, not hard-coded — for Llama 3.1 it is `<|eot_id|>`. `is_dummy=True` (the function default) forces one epoch and a 400-sample subset so the whole pipeline can be smoke-tested cheaply. The `# noqa: B006` on the mutable default list is acknowledged; the list is only read, never mutated.

---

### 5. The SFT data branch

```python
    if finetuning_type == "sft":

        def format_samples_sft(examples):
            text = []
            for instruction, output in zip(examples["instruction"], examples["output"], strict=False):
                message = alpaca_template.format(instruction, output) + EOS_TOKEN
                text.append(message)

            return {"text": text}

        dataset1 = load_dataset(f"{dataset_huggingface_workspace}/llmtwin", split="train")
        dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
        dataset = concatenate_datasets([dataset1, dataset2])
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")  # noqa
        print(f"Loaded dataset with {len(dataset)} samples.")  # noqa

        dataset = dataset.map(format_samples_sft, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.train_test_split(test_size=0.05)

        print("Training dataset example:")  # noqa
        print(dataset["train"][0])  # noqa
```

**Key points.**

- **Two datasets are concatenated.** The domain `llmtwin` data (~3,000 samples, per the book) plus 10,000 general Alpaca samples from `mlabonne/FineTome-Alpaca-100k`. The book explains why: the domain set is too small for the model to learn the chat template reliably; mixing general instruction data preserves general instruction-following and reduces catastrophic forgetting.
- **`batched=True`** formats samples in batches for speed.
- **`remove_columns=dataset.column_names`** drops the original columns, leaving only `text`, which `SFTTrainer` reads via `dataset_text_field="text"`.
- **`is_dummy`** trims to 400 samples and (above) one epoch.
- **5% held out** for evaluation.

---

### 6. `SFTTrainer` configuration

```python
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=True,
            args=TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                report_to="comet_ml",
                seed=0,
            ),
        )
```

**Every hyperparameter, with the "why".**

| Setting | Value | Why |
|---------|-------|-----|
| `packing=True` | on | Concatenates short samples into full-length sequences, removing padding waste. Big throughput win. |
| `fp16` / `bf16` | `is_bfloat16_supported()` picks | On Ampere+ it is bf16; on the **RTX 5000 (Turing) it is fp16**. `fp16=not is_bfloat16_supported()` and `bf16=is_bfloat16_supported()` guarantee exactly one is on. |
| `optim="adamw_8bit"` | 8-bit optimizer | Stores optimizer states in 8-bit; another memory saver. |
| `learning_rate=3e-4` | LoRA-appropriate | LoRA tolerates a much higher LR than full fine-tuning (which uses ~1e-5). |
| `lr_scheduler_type="linear"` | linear | LR rises during warmup then decays linearly. Cosine performs similarly. |
| `warmup_steps=10` | 10 | Stabilizes early training. |
| `weight_decay=0.01` | 0.01 | Mild regularization against large weights. |
| `per_device_train_batch_size=2` | 2 | Micro-batch that fits VRAM. |
| `gradient_accumulation_steps=8` | 8 | Builds a larger effective batch without more memory. |
| `num_train_epochs=3` | 3 | Typical for LoRA SFT; more risks overfitting. |
| `logging_steps=1` | 1 | Log every step for smooth Comet curves. |
| `report_to="comet_ml"` | Comet ML | Experiment tracking (Session 7.1). |
| `seed=0` | 0 | Reproducibility. |

**Effective batch size.**

```
effective_batch = per_device_train_batch_size × num_devices × gradient_accumulation_steps
                = 2 × 1 × 8
                = 16
```

Book example with 2 GPUs, batch 4, accumulation 4: `4 × 2 × 4 = 32`. Accumulation lets a 16 GB card behave like it has more memory, at the cost of wall-clock time per update.

**Why packing helps, concretely.** With `max_seq_length=2048` and samples of 200-300 tokens, an unpacked batch wastes most of the budget on padding. Packing can fit several samples per sequence, so the model sees far more tokens per step. The catch: attention must not cross sample boundaries, handled by attention masks inside `SFTTrainer`.

---

### 7. The DPO branch (context for Session 5.2)

```python
    elif finetuning_type == "dpo":
        PatchDPOTrainer()

        def format_samples_dpo(example):
            example["prompt"] = alpaca_template.format(example["prompt"], "")
            example["chosen"] = example["chosen"] + EOS_TOKEN
            example["rejected"] = example["rejected"] + EOS_TOKEN

            return {"prompt": example["prompt"], "chosen": example["chosen"], "rejected": example["rejected"]}

        dataset = load_dataset(f"{dataset_huggingface_workspace}/llmtwin-dpo", split="train")
        ...
        trainer = DPOTrainer(
            model=model,
            ref_model=None,               # Unsloth uses the frozen base as reference
            tokenizer=tokenizer,
            beta=beta,                    # default 0.5
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=max_seq_length // 2,
            max_prompt_length=max_seq_length // 2,
            args=DPOConfig(...),
        )
    else:
        raise ValueError("Invalid finetuning_type. Choose 'sft' or 'dpo'.")

    trainer.train()

    return model, tokenizer
```

The DPO branch reuses `load_model` and the same `PatchDPOTrainer` machinery. `ref_model=None` tells Unsloth to use the frozen base weights as the reference model, avoiding a second copy. `max_length = max_seq_length // 2` because a preference sample stores a prompt plus chosen and rejected completions. Full treatment in [Session 5.2](session_5.2_dpo.md).

---

### 8. `inference`, `save_model`, and the entry point

```python
def inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "Write a paragraph to introduce supervised fine-tuning.",
    max_new_tokens: int = 256,
) -> None:
    model = FastLanguageModel.for_inference(model)
    message = alpaca_template.format(prompt, "")
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True)


def save_model(model: Any, tokenizer: Any, output_dir: str, push_to_hub: bool = False, repo_id: Optional[str] = None):
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

    if push_to_hub and repo_id:
        print(f"Saving model to '{repo_id}'")  # noqa
        model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")


def check_if_huggingface_model_exists(model_id: str, default_value: str = "mlabonne/TwinLlama-3.1-8B") -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")  # noqa
        model_id = default_value
        print(f"Defaulting to '{model_id}'")  # noqa
        print("Train your own 'TwinLlama-3.1-8B' to avoid this behavior.")  # noqa

    return model_id
```

**`inference` details.**

- `FastLanguageModel.for_inference(model)` switches to the fast inference path.
- `alpaca_template.format(prompt, "")` supplies an **empty** response, which leaves `### Response:` at the end of the input. This forces the model to *answer* rather than continue the instruction.
- The streaming `TextStreamer` prints tokens as they generate.
- `use_cache=True` enables the KV cache for faster decoding.

**`save_model` details.**

- `merged_16bit` merges the LoRA adapter into the base weights and stores fp16 weights. This produces a standalone model that vLLM/SageMaker can serve without PEFT.
- Logging is declared in files via `print(..., # noqa)` rather than `logging`, matching the rest of the script.

**`check_if_huggingface_model_exists`** guards the DPO path: if the SFT model has not been pushed to the Hub, it falls back to the book's `mlabonne/TwinLlama-3.1-8B`.

---

### 9. The `__main__` SFT path

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="mlabonne")
    parser.add_argument("--model_output_huggingface_workspace", type=str, default="mlabonne")
    parser.add_argument("--is_dummy", type=bool, default=False, help="Flag to reduce the dataset size for testing")
    parser.add_argument(
        "--finetuning_type",
        type=str,
        choices=["sft", "dpo"],
        default="sft",
        help="Parameter to choose the finetuning stage.",
    )

    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()
    ...
    if args.finetuning_type == "sft":
        print("Starting SFT training...")  # noqa
        base_model_name = "meta-llama/Llama-3.1-8B"
        print(f"Training from base model '{base_model_name}'")  # noqa

        output_dir_sft = Path(args.model_dir) / "output_sft"
        model, tokenizer = finetune(
            finetuning_type="sft",
            model_name=base_model_name,
            output_dir=str(output_dir_sft),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
        )
        inference(model, tokenizer)

        sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
        save_model(model, tokenizer, "model_sft", push_to_hub=True, repo_id=sft_output_model_repo_id)
```

**Key points.**

- **`SM_OUTPUT_DATA_DIR`, `SM_MODEL_DIR`, `SM_NUM_GPUS`** are SageMaker-provided environment variables. Their presence as `default=` means the script assumes it runs inside a SageMaker training container; running locally requires setting them.
- **`--is_dummy` is positional-parsed oddly.** `type=bool` on an argparse string makes almost any non-empty value truthy; pass it deliberately.
- **`inference()` runs immediately after training** as a smoke test.
- Base model is `meta-llama/Llama-3.1-8B` (the book's `meta-llama/Meta-Llama-3.1-8B` is the gated Hub id; this repo uses `meta-llama/Llama-3.1-8B`). Output is `{workspace}/TwinLlama-3.1-8B`.
- **Note the load-in-4bit default:** the `finetune(...)` call omits `load_in_4bit`, so it uses `False` → **LoRA**. This is a correction to any draft that claimed 4-bit was on by default.

---

### 10. `sagemaker.py` — the managed-GPU launcher

```python
# llm_engineering/model/finetuning/sagemaker.py
from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger

try:
    from sagemaker.huggingface import HuggingFace
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.settings import settings

finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def run_finetuning_on_sagemaker(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")
    if not finetuning_requirements_path.exists():
        raise FileNotFoundError(f"The file {finetuning_requirements_path} does not exist.")

    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    hyperparameters = {
        "finetuning_type": finetuning_type,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "dataset_huggingface_workspace": dataset_huggingface_workspace,
        "model_output_huggingface_workspace": huggingface_user,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True

    # Create the HuggingFace SageMaker estimator
    huggingface_estimator = HuggingFace(
        entry_point="finetune.py",
        source_dir=str(finetuning_dir),
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        role=settings.AWS_ARN_ROLE,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        requirements_file=finetuning_requirements_path,
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
    )

    # Start the training job on SageMaker.
    huggingface_estimator.fit()


if __name__ == "__main__":
    run_finetuning_on_sagemaker()
```

**Key points.**

- The SageMaker import is wrapped in `try/except ModuleNotFoundError`, so the module imports locally even without the `aws` extra; it logs a hint to `poetry install --with aws`.
- `ml.g5.2xlarge` has an **A10G with 24 GB**, which comfortably runs the 8B LoRA job that is tight on the RTX 5000's 16 GB.
- `transformers_version="4.36"`, `pytorch_version="2.1"`, `py_version="py310"` pin the container.
- `requirements_file=finetuning_requirements_path` ships `finetuning/requirements.txt` to the container.
- Secrets are injected as environment variables, never written into the hyperparameters.
- Hyperparameters are passed as strings to SageMaker; `finetune.py`'s argparse coerces types.

The local invocation is `python -m llm_engineering.model.finetuning.sagemaker` (or via a Poe task); it requires `AWS_ARN_ROLE` and a Hugging Face token. See [Session 5.3](session_5.3_sagemaker_deployment.md) for deployment of the resulting model.

---

## 🔬 Deep Dive: Memory and Effective Batch Math

### Full fine-tuning baseline (book)

```
bytes/param = weights(2 or 4) + grads(4) + Adam states(8)  ≈ 16 bytes/param (fp32)
```

| Model | 16 bytes/param | Notes |
|-------|----------------|-------|
| 7B | ~112 GB | many high-end GPUs |
| 70B | ~1,120 GB | data-center scale |

Mixed precision and model parallelism push this to roughly 14-15 bytes/param, still far beyond one 16 GB card.

### LoRA memory (worked)

LoRA freezes `W` and trains `B @ A` with rank `r`. Trainable params for a linear layer of shape `(out, in)`:

```
params = r × (in + out)
```

For Llama 3 8B targeting every linear module at `r=16`, the book reports **~42 M trainable** out of 8 B = **0.5196%**. At the repo default `r=32` the adapter roughly doubles, still well under 1% of the base.

Approximate VRAM for an 8B **LoRA** run (frozen base in fp16, adapters fp16):

```
base weights        ~16 GB   (8B × 2 bytes)     ← already near the 16 GB card limit
adapters (r=32)     < 1 GB
gradients (adapters) small
optimizer (8-bit)   small
activations (seq 2048, batch 2) several GB
```

This is why an 8B LoRA run does **not** fit comfortably on a single RTX 5000 at seq 2048/batch 2: the fp16 base alone is ~16 GB. That is the real reason the project defaults to SageMaker and to QLoRA for local experimentation.

### QLoRA memory (book numbers, 7B)

| Phase | LoRA | QLoRA | Saving |
|-------|------|-------|--------|
| Initialization | 14 GB | 9.1 GB | ~35% |
| Fine-tuning | 15.6 GB | 9.3 GB | ~40% |

QLoRA quantizes the frozen base to 4-bit NF4 (≈ 0.5 byte/param + overhead), uses double quantization (quantizing the quantization constants), and paged optimizers to absorb spikes. Cost: ~30% slower training, with only minor quality differences.

Approximate 8B **QLoRA** VRAM:

```
base weights (NF4)  ~4-5 GB
adapters (r=32)     < 1 GB
optimizer (8-bit)   small
activations/grads   several GB
```

Total is roughly 8-11 GB depending on sequence length and batch, which fits 16 GB with headroom.

### Packing math

Unpacked, 2 samples × 2048 max length = 4096 token slots, but if samples are 250 tokens each, 4096 − 500 = 3596 slots are padding (88% waste). Packed, the same slots hold ~16 samples. `packing=True` recovers that throughput for free.

### Precision on the RTX 5000

```
Turing (sm_75):
  bf16            ✗  (is_bfloat16_supported() == False)   → use fp16
  FlashAttention-2 ✗  (needs Ampere sm_80+)
  fp16 / TF32      ✓
```

Consequences:

- `bf16=is_bfloat16_supported()` is `False`, `fp16=not is_bfloat16_supported()` is `True`. The code is correct as written; do not force bf16.
- No FlashAttention-2 means attention falls back to a slower kernel on this card. Throughput and memory guides measured on A100/H100 do not transfer; expect slower steps.
- fp16 training needs loss scaling, which Unsloth/transformers handle automatically.

---

## 🛠️ Hands-On: Prepare and Smoke-Test SFT

### Step 1: Check the dataset exists on the Hub

```python
from datasets import load_dataset

ds = load_dataset("<workspace>/llmtwin", split="train")
print(len(ds), ds.column_names)  # ['instruction', 'output']
print(ds[0])
```

### Step 2: Preview the formatted text

```python
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

ex = ds[0]
print(alpaca_template.format(ex["instruction"], ex["output"]))
```

Confirm the instruction and response land in the right slots.

### Step 3: Launch dummy training locally

SageMaker env vars are required by argparse defaults, so set them for a local run:

```powershell
$env:SM_OUTPUT_DATA_DIR = ".\output_data"
$env:SM_MODEL_DIR = ".\model"
$env:SM_NUM_GPUS = "1"
python llm_engineering/model/finetuning/finetune.py `
    --finetuning_type sft `
    --is_dummy True `
    --num_train_epochs 1 `
    --per_device_train_batch_size 2 `
    --learning_rate 3e-4
```

Watch the loss go down and confirm a `model_sft/` directory appears. On 16 GB, if it OOMs, lower `--per_device_train_batch_size` to 1 and/or set `load_in_4bit=True` by editing the call.

### Step 4: Verify precision on this GPU

```python
import torch

print("bf16 supported:", torch.cuda.is_bf16_supported())  # False on Turing RTX 5000
print("fp16 supported:", torch.cuda.is_available())
```

### Step 5: Inspect the merged output

```python
from pathlib import Path
p = Path("model_sft")
print([f.name for f in p.iterdir()][:10])
# expect config.json, tokenizer.*, model.safetensors (fp16, ~16 GB for 8B)
```

---

## 📝 Exercise 1: Ablation on LoRA Rank

### Task

Measure how LoRA rank affects quality and memory.

1. Run dummy SFT with `lora_rank=8`, `32`, and `64`.
2. Record peak VRAM (`torch.cuda.max_memory_allocated()`) and final training loss.
3. Generate the same prompt from each checkpoint.
4. Decide which rank gives the best quality-per-GB trade-off.

**Goal:** build intuition that rank is the primary capacity knob, and beyond a point it only raises memory (and overfitting risk).

---

## 📝 Exercise 2: LoRA vs QLoRA Head-to-Head

### Task

Compare LoRA and QLoRA on the same data and hardware.

1. Run dummy SFT twice on the RTX 5000: once with `load_in_4bit=False` (LoRA), once with `load_in_4bit=True` (QLoRA), same `r=32`, `max_seq_length=1024`, batch 2.
2. Record peak VRAM, seconds/step, and final train/eval loss for each.
3. Generate a fixed prompt from both and compare fluency and style adherence.
4. Write a recommendation: at what sequence length or model size would you switch from LoRA to QLoRA on 16 GB?

**Goal:** internalize the book's trade-off — QLoRA saves ~40% memory for ~30% more time with minor quality loss. On the RTX 5000, an 8B fp16 base (~16 GB) makes LoRA borderline impossible at long sequences, so QLoRA is usually the local default, while LoRA is preferred on the 24 GB SageMaker A10G.

**Deliverable:** a table of `variant × {peak VRAM, s/step, final loss}` plus a one-paragraph recommendation.

---

## 🐛 Common Pitfalls

- **Import order.** Moving `PatchDPOTrainer()` below the TRL imports silently breaks patching. Keep the ordering and the `# noqa: E402` markers.
- **bf16 on Turing.** `is_bfloat16_supported()` is `False` on the RTX 5000; the code correctly falls back to fp16. Never force bf16 here.
- **No FlashAttention-2.** sm_75 cannot use it; do not install/enable it expecting a speedup.
- **Missing EOS.** Forgetting `+ EOS_TOKEN` produces a model that never stops generating.
- **Confusing LoRA and QLoRA defaults.** The repo default is `load_in_4bit=False` → LoRA. Set it `True` for QLoRA.
- **8B fp16 base ≈ 16 GB.** A LoRA run at seq 2048/batch 2 does not fit cleanly on the RTX 5000; QLoRA or SageMaker is the pragmatic path.
- **SageMaker env vars at import.** `os.environ["SM_*"]` in argparse defaults raises `KeyError` locally if unset. Always export them for local runs.
- **`--is_dummy` parsing.** `type=bool` misparses; be explicit and verify the printed "Training in dummy mode" message.
- **Packing + very short samples.** Packing is beneficial, but keep `max_seq_length` bounded; extremely long packed sequences can still OOM.
- **gated base model.** `meta-llama/Llama-3.1-8B` requires accepting Meta's license and a valid `HUGGINGFACE_ACCESS_TOKEN`.
- **Comet offline.** `report_to="comet_ml"` without `COMET_API_KEY` will warn; set it or change `report_to`.
- **`save_model` vs `save_pretrained`.** `merged_16bit` produces a ~16 GB standalone model; the raw adapter would be far smaller but needs PEFT-aware serving (Session 8.4).

---

## 🎓 Knowledge Check

1. **Why is `PatchDPOTrainer()` called before the TRL imports?**
   It monkey-patches TRL classes; the patch must exist before those classes are imported.

2. **What does `load_in_4bit` do, and what stays trainable?**
   `True` quantizes the frozen base to 4-bit NF4 (QLoRA); only the LoRA adapters are trained. `False` keeps the base in its loaded precision (LoRA).

3. **Why concatenate a general Alpaca dataset with the domain dataset?**
   The domain set is too small to teach the chat template and it risks catastrophic forgetting; general data preserves instruction-following.

4. **What is `packing=True` and why does it help?**
   It concatenates short samples into full-length sequences, removing padding waste and raising tokens/step.

5. **Which precision is used on the RTX 5000, and why?**
   fp16, because Turing (sm_75) does not support bfloat16.

6. **What does `save_method="merged_16bit"` produce?**
   A standalone fp16 model with the adapter merged into the base weights.

7. **Compute the effective batch for batch=2, grad_accum=8, 1 GPU.**
   2 × 1 × 8 = 16.

8. **What is the default LoRA rank and alpha in this repo, and why?**
   `r=32, alpha=32`. Alpha equal to rank keeps the update scale near 1; rank 32 is enough to copy style and knowledge.

9. **Which two target-module groups does LoRA adapt, and why the MLP ones?**
   Attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`). Adapting the MLP projections is what lets a small rank capture style changes.

10. **Roughly how many trainable params does rank-16 LoRA add to Llama 3 8B?**
    ~42 M, or ~0.52% of the model (book).

11. **Why does QLoRA save memory, and what does it cost?**
    It quantizes the base to 4-bit NF4 with double quantization and paged optimizers; it is ~30% slower with minor quality loss.

12. **Why `+ EOS_TOKEN` on every training sample?**
    So the model learns where an answer ends; without it generation never stops.

13. **What does `alpaca_template.format(prompt, "")` do at inference?**
    Leaves `### Response:` at the end, forcing the model to answer instead of continue.

14. **Which instance type does `sagemaker.py` use, and what GPU?**
    `ml.g5.2xlarge`, one A10G with 24 GB.

15. **Why can't the RTX 5000 use FlashAttention-2?**
    It needs Ampere (sm_80) or newer; Turing is sm_75.

---

## 📖 Glossary

- **SFT** — supervised fine-tuning on (instruction, answer) pairs to teach instruction following.
- **Alpaca template** — a plain-text instruction/response format used here as both data format and chat template.
- **ChatML** — a chat template using `<|im_start|>` / `<|im_end|>` delimiters.
- **EOS token** — the end-of-sequence token that terminates an answer.
- **LoRA** — Low-Rank Adaptation; freezes the base and trains two small matrices per targeted layer.
- **QLoRA** — LoRA on a 4-bit NF4-quantized frozen base, with double quantization and paged optimizers.
- **NF4** — 4-bit NormalFloat, the quantile-based data type used by QLoRA.
- **Rank (r)** — size of the LoRA update matrices; the capacity knob.
- **Alpha** — LoRA scaling factor; commonly set equal to or twice the rank.
- **Effective batch size** — `per_device_batch × devices × grad_accum_steps`.
- **Gradient accumulation** — summing gradients over several micro-batches before one update.
- **Packing** — concatenating short samples to fill sequence slots.
- **Gradient checkpointing** — recomputing activations to save memory at the cost of time.
- **Catastrophic forgetting** — loss of pre-trained knowledge after destructive fine-tuning.
- **Merged 16-bit** — the base-plus-adapter model saved as standalone fp16 weights.
- **`is_bfloat16_supported()`** — Unsloth helper; `False` on Turing, so fp16 is used.
- **`ml.g5.2xlarge`** — the SageMaker instance (24 GB A10G) the project trains on.

---

## 🔗 Next Session

**Session [5.2](session_5.2_dpo.md): Direct Preference Optimization (DPO)**

We align the SFT model with the preference dataset using `DPOTrainer`, reusing the same `load_model` and `PatchDPOTrainer` plumbing.

Related: [Session 3.1](session_3.1_instruction_dataset.md) (instruction data), [Session 5.3](session_5.3_sagemaker_deployment.md) (deploying the merged model), [Session 7.1](session_7.1_comet_ml.md) (Comet tracking), [Session 8.4](session_8.4_inference_optimization.md) (serving adapters vs merged).

---

## 📚 Additional Resources

- [Unsloth documentation](https://docs.unsloth.ai/)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al.)](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.)](https://arxiv.org/abs/2106.09685)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Alpaca / FineTome dataset](https://huggingface.co/datasets/mlabonne/FineTome-Alpaca-100k)
- Source: `llm_engineering/model/finetuning/finetune.py`, `llm_engineering/model/finetuning/sagemaker.py`.

---

**Estimated Time**: 5-6 hours

**Prerequisites**: [Session 3.1](session_3.1_instruction_dataset.md)

**Outcome**: You can configure and run a memory-efficient SFT job, choose LoRA versus QLoRA on the RTX 5000, explain every trainer hyperparameter, and launch the same script on SageMaker.
