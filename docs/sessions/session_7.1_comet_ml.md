# Session 7.1: Experiment Tracking with Comet ML

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how the trainings report metrics to Comet ML
- Configure Comet ML locally and inside SageMaker
- Compare SFT and DPO runs side by side
- Know which metrics to watch during LLM fine-tuning
- Recognize that Opik traces and Comet experiments share one account
- Explain why an experiment tracker exists and what it buys you over console logs
- Read every auto-logged hyperparameter and interpret the DPO reward charts
- Add custom metrics, tags, and artifacts with the Comet SDK
- Diagnose runs that fail to appear or land in the wrong project
- Reason about cost, privacy, and reproducibility tradeoffs of hosted tracking

---

## ✅ Prerequisites

- **Sessions 5.1 (SFT)** and **5.2 (DPO)** — the training scripts this session instruments.
- A free [Comet ML account](https://www.comet.com/) and an API key.
- `.env` with `COMET_API_KEY` and `COMET_PROJECT`.
- For the SageMaker path: `settings.HUGGINGFACE_ACCESS_TOKEN` and `settings.AWS_ARN_ROLE`,
  and the `aws` dependency group installed (`poetry install --with aws`).
- Familiarity with the Hugging Face `Trainer` / `TrainingArguments` / `DPOConfig` API.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      Comet ML account                             │
│                                                                    │
│  Comet Experiments (training metrics)                              │
│      ▲                    ▲                                        │
│      │ report_to          │ report_to                              │
│      │ "comet_ml"         │ "comet_ml"                              │
│  SFT TrainingArguments    DPOConfig                               │
│      ▲                    ▲                                        │
│      └────────┬───────────┘                                        │
│          COMET_API_KEY  +  COMET_PROJECT / COMET_PROJECT_NAME      │
│                                                                    │
│  Opik (prompt/LLM tracing, Session 7.2)                            │
│      ▲                                                             │
│      └── same COMET_API_KEY, project routed via OPIK_PROJECT_NAME  │
└──────────────────────────────────────────────────────────────────┘
```

One vendor, two products: **Comet ML** for experiment metrics, **Opik** for LLM/prompt
tracing. Both authenticate with `COMET_API_KEY`.

### Where tracking sits in the training lifecycle

```
data prep ──▶ train (Trainer) ──▶ evaluate ──▶ save/push ──▶ deploy
                  │
                  └── report_to="comet_ml" ──▶ Comet experiment
                                                 (metrics, params,
                                                  system stats,
                                                  artifacts)
```

The integration is a single argument. Everything else — metric streaming, hyperparameter
capture, system utilization, chart rendering — is automatic once the Comet integration is
installed and the key is present.

### Why a tracker instead of logs

Training an LLM is iterative: you launch many runs, compare them on metrics, and pick one for
production. Console logs and TensorBoard directories do not scale to dozens of runs across
fresh containers (SageMaker, Colab, Vast.ai). A tracker gives you:

- **A shared namespace** for runs across machines, with no file copying.
- **Comparison views**: overlay `loss` from SFT and DPO runs.
- **Hyperparameter capture** so a chart can be sliced by learning rate or batch size.
- **System metrics** (GPU/CPU/RAM) that reveal whether a run is compute- or memory-bound.
- **Artifacts**: checkpoint directories, model repos, evaluation reports.

The book's rationale: Comet differentiates on ease of use and an intuitive interface; W&B,
MLflow, and Neptune have comparable features.

---

## 📁 Key Files Explained

### 1. `settings.py` - Comet Configuration

```python
# llm_engineering/settings.py
    # Comet ML (during training)
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "twin"
```

**Key Concepts**:
- **`COMET_API_KEY`** is required to log anywhere. It is `None` by default, so an unset key
  is a silent misconfiguration unless you watch the logs.
- **`COMET_PROJECT`** defaults to `twin`. Locally the trainer uses this as the Comet project;
  in the SageMaker job it is passed as `COMET_PROJECT_NAME` (the env var name Comet's HF
  integration reads inside the container).
- Settings are loaded from `.env` via `pydantic_settings`
  (`SettingsConfigDict(env_file=".env")`), or from the ZenML secret store if `load_settings`
  finds one. That ordering matters: **ZenML secrets win over `.env`** when present.

```python
# llm_engineering/settings.py (loading order)
    @classmethod
    def load_settings(cls) -> "Settings":
        try:
            logger.info("Loading settings from the ZenML secret store.")
            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError):
            logger.warning(
                "Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file."
            )
            settings = Settings()
        return settings

settings = Settings.load_settings()
```

**Consequence**: if a stale `settings` secret exists in ZenML, editing `.env` will appear to
have no effect. This is one of the most confusing failure modes in the whole project.

---

### 2. SFT Reporting

```python
# llm_engineering/model/finetuning/finetune.py (SFT TrainingArguments)
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
```

**Key Concepts**:
- **`report_to="comet_ml"`** is the single switch. Hugging Face `Trainer` detects the Comet
  integration and starts an experiment automatically.
- **`logging_steps=1`** logs every step; useful for small/dummy runs and for spotting
  divergence early. On a large run it generates a lot of points and can slow logging.
- **`output_dir`** is the local checkpoint directory; it is also logged as an artifact
  location.
- All `TrainingArguments` fields are auto-logged as hyperparameters, so the experiment
  dashboard shows the full config. This includes `optim="adamw_8bit"`, the precision flags,
  and the warmup schedule.
- **`seed=0`** makes runs reproducible; log it so two similar-looking runs are not mistaken
  for duplicates.

> **Precision note.** `fp16=not is_bfloat16_supported()` and
> `bf16=is_bfloat16_supported()` are mutually exclusive. On a Turing GPU (RTX 5000) bf16 is
> not supported, so `fp16=True, bf16=False`. On Ampere+ (SageMaker `ml.g5`) the reverse. The
> chosen precision is recorded in the run config, which is why tracking matters across
> different hardware.

---

### 3. DPO Reporting

```python
# finetune.py (DPOConfig)
                args=DPOConfig(
                    learning_rate=learning_rate,
                    num_train_epochs=num_train_epochs,
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
```

**Key Concepts**:
- Same `report_to="comet_ml"`, so SFT and DPO runs land in the **same project** and can be
  overlaid.
- **`eval_steps=0.2`** produces 5 evaluation points per epoch, which gives enough resolution
  to watch the preference loss fall.
- The DPO run uses `learning_rate=2e-6` and `num_train_epochs=1` in `__main__` — vastly
  smaller than SFT's `3e-4`, 3 epochs. Comparing the two on a raw loss axis is therefore
  misleading; compare within a task.

**DPO-specific metrics to watch**:
| Metric | Meaning | Healthy signal |
|--------|---------|----------------|
| `loss` | DPO objective | decreasing |
| `rewards/chosen` | log-prob advantage of chosen | rising |
| `rewards/rejected` | log-prob advantage of rejected | falling |
| `rewards/accuracies` | fraction where chosen outranks rejected | rising toward 1 |
| `rewards/margins` | chosen − rejected gap | rising |
| `logps/chosen` / `logps/rejected` | raw log-probs | diverge from reference |
| `grad_norm` | gradient magnitude | stable, not exploding |

#### Reading the reward charts

`rewards/accuracies` is the most interpretable single number: it is the fraction of
preference pairs the policy ranks correctly relative to the reference. If it stays at 0.5,
the model is not learning the preference; if it hits 1.0 quickly, you may be over-fitting the
preference set. `rewards/margins` widening is expected; a margin that grows without bound
suggests the KL anchoring (controlled by `beta`) is too weak.

```python
# finetune.py — the beta default that anchors DPO to the reference model
def finetune(
    ...
    beta: float = 0.5,  # Only for DPO
    ...
):
```

`beta=0.5` is the KL penalty strength. Higher `beta` keeps the policy closer to the reference
(less reward hacking, slower learning); lower `beta` lets it drift further.

---

### 4. SageMaker Environment Pass-Through

```python
# llm_engineering/model/finetuning/sagemaker.py
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
```

Full context:

```python
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
```

**Key Concepts**:
- **`COMET_PROJECT_NAME`** (not `COMET_PROJECT`) is the variable Comet's Hugging Face
  integration reads inside the training container. The `.env` name and the container name
  differ; the pass-through bridges them.
- **`COMET_API_KEY`** is injected as an environment variable so logs flow from the remote job
  back to your account.
- Locally, Comet reads `COMET_API_KEY` and `COMET_PROJECT` from `.env` directly.
- The `source_dir` is the finetuning package directory; `entry_point="finetune.py"` is the
  script SageMaker runs. `requirements_file` installs the training dependencies (including
  the Comet integration) in the container.
- The same key is used later by Opik in the business service (Session 7.2), so one credential
  covers both products.

---

### 5. Why the integration is "invisible"

Comet's HF integration registers itself as a `TrainerCallback` when `comet_ml` is importable
and `report_to` includes `"comet_ml"`. It hooks `on_log`, `on_train_begin`, `on_train_end`,
and so on, and forwards `state.log_history` entries to the experiment. That is why there is
no explicit `experiment.log_metric` call in `finetune.py`: the HF callback does it. The
environment variables `COMET_API_KEY` and `COMET_PROJECT_NAME` are read at experiment
creation time.

```
Trainer.train()
   │
   ├─ on_train_begin  ──▶ CometExperiment(project=COMET_PROJECT_NAME, api_key=...)
   ├─ on_log (each logging_steps) ──▶ log_metrics({loss, learning_rate, grad_norm, ...})
   └─ on_train_end    ──▶ finalize, upload artifacts
```

---

## 🔬 Deep Dive: What Gets Logged, Automatically

| Category | Examples | Source |
|----------|----------|--------|
| Scalar metrics | `loss`, `learning_rate`, `grad_norm`, `epoch` | Trainer log history |
| Eval metrics | `eval_loss`, DPO `rewards/*` | Trainer evaluation |
| Hyperparameters | every `TrainingArguments`/`DPOConfig` field | Config serialization |
| System metrics | GPU utilization, GPU memory, CPU, RAM | Comet system monitor |
| Artifacts | `output_dir` checkpoints | Trainer `output_dir` |
| Metadata | git commit (when available), script name | Integration |

The book highlights training/eval loss and gradient norm as the core tracked metrics, plus
hyperparameters and out-of-the-box system metrics such as GPU/CPU/memory utilization. Those
system metrics answer "what resources do I need and where is the bottleneck?"

### A worked comparison: two SFT runs

Suppose you change only `learning_rate`:

| Run | lr | epochs | final train loss | eval loss | note |
|-----|----|--------|------------------|-----------|------|
| A | 3e-4 | 1 (dummy) | 1.62 | 1.58 | baseline |
| B | 1e-4 | 1 (dummy) | 1.71 | 1.63 | slower |

Slicing the loss chart by `learning_rate` shows B descending more slowly. Because every
argument is a hyperparameter, this slice is available with no extra code. On a real run you
would also overlay `grad_norm`: a spike coinciding with a loss jump means the learning rate
is too high or warmup is too short.

---

## 🛠️ Hands-On: Run a Tracked Experiment

### Step 1: Configure the environment

```env
COMET_API_KEY=your-comet-key
COMET_PROJECT=twin
```

### Step 2: Run a dummy SFT experiment

```bash
python llm_engineering/model/finetuning/finetune.py --finetuning_type sft --is_dummy True
```

`is_dummy=True` sets `num_train_epochs=1` and trims the dataset to 400 samples. Note the
`argparse` default for `--is_dummy` is `False`, and `type=bool` means any non-empty string
(including `"False"`) is truthy — pass `True` deliberately.

### Step 3: Open the Comet dashboard

Go to `https://www.comet.com/` and open the `twin` project. You should see:
- A run named after the script/session.
- **Charts**: `loss`, `learning_rate`, and for DPO also `rewards/*`.
- **Hyperparameters**: every `TrainingArguments` field.
- **System metrics**: GPU utilization and memory (if enabled).

### Step 4: Compare runs

Select the SFT and DPO runs and overlay the `loss` chart. Note that the DPO loss is on a
different scale and only runs one epoch.

### Step 5: Verify the environment reached the container (SageMaker path)

```bash
python -m llm_engineering.model.finetuning.sagemaker
```

Before the job starts, confirm the estimator's `environment` map includes `COMET_API_KEY` and
`COMET_PROJECT_NAME`. If the run appears under a project literal-named `"None"`, the key was
missing at container start.

---

## 📝 Exercise: Add a Custom Metric Callback

### Task

Log the average training loss manually at the end, plus a tag.

```python
from transformers import TrainerCallback
import comet_ml  # noqa: F401


class CometSummaryCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        experiment = comet_ml.get_global_experiment()
        if experiment:
            experiment.log_metric("final_train_loss", state.log_history[-1].get("loss"))
            experiment.add_tag("twin-finetune")
```

1. Pass `callbacks=[CometSummaryCallback()]` to `SFTTrainer`.
2. Re-run a dummy job and confirm `final_train_loss` and the tag appear.

**Goal**: Learn the Comet API for custom metrics and tags beyond what the HF integration logs
automatically.

---

## 🧪 Second Exercise: Log a Model Artifact and a Confusion Matrix

### Task

After training, upload the merged model directory as a Comet artifact and log a small eval
table from `Session 7.3`'s judge output.

```python
import comet_ml
from pathlib import Path


def log_training_artifacts(output_dir: str, tag: str = "twin-artifact"):
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return

    artifact = comet_ml.Artifact(name=tag, artifact_type="model")
    artifact.add(Path(output_dir) / "config.json")
    artifact.add(Path(output_dir) / "model.safetensors")
    experiment.log_artifact(artifact)

    # A tiny evaluation table (replace with real judge scores from Session 7.3).
    rows = [
        ["metric", "sft", "dpo"],
        ["helpfulness", 0.82, 0.88],
        ["faithfulness", 0.79, 0.84],
    ]
    experiment.log_table("eval_table.csv", tabular_data=rows)

    # Link versioned code to the run for reproducibility.
    experiment.log_other("output_dir", output_dir)
```

1. Call `log_training_artifacts(output_dir)` after `trainer.train()`.
2. Re-run a dummy job; confirm the artifact and table appear on the experiment page.
3. Open two runs' tables and compare their metric columns.

**Goal**: Move from scalar metrics to full experiment lineage: the weights, the eval table,
and the code/config needed to reproduce the run.

---

## 🐛 Common Pitfalls

- **Wrong env var in SageMaker**: using `COMET_PROJECT` instead of `COMET_PROJECT_NAME` in
  the container means runs land in the default project. The code correctly uses
  `COMET_PROJECT_NAME`.
- **Missing key**: without `COMET_API_KEY`, `report_to="comet_ml"` logs a warning and
  training continues untracked.
- **Dummy vs real scale**: dummy runs (1 epoch, 400 samples) are for plumbing checks, not for
  comparing model quality.
- **Comparing incompatible runs**: SFT and DPO have different metric spaces; compare
  `rewards/*` within DPO and use the Session 7.3 evaluation for cross-model quality.
- **Stale ZenML secret shadowing `.env`**: `load_settings` prefers the ZenML secret store, so
  an old `settings` secret overrides your edited `.env`. Delete it with
  `zenml secret delete settings` if changes do not take effect.
- **`--is_dummy False` is truthy**: `argparse`'s `type=bool` does not parse the string
  `"False"`; it is non-empty and therefore `True`. Omit the flag to keep the default `False`.
- **Logging every step at scale**: `logging_steps=1` is fine for dummies but produces huge
  point counts and network chatter on long runs. Raise it for real training.
- **Precision mismatch across hosts**: the same recipe on a Turing box (fp16) and an Ampere
  box (bf16) can produce slightly different curves. Keep precision in mind when comparing.
- **Assuming the run name is unique**: two runs with the same name are distinct experiments;
  tag or namespace them (see the custom callback).
- **API key in the wrong scope**: the key must belong to the workspace that owns the project.

---

## 🧭 Edge Cases and Failure Modes

| Scenario | Symptom | Fix |
|----------|---------|-----|
| No `COMET_API_KEY` | Training runs, nothing in Comet | Set the key in `.env`/ZenML |
| SageMaker uses `COMET_PROJECT` | Run in default project | Use `COMET_PROJECT_NAME` |
| ZenML secret exists | `.env` edits ignored | `zenml secret delete settings` |
| `--is_dummy False` | Full dataset unexpectedly | Omit flag; default is `False` |
| Network blocked from container | Metrics buffer/drop | Allow egress; check container logs |
| `comet_ml` not installed | `report_to` warns, no run | Add the integration to requirements |
| Run appears twice | Duplicate experiment names | Distinguish by tag/hyperparams |
| DPO `rewards/accuracies` flat | No preference learning | Check `beta`, LR, data pairing |

---

## 🎓 Knowledge Check

1. **What single argument enables Comet logging in the trainers?**
   - Answer: `report_to="comet_ml"`.

2. **Which env var name does the SageMaker container need for the project?**
   - Answer: `COMET_PROJECT_NAME`.

3. **Which DPO metric tells you the chosen answer is being preferred more often?**
   - Answer: `rewards/accuracies` (and `rewards/margins`).

4. **What does `logging_steps=1` do?**
   - Answer: Logs a point every training step.

5. **Do Opik and Comet experiments use different credentials?**
   - Answer: No, both use `COMET_API_KEY`.

6. **Why can SFT and DPO runs be compared directly in Comet?**
   - Answer: Both report to the same project, so their charts can be overlaid.

7. **Why is there no explicit `log_metric` call in `finetune.py`?**
   - Answer: Comet's Hugging Face integration registers a `TrainerCallback` that forwards the
     Trainer's log history automatically.

8. **What does `settings.load_settings()` prefer over the `.env` file?**
   - Answer: The ZenML secret store, if a `settings` secret exists.

9. **What does DPO's `beta` control?**
   - Answer: The KL penalty anchoring the policy to the reference model; higher beta means
     less drift.

10. **Why can a Turing GPU and an Ampere GPU produce different curves for the same recipe?**
    - Answer: Different default precision (fp16 vs bf16), captured in the run config.

11. **What is the risk of `logging_steps=1` on a long run?**
    - Answer: A very large number of logged points, added overhead, and network chatter.

12. **How do you reproduce a specific run?**
    - Answer: Use its logged hyperparameters, seed, git metadata, and artifacts.

13. **What does `--is_dummy False` actually do, given `type=bool`?**
    - Answer: It evaluates to `True` (non-empty string), so the dummy path runs. Omit the
      flag for real training.

14. **Where do system metrics (GPU/CPU/RAM) come from?**
    - Answer: Comet's system monitor, enabled out of the box.

15. **Which tracker fields map to cost analysis?**
    - Answer: System metrics plus token/sample counts; combined with instance-type
      hyperparameters.

---

## 📖 Glossary

- **Experiment tracker** — a system that records runs, metrics, params, and artifacts.
- **Run / experiment** — one execution of a training script.
- **Hyperparameter** — a config value (learning rate, batch size) not learned by training.
- **`report_to`** — HF Trainer argument listing integrations that receive logs.
- **TrainerCallback** — an HF hook invoked at lifecycle events; how integrations attach.
- **System metrics** — resource utilization (GPU, CPU, RAM) captured during a run.
- **Artifact** — a versioned file/directory attached to a run (weights, tables, reports).
- **DPO** — Direct Preference Optimization; aligns a model to chosen/rejected pairs.
- **`beta`** — DPO KL penalty strength relative to the reference model.
- **Gradient norm** — magnitude of gradients; a spike signals instability.
- **ZenML secret** — a stored settings bundle that can shadow `.env`.

---

## 🔗 Next Session

**Session 7.2**: Prompt Monitoring with Opik

We inspect the traces generated by the RAG inference path.

Related reading in this repo:
- [Session 5.1: SFT](session_5.1_sft.md)
- [Session 5.2: DPO](session_5.2_dpo.md)
- [Session 5.3: SageMaker Deployment](session_5.3_sagemaker_deployment.md)
- [Session 7.2: Opik Monitoring](session_7.2_opik_monitoring.md)

---

## 📚 Additional Resources

- [Comet ML Documentation](https://www.comet.com/docs/v2/)
- [Comet ML + Hugging Face integration](https://www.comet.com/docs/v2/integrations/ml-frameworks/hugging-face/)
- [Hugging Face Trainer integrations](https://huggingface.co/docs/transformers/main_classes/trainer#integrations)
- [TRL DPO metrics](https://huggingface.co/docs/trl/dpo_trainer)

---

## 📑 References

- Iusztin, P. & Labonne, M. *LLM Engineer's Handbook.* Packt, 2024. **Chapter 2, "Tooling and
  Installation"** (pp. 74-76): Comet ML as the experiment tracker, system metrics, the public
  LLM Twin training dashboard. **Chapter 11, "MLOps and LLMOps"**: experiment tracking within
  the LLMOps loop.
- Repo source of truth: `llm_engineering/settings.py`,
  `llm_engineering/model/finetuning/finetune.py`,
  `llm_engineering/model/finetuning/sagemaker.py`.
- Public LLM Twin training experiments: https://www.comet.com/mlabonne/llm-twin-training

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 5.1, 5.2

**Outcome**: You can configure Comet ML locally and on SageMaker, interpret SFT/DPO training
charts, and extend logging with custom metrics, tags, and artifacts.
