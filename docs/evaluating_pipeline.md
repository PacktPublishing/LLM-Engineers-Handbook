# `evaluating` Pipeline

Triggered by `poetry poe run-evaluation-pipeline`. This pipeline runs *after* `training`. It launches an AWS SageMaker **Training** job (used here only as a GPU runner; nothing is trained) that generates answers with three models (your SFT model, your DPO model, and the `Llama-3.1-8B-Instruct` baseline) on the `test` split of the instruct dataset, scores each answer with an LLM judge (GPT-4o-mini by default) for accuracy and style, and prints an average score per model.

## Pipeline task

There is a single, non-composite poe task ([pyproject.toml:90](../pyproject.toml#L90)):

```toml
run-evaluation-pipeline = "poetry run python -m tools.run --no-cache --run-evaluation"
```

- `--run-evaluation` tells [main()](../tools/run.py#L113) to execute the `if run_evaluation:` branch ([tools/run.py:192-196](../tools/run.py#L192-L196)), which loads [configs/evaluating.yaml](../configs/evaluating.yaml)
- `--no-cache` disables ZenML step caching for the run

The config has a single parameter:

| Parameter | `evaluating.yaml` | Notes |
| --- | --- | --- |
| `is_dummy` | `false` | `true` evaluates only the first 10 test samples per model (see Key read); `false` uses the full `test` split |

The model and dataset namespaces are **not** in the config. Both come from the account that owns `HUGGINGFACE_ACCESS_TOKEN` (see Key read).

The step needs `HUGGINGFACE_ACCESS_TOKEN`, `OPENAI_API_KEY` and `AWS_ARN_ROLE` (all three are asserted in [sagemaker.py:30-32](../llm_engineering/model/evaluation/sagemaker.py#L30-L32)), and the AWS extras installed (`poetry install --with aws`). The token must have access to the gated `meta-llama/Llama-3.1-8B-Instruct` repo and write access to your own namespace. `OPENAI_BASE_URL` and `OPENAI_MODEL_ID` are forwarded to the job, so the judge can be any OpenAI-compatible endpoint.

## Diagram

```mermaid
flowchart TD
    A["poetry poe run-evaluation-pipeline"] --> B["tools/run.py: main()"]
    B --> C["Load config YAML\n(evaluating.yaml)"]
    C --> D["pipelines/evaluating.py\n@pipeline evaluating(is_dummy)"]

    D --> E["step: evaluate"]
    E --> F["run_evaluation_on_sagemaker():\nHfApi().whoami() -> hf_user\nbuild env vars"]
    F --> G["SageMaker HuggingFace estimator\nentry_point=evaluate.py, ml.g5.2xlarge\n.fit()"]

    G --> H["SageMaker job: evaluate.py"]
    H --> I["resolve model_ids\n(fallback to mlabonne/* if missing)"]
    I --> M1["{hf_user}/TwinLlama-3.1-8B"]
    I --> M2["{hf_user}/TwinLlama-3.1-8B-DPO"]
    I --> M3["meta-llama/Llama-3.1-8B-Instruct"]

    M1 --> J
    M2 --> J
    M3 --> J
    K[("Hugging Face Hub\n{hf_user}/llmtwin (test split)\nfallback: mlabonne/llmtwin")] --> J

    J["Phase 1, per model: generate_answers()\nAlpaca prompt (TwinLlama) or chat template (baseline)\n-> vLLM generate (temp 0.8, max 2048 tokens)"]
    J --> R[("Hugging Face Hub\n{hf_user}/{model}-results\ninstruction, prompt, answers")]

    R --> L["Phase 2, per model: evaluate_answers()\nbatches of 5, 10 threads\nLLM judge (OPENAI_MODEL_ID)\n-> accuracy 1-3, style 1-3"]
    L --> R2[("Hugging Face Hub\n{hf_user}/{model}-results\n+ evaluation, accuracy, style")]

    R2 --> P["Phase 3, per model:\nprint mean accuracy / mean style\n(SageMaker job logs only)"]
```

## Key read

Three things about this pipeline aren't obvious from the poe task:

1. **The ZenML pipeline is only a launcher, and the results never reach ZenML.** Like `training`, it has one step, `evaluate`, which returns `None`. The step builds a SageMaker `HuggingFace` estimator and calls `.fit()` ([sagemaker.py:57-72](../llm_engineering/model/evaluation/sagemaker.py#L57-L72)). It is submitted as a *Training* job only because accounts often have a 0-instance quota for `ml.g5.2xlarge for processing job usage` while the training quota is already available. Nothing is trained, and it doesn't go through Comet ML. `.fit()` blocks and streams the job's logs, including the final scores, to your terminal. `max_run` caps the job at 4 hours. The real work happens in [evaluate.py](../llm_engineering/model/evaluation/evaluate.py) on an `ml.g5.2xlarge` GPU instance. There are two outputs:
   - One Hub dataset per model, `{hf_user}/{model}-results`, holding each instruction, the model's answer, the judge's raw JSON, and the `accuracy` / `style` scores.
   - The final averages, which are only `print`ed ([evaluate.py:266-279](../llm_engineering/model/evaluation/evaluate.py#L266-L279)). To see them, read the terminal output of the run, or the job's logs in CloudWatch (log group `/aws/sagemaker/TrainingJobs`, job name prefix `evaluate-llm-twin`), or compute them yourself from the results datasets.

2. **Every namespace comes from the token owner, and everything silently falls back to `mlabonne`.** The launcher sets both `DATASET_HUGGINGFACE_WORKSPACE` and `MODEL_HUGGINGFACE_WORKSPACE` to `HfApi().whoami()` ([sagemaker.py:39-55](../llm_engineering/model/evaluation/sagemaker.py#L39-L55)). `training` reads datasets from `dataset_huggingface_workspace` in its config instead, so this pipeline has nothing equivalent to that setting. Then:
   - If `{hf_user}/TwinLlama-3.1-8B` or `{hf_user}/TwinLlama-3.1-8B-DPO` doesn't exist, the job evaluates `mlabonne/TwinLlama-3.1-8B` or `mlabonne/TwinLlama-3.1-8B-DPO` instead ([evaluate.py:244-252](../llm_engineering/model/evaluation/evaluate.py#L244-L252)).
   - If `{hf_user}/llmtwin` doesn't exist, it uses `mlabonne/llmtwin` ([evaluate.py:257-259](../llm_engineering/model/evaluation/evaluate.py#L257-L259)).
   - The fallbacks only print a message and don't stop the job, so check the job log for `Defaulting to` before you trust the scores.

   Results always go to your own namespace, so a fallback run still writes `{hf_user}/TwinLlama-3.1-8B-results`, which looks exactly like a run on your own model.

3. **Only the instruct dataset's `test` split is used.** This is the 10% split `generate_datasets` held back and `training` never touched. The preference (`llmtwin-dpo`) dataset isn't used, so the DPO model is judged on the same instructions as the SFT model. The models are compared, not tested against reference answers. The judge scores each answer on its own, with no access to the dataset's ground-truth `output`.

Other details worth knowing:

- **Dummy mode.** With `is_dummy: true` the launcher sets `IS_DUMMY=True` and each model answers only the first 10 test samples ([evaluate.py:39-43](../llm_engineering/model/evaluation/evaluate.py#L39-L43)). With `false` the variable is left unset, so the full `test` split is used.
- **Each model gets the prompt format it expects.** The TwinLlama models receive the Alpaca-style `### Instruction: / ### Response:` text they were fine-tuned on. `Llama-3.1-8B-Instruct` is listed in `CHAT_TEMPLATE_MODELS` ([evaluate.py:29](../llm_engineering/model/evaluation/evaluate.py#L29)) and gets its own chat template, with generation stopped at `<|eot_id|>` ([evaluate.py:51-66](../llm_engineering/model/evaluation/evaluate.py#L51-L66)). Without this, the baseline doesn't know where its answer ends and keeps writing up to `max_tokens` (about 7,000 characters per answer in an earlier run), which penalises its style score for a format mismatch rather than for the model itself. The exact prompt each model received is stored in the `prompt` column of its results dataset.
- **The scores are noisy.** Generation samples at `temperature=0.8` and the judge runs at `temperature=0.9` ([evaluate.py:69](../llm_engineering/model/evaluation/evaluate.py#L69), [evaluate.py:147](../llm_engineering/model/evaluation/evaluate.py#L147)), so two runs on 10 dummy samples can give noticeably different averages.
- **Unparseable judge replies are skipped in the averages.** Missing scores are stored as `None` ([evaluate.py:196-199](../llm_engineering/model/evaluation/evaluate.py#L196-L199)) and excluded from the final means, and the log reports how many were skipped. An API error inside `evaluate_answer()` is still re-raised by `future.result()` and stops the job; because phase 1 has already pushed the answers, you can re-run only phases 2 and 3.
- **Unset settings are dropped from the job env.** SageMaker only accepts string values, so the launcher removes empty settings such as an unset `OPENAI_BASE_URL` ([sagemaker.py:55](../llm_engineering/model/evaluation/sagemaker.py#L55)); `evaluate.py` then uses its defaults.
- **Judge cost.** The job makes one judge call per answer: 3 models × N test samples (30 calls in dummy mode). With the default `gpt-4o-mini` this is cheap, but a larger `OPENAI_MODEL_ID` multiplies it.

## Function call reference

| Diagram node | Function | Location |
| --- | --- | --- |
| `main()` | `main()` | [tools/run.py:113](../tools/run.py#L113) |
| branch that builds and runs the pipeline | `if run_evaluation:` block | [tools/run.py:192-196](../tools/run.py#L192-L196) |
| `evaluating(is_dummy)` | `evaluating()` | [pipelines/evaluating.py:7](../pipelines/evaluating.py#L7) |
| `step: evaluate` | `evaluate()` | [steps/evaluating/evaluate.py:7](../steps/evaluating/evaluate.py#L7) |
| → resolve HF user, build env, `.fit()` | `run_evaluation_on_sagemaker()` | [llm_engineering/model/evaluation/sagemaker.py:20](../llm_engineering/model/evaluation/sagemaker.py#L20) |
| resolve `model_ids` (with fallback) | `check_if_huggingface_model_exists()` | [evaluate.py:214](../llm_engineering/model/evaluation/evaluate.py#L214) |
| resolve dataset (with fallback) | `check_if_huggingface_dataset_exists()` | [evaluate.py:229](../llm_engineering/model/evaluation/evaluate.py#L229) |
| SageMaker job entry point (3 phases) | `if __name__ == "__main__":` block | [evaluate.py:254](../llm_engineering/model/evaluation/evaluate.py#L254) |
| Phase 1: `generate_answers()` (vLLM) | `generate_answers()` | [evaluate.py:32](../llm_engineering/model/evaluation/evaluate.py#L32) |
| Phase 2: `evaluate_answers()` (thread pool) | `evaluate_answers()` | [evaluate.py:159](../llm_engineering/model/evaluation/evaluate.py#L159) |
| → one batch of 5, one OpenAI client | `evaluate_batch()` | [evaluate.py:154](../llm_engineering/model/evaluation/evaluate.py#L154) |
| → judge prompt + JSON response | `evaluate_answer()` | [evaluate.py:101](../llm_engineering/model/evaluation/evaluate.py#L101) |
| Phase 3: print mean accuracy / style | loop at end of `__main__` | [evaluate.py:266-279](../llm_engineering/model/evaluation/evaluate.py#L266-L279) |

## Relevant files

- [pyproject.toml](../pyproject.toml): `run-evaluation-pipeline` poe task definition
- [tools/run.py](../tools/run.py): CLI entry point
- [configs/evaluating.yaml](../configs/evaluating.yaml): evaluation pipeline parameters (`is_dummy` only)
- [pipelines/evaluating.py](../pipelines/evaluating.py): ZenML pipeline definition
- [steps/evaluating/evaluate.py](../steps/evaluating/evaluate.py)
- [llm_engineering/model/evaluation/sagemaker.py](../llm_engineering/model/evaluation/sagemaker.py): builds and launches the SageMaker job (as a Training job)
- [llm_engineering/model/evaluation/evaluate.py](../llm_engineering/model/evaluation/evaluate.py): script that runs on SageMaker (vLLM generation, LLM-as-judge scoring)
- [llm_engineering/model/evaluation/requirements.txt](../llm_engineering/model/evaluation/requirements.txt): packages installed in the SageMaker container (`vllm`, `openai`, `datasets`, …)
- [llm_engineering/settings.py](../llm_engineering/settings.py): `OPENAI_MODEL_ID` (default `gpt-4o-mini`), `OPENAI_BASE_URL`, `HUGGINGFACE_ACCESS_TOKEN`, `AWS_ARN_ROLE`
