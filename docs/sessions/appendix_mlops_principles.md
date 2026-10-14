# Appendix: MLOps Principles (Book Appendix)

## 🎯 Learning Objectives

By the end of this session, you will:
- Apply the six MLOps principles to the LLM Twin
- Know the manual → CT → CI/CD automation ladder
- Version code, data, and models independently
- Test data, model, and code across six test types
- Monitor logs, metrics, and the three drift types
- Make processes reproducible with seeds and tracked inputs
- Design a test for each of code, data, and model
- Recognize that tools implement principles; they are not the principles

> This session covers the **book Appendix: MLOps Principles** (pp. 490-503), the tool-independent foundation behind Chapters 8-11. It is the conceptual backbone for [`session_8.1_docker.md`](session_8.1_docker.md)-[`8.3`](session_8.3_zenml.md), [`session_9.2_performance.md`](session_9.2_performance.md), and [`session_11.1_ct_pipeline_alerting.md`](session_11.1_ct_pipeline_alerting.md).

---

## 🏗️ Architecture Overview

```
Six MLOps principles
1. Automation / operationalization   manual → CT → CI/CD
2. Versioning                        code | data | model
3. Experiment tracking               run, compare, promote
4. Testing                           unit → integration → system → acceptance → stress (+ regression)
5. Monitoring                        logs, metrics, drifts, alerts
6. Reproducibility                   tracked inputs + seeds

         ┌───────────────────────────────────────────────┐
         │  These are PRINCIPLES, not tools.               │
         │  ZenML, Comet ML, Git, DVC, pytest are          │
         │  implementations of the principles.             │
         └───────────────────────────────────────────────┘
```

The principles are ordered by dependency, not importance. Automation is impossible without versioning; versioning is meaningful only with tracking; all of it fails without reproducibility.

```
 Reproducibility ──► Versioning ──► Experiment tracking
       │                                   │
       └──────────► Testing ◄──────────────┘
                       │
                       ▼
                  Automation ──► Monitoring ──► Alerts ──► (retrain: back to top)
```

| Principle | Question it answers | Failure without it |
|-----------|--------------------|--------------------|
| Automation | Can a human trigger the whole system? | Manual, error-prone, slow iteration |
| Versioning | What produced this artifact? | Unreproducible results, silent drift |
| Experiment tracking | Which run was best, and why? | Decisions from memory; lost knowledge |
| Testing | Does it still work, on data and model too? | Regressions ship quietly |
| Monitoring | Is it degrading in production? | Users notice before you do |
| Reproducibility | Can I get the same result again? | No comparison, no trust |

---

## 1. Automation / Operationalization

Three tiers, adopted gradually:

- **Manual process**: notebooks, manual steps; output is the code to prepare data and train.
- **Continuous training (CT)**: automate data + training via an orchestrator (ZenML), triggered on a schedule or an event (new data, performance drop).
- **CI/CD**: automate building, testing, and deployment of code, models, and pipeline components across staging/production.

On top of the **FTI (feature/training/inference)** architecture, the manual step gives way to **CI/CD/CT**: CT automates the FTI pipelines; CI/CD builds, tests, and ships their code.

### The ladder, concretely

```
Tier 0  Manual
        notebook ──► run cells ──► copy output ──► deploy by hand
        ✓ fast to start   ✗ not repeatable

Tier 1  Continuous Training (CT)
        ZenML pipeline runs data → features → datasets → training
        trigger: manual | schedule | event
        ✓ repeatable pipeline   ✗ you still ship the code by hand

Tier 2  CI/CD/CT
        CI tests on PR ──► CD builds & ships image ──► CT runs the pipelines
        ✓ every dimension automated   ✗ most setup cost

Rule: adopt in order. You cannot CI/CD your way out of a non-reproducible notebook.
```

**Why gradual**: each tier requires the previous one to be trustworthy. Automating an unreproducible process just makes it fail faster.

**In the repo**: [`session_8.2_cicd.md`](session_8.2_cicd.md) (CI/CD with GitHub Actions), [`session_8.3_zenml.md`](session_8.3_zenml.md) (ZenML orchestration), [`session_11.1_ct_pipeline_alerting.md`](session_11.1_ct_pipeline_alerting.md) (CT triggers).

---

## 2. Versioning

The whole system changes if code, data, or model changes, so version all three independently.

- **Code**: Git commits + semantic releases (`vMAJOR.MINOR.PATCH`). Tools: GitHub, GitLab.
- **Model**: a **model registry** stores and versions models (semantic versioning + alpha/beta). Attach metadata (training data, architecture, performance, latency) via an ML metadata store to build a navigable catalog.
- **Data**: harder, since it depends on type and scale. Options: a SQL version column, Git-like tools (**DVC**), or artifact layers in Comet ML / W&B / ZenML. Store data on-prem or in object storage (AWS S3).

### Why "independently"

Versioning all three as one bundle is tempting (a single tag) but wrong. A model can be retrained on new data without a code change; code can change without any data change. Coupling the versions means you cannot tell which axis moved.

```
        code v1.3.0  ─┐
                      ├──► the deployed system
        data v2024-10 ─┤
        model v0.2.1  ─┘

Change ONE axis and you have a NEW system with a NEW combination.
Version each so the triple is discoverable, not guessed.
```

**Semantic versioning for models**:

| Bump | When | Example |
|------|------|---------|
| MAJOR | Incompatible behavior/interface | input schema change |
| MINOR | Improved quality, compatible | new base checkpoint |
| PATCH | Small fix, compatible | prompt typo fix |
| alpha/beta suffix | Pre-release, not production | `0.2.0-beta` |

**Data versioning decision table**:

| Data scale / type | Suggested approach |
|-------------------|--------------------|
| small, text | Git or DVC pointer |
| tabular, in DB | version column / snapshot |
| large, iterative | artifact store + content hash (ZenML) |
| object storage | S3 versioning + manifest |

**In the repo**: Git for code; Hugging Face Hub as the model registry (`{workspace}/TwinLlama-3.1-8B(-DPO)`); ZenML artifacts version datasets; `data/data_warehouse_raw_data/*.json` for warehouse backups ([`session_9.1_data_warehouse.md`](session_9.1_data_warehouse.md)).

**The lineage question to be able to answer**: given a deployed model id, which code commit, which dataset version, and which config produced it? If that is not answerable in minutes, versioning is incomplete.

---

## 3. Experiment Tracking

Training is iterative: run parallel experiments, compare on predefined metrics, promote the best. A tracking tool logs metrics and visualizations for comparison.

**Tools**: Comet ML, W&B, MLflow, Neptune.

**What to log (minimum viable)**:
- **Config**: hyperparameters, model id, dataset version, code commit.
- **Metrics**: loss curves, eval metrics, latency, memory.
- **Artifacts**: checkpoints, predictions, plots.
- **Environment**: package versions, seed, hardware.

**Why track instead of naming folders**: a folder named `run_final_v2_actually` is not searchable, comparable, or auditable. A tracking system makes runs queryable and plots comparable, which is what turns iteration into a decision.

**Promotion discipline**: define the metric and the threshold *before* the run. "Promote if eval accuracy improves and latency does not regress more than 10%" is a rule; "this one looked better" is not.

**In the repo**: `report_to="comet_ml"` in the SFT/DPO trainers, `COMET_PROJECT`, and the SageMaker `COMET_API_KEY` pass-through ([`session_7.1_comet_ml.md`](session_7.1_comet_ml.md)).

---

## 4. Testing

Test across **code, data, and model**, and validate integration with external services (feature store, etc.). **pytest** is the standard.

### Six test types

- **Unit**: a single component (a function, a transform).
- **Integration**: components working together (a pipeline + its store).
- **System**: the entire system end to end (performance, security, UX).
- **Acceptance (UAT)**: meets requirements, ready to deploy.
- **Regression**: changes do not reintroduce known bugs (applied across all levels, not a separate phase).
- **Stress**: behavior under extreme load/limited resources.

```
unit ──► integration ──► system ──► acceptance
  │           │             │            │
  └───────────┴─────────────┴────────────┘
              regression applies at every level (not a phase)
stress runs alongside system (load, limits)
```

### What to test

- **Inputs**: data types, format, length, edge cases (min/max, small/large).
- **Outputs**: types, formats, exceptions, intermediate and final outputs.

### The code / data / model triad

| Layer | Example test | Why it is not optional |
|-------|-------------|------------------------|
| Code | `clean_text` removes markup | cheap, fast, catches logic bugs |
| Data | features non-null, categories allowed, lengths in range | garbage in, garbage out |
| Model | loss decreases, overfit a tiny batch, shapes correct | catches training bugs before hours of compute |

### Test examples

- **Code**: assert `clean_text` cleans as expected; assert the chunker behaves across sentence sets and sizes.
- **Data**: validity checks when raw data is ingested or features computed — non-null, allowed categories, positive floats (tabular); length, encoding, language, special characters (text).
- **Model**: input/output tensor shapes; loss decreases after a batch; overfit a small batch (loss → 0); pipeline works on CPU and GPU; early-stopping and checkpoint logic.
- **Behavioral (CheckList)**: **invariance** (paraphrase must not change output), **directional** (known input changes must change output), **minimum functionality** (simplest inputs must pass).

### Worked example: overfit a tiny batch

The single most valuable model test is proving the training loop can drive loss to near zero on a handful of samples. If it cannot, the bug is in the loop, not the data.

```python
def test_can_overfit_one_batch():
    # tiny synthetic batch; run N steps; assert loss collapses
    losses = train_steps(samples=[...], steps=50)
    assert losses[-1] < 0.05 * losses[0]
```

This is a *model* test, not a code test: it validates that gradients flow and the optimizer is wired correctly.

**In the repo**: `tests/unit/unit_example_test.py` and `tests/integration/integration_example_test.py`, run by `poetry poe test` (with `ENV_FILE=.env.testing`) in the CI `test` job ([`session_8.2_cicd.md`](session_8.2_cicd.md)).

---

## 5. Monitoring

ML systems are probabilistic and degrade as production data drifts from training data. Monitoring detects degradation and triggers retraining.

### Logs
Capture configuration, queries/results, component start/end/crash, and tag each entry with its origin. Use automated log analysis to manage volume.

### Metrics
- **System**: latency, throughput, error rates, CPU/GPU/memory.
- **Model**: accuracy/precision/F1, plus business metrics (ROI, click rate).
- **Windowed metrics**: compute over sliding intervals (e.g. hourly), not cumulative, to detect issues early.
- **Proxy metrics**: when ground truth is delayed/unavailable, approximate performance with drift detection.

**Why windowed, not cumulative**: a cumulative average hides a regression because old good data dilutes it. Hourly windows surface a drop the hour it happens.

### Drifts

| Drift | Formulation | Meaning |
|-------|-------------|---------|
| **Data** | `P(X) ≠ P_ref(X)` | input distribution shifts |
| **Target** | `P(y) ≠ P_ref(y)` | output/label distribution shifts |
| **Concept** | `P(y|X) ≠ P_ref(y|X)` | the input→output relationship shifts |

Concept drift can appear gradually, suddenly, or periodically (e.g. moving a model to a new geographic market).

```
data drift    : the inputs change    (P(X))
target drift  : the labels change    (P(y))
concept drift : the rule changes      (P(y|X))  ← hardest to detect
```

**Detection**: compare a **reference window** (training data) against a **test window** (production) with hypothesis tests. **Kolmogorov-Smirnov** for a single continuous feature (univariate); **chi-squared** for categorical; for text embeddings, reduce dimensionality and use **Maximum Mean Discrepancy (MMD)**:

```python
from alibi_detect.cd import KSDrift, MMDDrift

cd = KSDrift(X_ref, p_val=.05, preprocess_fn=preprocess_fn, input_shape=(max_len,))
# or, for embeddings:
cd = MMDDrift(x_ref, backend="pytorch", p_val=.05)
cd.predict(x)
```

**Monitoring vs observability**: monitoring collects and visualizes data; observability is the system's ability to expose meaningful internal state to diagnose root causes. Monitoring tells you *that* accuracy dropped; observability lets you find *why*.

### Alerts
Alert on a static threshold (e.g. accuracy < 0.8) or on drift p-values. Tune thresholds to avoid alert fatigue. Channels: Slack, Discord, email, PagerDuty. Before retraining, check data validity, schema, and outliers; then trigger the training pipeline.

**In the repo**: Opik traces the model config, tokens, and latency ([`session_7.2_opik_monitoring.md`](session_7.2_opik_monitoring.md) / [`session_11.1_ct_pipeline_alerting.md`](session_11.1_ct_pipeline_alerting.md)); ZenML alerters handle notifications ([`session_11.1_ct_pipeline_alerting.md`](session_11.1_ct_pipeline_alerting.md)). Drift detection is **not implemented** and is a clear extension point.

---

## 6. Reproducibility

Every process should produce identical results given the same input. Two aspects:

1. **Know your inputs**: track exactly which dataset version and config produced each asset.
2. **Control randomness**: ML is non-deterministic (random weight init, random imputation, random splits). Always set a **seed**.

> "Always try to make your processes as deterministic as possible, and in case you have to introduce randomness, always provide a seed that you control."

### Sources of non-determinism

| Source | Where | Control |
|--------|-------|---------|
| Weight init | model construction | seed |
| Data shuffling | DataLoader / splits | seed, `random_state` |
| Dropout | training | seed (still approximate) |
| GPU kernels | cuDNN/cuBLAS | deterministic flags, slower |
| Data ordering | crawl/sort | stable sort, snapshot |
| Library versions | environment | lockfile (poetry.lock) |

**The realistic goal**: exact reproducibility on the same hardware with deterministic flags is possible but costly. The pragmatic target is **statistical reproducibility** — same inputs and seed yield results within a tight band — plus a fully recorded environment so any difference is explainable.

**In the repo**: `seed=0` in SFT/DPO `TrainingArguments`/`DPOConfig`; `random_state=42` in the dataset splits; ZenML artifacts capture dataset versions and configs.

---

## 🛠️ Hands-On: Audit the Project Against the Six Principles

### Step 1: Score each principle

| Principle | Where it lives | Status |
|-----------|----------------|--------|
| Automation | `end_to_end_data`, CI/CD, ZenML | partial (CT manual) |
| Versioning | Git + HF registry + ZenML artifacts | good |
| Experiment tracking | Comet ML | good |
| Testing | `tests/`, CI pytest | minimal (dummy tests) |
| Monitoring | Opik traces + ZenML alerts | partial (no drift) |
| Reproducibility | seeds + artifacts | good |

### Step 2: Pick the weakest and fix it

Add a real unit test for `clean_text`:

```python
def test_clean_text_removes_symbols():
    from llm_engineering.application.preprocessing.operations.cleaning import clean_text
    assert clean_text("# Hello *world*!") == "Hello world!"
    assert clean_text("a    b") == "a b"
```

Run `poetry poe test` and confirm it passes in CI.

### Step 3: Add a drift check for embeddings

Implement an `MMDDrift` check comparing the reference and current embedded-chunk distributions (see the snippet above).

### Step 4: Close the lineage loop

For one deployed model, write down: code commit, dataset version, config, seed, image tag. If any is missing, that principle is not yet satisfied.

---

## 📝 Exercise 1: Write a Reproducibility Checklist

### Task

For the SFT pipeline, write a checklist that guarantees a run is reproducible:

1. Which dataset version and config were used?
2. Which seed and hyperparameters?
3. Which code commit and container image (SHA tag)?
4. Where are the resulting artifacts/versions recorded?

Then verify each item is actually captured by ZenML + Comet + Git.

**Goal**: Turn "reproducible" from a promise into a verifiable checklist.

---

## 📝 Exercise 2: Design a Three-Layer Test Suite

### Task

Add one test at each layer to the existing `tests/` tree, then run `poetry poe test`.

1. **Code**: unit-test `clean_text` on markup, whitespace, and an empty string.
2. **Data**: validate a toy `ArticleDocument` — `content` non-empty, `author_full_name` present, `platform` in an allow-list.
3. **Model**: assert that a tiny forward pass returns the expected tensor shape, or that a 50-step run on one batch reduces loss by 90%.

```python
# tests/unit/test_cleaning.py  (code layer)
from llm_engineering.application.preprocessing.operations.cleaning import clean_text


def test_clean_text_edge_cases():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("# Hello *world*!") == "Hello world!"


# tests/unit/test_data_validity.py  (data layer)
def test_article_document_valid(article_factory):
    doc = article_factory(content="")
    assert doc.content == ""  # empty is allowed by the model, rejected downstream
```

**Goal**: Experience that testing ML means testing three things, and that data/model tests look different from conventional unit tests.

---

## 🔗 How the Six Principles Interact

The principles are not independent checkboxes; they form a dependency graph. A gap in one weakens the others.

```
   Reproducibility ──────────────────────────────┐
        │                                         │
        │ (you must know inputs before you can    │
        │  version them)                          │
        ▼                                          ▼
   Versioning ──► Experiment tracking ──► Testing ──► Automation
        │                       │                        │
        │                       │                        ▼
        │                       │                    Monitoring
        │                       │                        │
        └───────────────────────┴────────► Alerts ──► Retraining (CT)
```

Worked interactions:

- **Reproducibility × Versioning**: you cannot version a dataset you cannot identify. Recording inputs is a prerequisite for meaningful data versions.
- **Versioning × Tracking**: a tracked run with an unpinned dataset version is not reproducible. Tracking tools log the version; versioning supplies it.
- **Tracking × Testing**: experiment tracking tells you *which* run; testing tells you whether the run is *valid*. A great metric from a buggy pipeline is a false positive.
- **Testing × Automation**: you should not automate a process you cannot test. CI runs the tests; CT runs the pipelines. Tests are the gate for both.
- **Monitoring × CT**: monitoring is the trigger source for continuous training. No drift detection means CT runs on a schedule or by hand, not in response to reality.

**Corollary**: fix the *lowest* broken principle first. Adding alerting (top) on top of unseeded training (bottom) produces alerts about results you cannot reproduce.

---

## 📊 Mapping to the ML Test Score

[Breck et al., *The ML Test Score*](https://research.google/pubs/pub46555/) operationalizes these principles as a scored rubric. Use it to audit systematically.

| Category | Representative checks | Maps to principle |
|----------|----------------------|-------------------|
| Data | feature expectations documented; schema validated | Testing, Versioning |
| Model | trained model specs; overfit small batch; no-leak checks | Testing |
| Infrastructure | reproducible training; pipeline integration tests | Reproducibility, Automation |
| Monitoring | dependency changes; training/serving skew; prediction bias | Monitoring |
| (Process) | run metadata captured; rollback path | Tracking, Versioning |

A pragmatic scorecard for this repo:

| Area | Current evidence | Grade |
|------|------------------|-------|
| Data validation | none in `tests/` | weak |
| Model sanity | none committed | weak |
| Reproducible training | `seed=0`, random_state=42 | good |
| Pipeline tests | dummy example tests only | weak |
| Production monitoring | Opik metrics + ZenML alerts | partial |
| Drift detection | absent | missing |
| Run metadata | Comet + ZenML | good |

**How to use it**: a score, not a verdict. The value is in naming the weak cells and fixing them in dependency order.

---

## 📖 Case Study: One Feature Through All Six Principles

Take a single concrete change — *adding a `language` field to cleaned documents* — and follow it through each principle.

1. **Automation**: the cleaning step (`CleaningDispatcher.dispatch`) computes the field; the feature pipeline carries it forward without manual editing.
2. **Versioning**: the change is a code commit; the resulting cleaned-document artifact gets a new ZenML version; the derived datasets get new versions.
3. **Experiment tracking**: a training run on the new features logs a new Comet experiment with the schema version in its config.
4. **Testing**: a data test asserts `language` is a non-null value from the allow-list; a code test checks the detector on edge inputs (empty string, mixed script).
5. **Monitoring**: production traces gain a `language` dimension; a windowed metric tracks its distribution, and drift in language mix is a signal.
6. **Reproducibility**: the run records the dataset version, seed, and code commit, so the language-aware model can be rebuilt exactly.

```
code commit ──► CT pipeline ──► new artifact version ──► tracked run
     │                                   │                    │
   tests                            data test             metrics
     │                                   │                    ▼
     └──────────────────────────► reproducible ◄──── monitored in prod
```

**The lesson**: a one-field change touches all six. That is the point of the principles — they are a lens that ensures no dimension is forgotten.

---

## 🧪 A Minimal Test Suite Skeleton

A starting layout that covers code, data, and model without pretending `tests/` is complete.

```
tests/
├── unit/
│   ├── test_cleaning.py          # code: clean_text edge cases
│   ├── test_chunking.py          # code: chunk counts/sizes
│   └── test_data_validity.py     # data: schema, nulls, categories
├── integration/
│   └── test_feature_pipeline.py  # pipeline + store together
└── model/
    └── test_training_loop.py     # model: overfit-one-batch
```

```python
# tests/model/test_training_loop.py  (model layer)
def test_overfit_one_batch():
    losses = train_steps(samples=make_batch(n=4), steps=50)
    assert losses[-1] < 0.1 * losses[0]
```

**How to grow it**: add a test every time a bug escapes to review. A bug that reached production becomes a regression test at the lowest layer that can catch it.

---

## 📁 Tool Landscape by Principle

Tools are interchangeable; the principle is not. This table prevents tool-driven thinking.

| Principle | Tools seen | Repo's choice | Alternative |
|-----------|-----------|---------------|-------------|
| Automation | orchestrators, CI runners | ZenML, GitHub Actions | Airflow, Prefect, GitLab CI |
| Versioning | git, DVC, registries | Git + HF Hub + ZenML | DVC, MLflow Registry, W&B |
| Experiment tracking | Comet, W&B, MLflow, Neptune | Comet ML | MLflow, W&B |
| Testing | pytest, Great Expectations | pytest | unittest, dbt tests, GE |
| Monitoring | Opik, Evidently, alibi-detect | Opik (+ alerts) | Evidently, WhyLabs, Arize |
| Reproducibility | seeds, lockfiles, containers | poetry.lock + seeds | conda-lock, Docker digests |

**Rule when choosing**: pick by the constraint (managed vs self-hosted, cost, team familiarity), then verify it implements the principle. Never adopt a tool because a peer used it, then discover it does not version data.

---

## 🐛 Common Pitfalls

- **Versioning code only**: data and model drift silently; version all three.
- **Cumulative metrics**: hide regressions; use windowed metrics.
- **Alert fatigue**: too many false positives make alerts untrustworthy.
- **Unseeded randomness**: results cannot be reproduced or compared.
- **No drift detection**: quality degrades before anyone notices. This is the project's biggest monitoring gap.
- **Testing only code**: data and model need tests too.
- **Treating the tool as the principle**: adopting ZenML does not make you "do MLOps" if inputs are still untracked.
- **Promoting by memory**: without a predefined metric and threshold, "best run" is a guess.
- **Bundling versions**: a single tag for code+data+model hides which axis changed.
- **Exact-reproducibility chasing**: spending days on bit-exact GPU determinism while the data version is unrecorded is misallocated effort.

---

## 🎓 Knowledge Check

1. **What are the three automation tiers?**
   - Answer: manual → continuous training (CT) → CI/CD.

2. **What are the three things to version independently?**
   - Answer: code, data, and model.

3. **Name the three drift types and their formulations.**
   - Answer: data (`P(X)`), target (`P(y)`), and concept (`P(y|X)`).

4. **Which test types apply across all phases rather than as a separate phase?**
   - Answer: regression tests.

5. **What is the difference between monitoring and observability?**
   - Answer: monitoring collects/visualizes data; observability exposes internal state to diagnose root causes.

6. **What is the single most important reproducibility practice?**
   - Answer: control randomness with an explicit seed, and track all inputs.

7. **Why is versioning data harder than versioning code?**
   - Answer: Data is large, changes continuously, and has no native diff; it needs snapshots, hashes, or artifact stores rather than commits.

8. **When should you use a schedule trigger versus an event trigger?**
   - Answer: Schedule for slowly-changing data with predictable cost; event for fast-moving data where staleness is costly.

9. **What is the exact difference between data drift and concept drift?**
   - Answer: Data drift is a change in inputs `P(X)`; concept drift is a change in the input→output relationship `P(y|X)` with inputs possibly unchanged.

10. **Which statistical test for a continuous feature, and which for embeddings?**
    - Answer: Kolmogorov-Smirnov for a continuous feature; Maximum Mean Discrepancy (MMD) for high-dimensional embeddings.

11. **Why use windowed rather than cumulative metrics?**
    - Answer: Cumulative averages dilute recent regressions; windows surface a drop when it happens.

12. **What are the three CheckList behavioral test categories?**
    - Answer: invariance, directional, and minimum functionality.

13. **Why is "overfit a tiny batch" a model test?**
    - Answer: It proves the training loop can reduce loss, isolating loop bugs from data quality problems.

14. **What must be recorded for a run to be reproducible?**
    - Answer: dataset version, config, seed, code commit, container image tag, and where artifacts/versions are stored.

15. **Why is a tool not the principle?**
    - Answer: ZenML, Comet, Git, and pytest are implementations; the principles (automation, versioning, tracking, testing, monitoring, reproducibility) hold regardless of tooling.

---

## 📖 Glossary

- **FTI**: Feature, Training, Inference — the three-pipeline architecture.
- **CT**: Continuous Training; automated retraining by trigger.
- **CI/CD**: Continuous Integration / Continuous Delivery (or Deployment).
- **Model registry**: A versioned store of models with metadata.
- **Metadata store**: A catalog of runs, artifacts, and their relationships.
- **Experiment tracking**: Logging and comparing training runs.
- **Semantic versioning**: `MAJOR.MINOR.PATCH` release scheme.
- **Data drift**: Change in the input distribution `P(X)`.
- **Target drift**: Change in the label distribution `P(y)`.
- **Concept drift**: Change in the input→output relationship `P(y|X)`.
- **MMD**: Maximum Mean Discrepancy, a kernel two-sample test for embeddings.
- **KS test**: Kolmogorov-Smirnov two-sample test for a continuous feature.
- **Windowed metric**: A metric computed over a sliding interval.
- **Proxy metric**: An approximation used when ground truth is delayed.
- **CheckList**: A behavioral-testing methodology (invariance, directional, minimum functionality).
- **Seed**: A value that makes a random process reproducible.
- **Lineage**: The recorded chain from inputs to a produced artifact.
- **Observability**: The ability to infer internal state from external signals.

---

## 🔗 Course Complete

This appendix closes the book. Pair it with:

- [`session_8.2_cicd.md`](session_8.2_cicd.md) (CI/CD), [`session_8.3_zenml.md`](session_8.3_zenml.md) (ZenML), [`session_11.1_ct_pipeline_alerting.md`](session_11.1_ct_pipeline_alerting.md) (CT + alerting) for principle 1.
- [`session_9.1_data_warehouse.md`](session_9.1_data_warehouse.md) (warehouse backup) for principle 2.
- [`session_7.1_comet_ml.md`](session_7.1_comet_ml.md) (Comet ML) for principle 3.
- [`session_9.2_performance.md`](session_9.2_performance.md) (performance) for principles 4-6 in practice.

---

## 📚 Additional Resources

- [Testing Machine Learning Systems (Made With ML)](https://madewithml.com/courses/mlops/testing/)
- [Monitoring Machine Learning Systems (Made With ML)](https://madewithml.com/courses/mlops/monitoring/)
- [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118)
- [The ML Test Score (Breck et al.)](https://research.google/pubs/pub46555/)
- [Semantic Versioning](https://semver.org/)
- [Alibi Detect: drift detection](https://docs.seldon.io/projects/alibi-detect/en/stable/)
- [DVC: data version control](https://dvc.org/)

---

## 🔎 References

- **Book**: *LLM Engineer's Handbook* — Appendix: MLOps Principles (pp. 490-503).
- **Repo**: `tests/`, `pyproject.toml` (`poe test`), `llm_engineering/application/preprocessing/operations/cleaning.py`, SFT/DPO `TrainingArguments`/`DPOConfig` seeds, ZenML pipeline artifacts.
- **Related sessions**: [`session_7.1_comet_ml.md`](session_7.1_comet_ml.md), [`session_7.2_opik_monitoring.md`](session_7.2_opik_monitoring.md), [`session_8.2_cicd.md`](session_8.2_cicd.md), [`session_8.3_zenml.md`](session_8.3_zenml.md), [`session_9.1_data_warehouse.md`](session_9.1_data_warehouse.md), [`session_9.2_performance.md`](session_9.2_performance.md), [`session_11.1_ct_pipeline_alerting.md`](session_11.1_ct_pipeline_alerting.md).
- **Curriculum**: [`../CURRICULUM.md`](../CURRICULUM.md) and [`../BOOK-MAP.md`](../BOOK-MAP.md).

---

**Estimated Time**: 4-5 hours

**Prerequisites**: [Session 8.1](session_8.1_docker.md), [Session 8.2](session_8.2_cicd.md), [Session 8.3](session_8.3_zenml.md), [Session 11.1](session_11.1_ct_pipeline_alerting.md)

**Outcome**: You can audit any ML system against the six MLOps principles and close its weakest gaps.
