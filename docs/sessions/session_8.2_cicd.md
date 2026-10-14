# Session 8.2: CI/CD with GitHub Actions

## 🎯 Learning Objectives

By the end of this session, you will:
- Read the CI and CD GitHub Actions workflows line by line
- Understand the quality gates: secret scan, lint, format, tests
- Map each workflow step to its Poe task and explain why the shared vocabulary matters
- Configure the required repository secrets safely
- Know what CD builds, tags, and where it pushes
- Reason about branch protection, concurrency, and immutable artifacts
- Extend the pipeline with a new gate (type checking) in both CI and local dev
- Diagnose the common failure modes of leaky secrets, format drift, and fork PRs

---

## 🏗️ Architecture Overview

```
Pull request                          Push to main
     │                                     │
     ▼                                     ▼
┌──────────────────────┐         ┌──────────────────────────┐
│ CI  (ci.yaml)         │         │ CD  (cd.yaml)            │
│                       │         │                          │
│ Job: QA               │         │ Job: Build & Push        │
│  ├─ gitleaks-check    │         │  ├─ checkout             │
│  ├─ ruff check        │         │  ├─ setup buildx         │
│  └─ ruff format       │         │  ├─ configure AWS creds  │
│                       │         │  ├─ login to ECR         │
│ Job: Test             │         │  └─ build amd64 image    │
│  └─ pytest            │         │       push :sha + :latest│
└──────────────────────┘         └──────────────────────────┘
```

The pattern: **CI gates merges; CD publishes immutable images** tagged by commit SHA plus a moving `latest`.

### Why split CI and CD?

| Concern | CI | CD |
|---------|----|----|
| Trigger | `pull_request` | `push` to `main` |
| Question answered | "Is this change safe to merge?" | "What artifact represents the merged code?" |
| Output | pass/fail status checks | a versioned container image |
| Secrets needed | none (or read-only) | AWS ECR credentials |
| Failure cost | developer fixes and pushes again | a bad artifact may reach a registry; pin by SHA |

Separating them means a contributor without AWS credentials can still get full validation, and no PR can accidentally publish an image. This is the core security property: **the only path to a published artifact is a merge to `main`.**

### The environment ladder

```
 local                     CI                         CD                    Deploy
 ┌─────────┐          ┌──────────────┐         ┌──────────────┐      ┌──────────┐
 │ ruff    │          │ gitleaks     │         │ build amd64  │      │ SageMaker│
 │ pytest  │  ──PR──► │ ruff (check) │ ─merge─►│ push to ECR  │─────►│ pulls by │
 │ poe *   │          │ pytest       │         │ :sha+latest  │      │ :sha tag │
 └─────────┘          └──────────────┘         └──────────────┘      └──────────┘
   same commands         same commands            docker build        ZenML parent_image
```

Every rung runs the **same Poe tasks**, so local success predicts CI success.

---

## 📁 Key Files Explained

### 1. `.github/workflows/ci.yaml` - Continuous Integration

The real file is 69 lines:

```yaml
name: CI

on:
  pull_request:
    
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  qa:
    name: QA
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.11"

      - name: Install poetry
        uses: abatilo/actions-poetry@v2
        with:
          poetry-version: 1.8.3

      - name: Install packages
        run: |
          poetry install --only dev
          poetry self add 'poethepoet[poetry_plugin]'

      - name: gitleaks check
        run: poetry poe gitleaks-check

      - name: Lint check [Python]
        run: poetry poe lint-check

      - name: Format check [Python]
        run: poetry poe format-check

  test:
    name: Test
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.11"

      - name: Install poetry
        uses: abatilo/actions-poetry@v2
        with:
          poetry-version: 1.8.3

      - name: Install packages
        run: |
          poetry install
          poetry self add 'poethepoet[poetry_plugin]'

      - name: Run tests
        run: |
          echo "Running tests..."
          poetry poe test
```

**Key Concepts**:

- **Two independent jobs**: `qa` and `test` run **in parallel**. Both must pass for the PR to be mergeable (once branch protection requires them). Running them in parallel cuts wall-clock time roughly in half.
- **`qa` installs only dev dependencies** (`poetry install --only dev`) - lint/format/secret-scan need no runtime packages, so the job starts faster. This is the `[tool.poetry.group.dev.dependencies]` group (`ruff`, `pre-commit`, `pytest`).
- **`test` installs everything** because the tests import the application, which imports `torch`, `zenml`, `qdrant-client`, and the rest.
- **`concurrency` with `cancel-in-progress: true`** supersedes in-flight runs when new commits are pushed. The `group` keys on workflow + ref, so a new push to the same PR cancels the previous run. This saves CI minutes and gives faster feedback on the latest commit.
- **`poetry poe <task>`** is the single source of truth: CI runs the same commands you run locally. There is no "CI-only" test script to drift out of sync.
- **Ordering inside `qa`**: gitleaks runs first. Secrets before code - a leaked credential is the most expensive possible failure, so it is checked before spending time on lint.

### The Poe task definitions behind each step

```toml
# pyproject.toml (lines 134-139, 157-159)
lint-check = "poetry run ruff check ."
format-check = "poetry run ruff format --check ."
gitleaks-check = "docker run -v .:/src zricethezav/gitleaks:latest dir /src/llm_engineering"
lint-fix = "poetry run ruff check --fix ."
format-fix = "poetry run ruff format ."

[tool.poe.tasks.test]
cmd = "poetry run pytest tests/"
env = { ENV_FILE = ".env.testing" }
```

**Key Concepts**:

- **`lint-check`** runs `ruff check` (flake8/import-sort/bugbear rules; see the config below).
- **`format-check`** runs `ruff format --check`, which fails if any file is not formatted. **Never let CI format code; it only verifies.** Auto-formatting in CI hides drift and creates commits you did not author.
- **`gitleaks-check`** mounts the repo into a Gitleaks container and scans `llm_engineering/` for committed secrets. This is the security gate. Note the path: it scans the application package, not the whole repo.
- **`test`** runs `pytest tests/` with `ENV_FILE=.env.testing`, so tests use a dedicated env file rather than production `.env`. Never let the test job read real credentials.
- **`lint-fix` / `format-fix`** are the local counterparts. CI failing on format is a signal to run these, not to change the check.

### The linter configuration

```toml
# ruff.toml (full file, 23 lines)
line-length = 120
target-version = "py311"
extend-exclude = [
    ".github",
    "graphql_client",
    "graphql_schemas"
]

[lint]
extend-select = [
  "I",
  "B",
  "G",
  "T20",
  "PTH",
  "RUF"
]

[lint.isort]
case-sensitive = true

[lint.pydocstyle]
convention = "google"
```

**Rule-by-rule, and why each is here**:

| Code | Selection | What it enforces | Why it matters for this project |
|------|-----------|------------------|---------------------------------|
| `line-length=120` | — | Max 120 columns | Wide ML signatures and config dicts stay readable |
| `target-version="py311"` | — | Python 3.11 syntax | `X | Y` unions, `list[str]`, `match` are allowed |
| `I` | isort | Import ordering | Deterministic diffs; merged imports do not churn |
| `B` | flake8-bugbear | Likely bugs (mutable defaults, etc.) | Catches real defects, not style |
| `G` | flake8-logging-format | Lazy `%` logging format | Avoids eager f-string cost in `logger.info` |
| `T20` | flake8-print | Bans `print()` | Forces structured logging via `loguru` |
| `PTH` | flake8-use-pathlib | Prefer `pathlib` over `os.path` | Consistent path handling across OSes |
| `RUF` | Ruff-native rules | Misc correctness rules | Includes the `# noqa` audit |

- **`extend-exclude`** skips `.github` (YAML, not Python) and the `graphql_*` generated trees. Generated code should not be hand-fixed to satisfy a linter.
- **`T20` is why the scripts use `print(...)  # noqa`** in `finetune.py` and `evaluate.py`: those prints are intentional console output and explicitly exempted. If you add a new script meant to print, annotate it the same way rather than weakening the rule.

### The pre-commit hook

```yaml
# .pre-commit-config.yaml (full file, 10 lines)
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.5
    hooks:
      - id: ruff # Run the linter.
      - id: ruff-format # Run the formatter.
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.2
    hooks:
      - id: gitleaks
```

The hook localizes the same three checks (lint, format, secret scan) before code ever leaves the machine. This is the first line of defense; CI is the second. Note the hook's ruff revision (`v0.3.5`) can lag the pinned dev dependency (`ruff = "^0.4.9"`); pin them together when you upgrade to avoid "passes locally, fails in CI" surprises.

---

### 2. `.github/workflows/cd.yaml` - Continuous Delivery

The real file is 43 lines:

```yaml
name: CD

on:
  push:
    branches:
      - main

concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: true

jobs:
  build:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build images & push to ECR
        id: build-image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          tags: |
            ${{ steps.login-ecr.outputs.registry }}/${{ secrets.AWS_ECR_NAME }}:${{ github.sha }}
            ${{ steps.login-ecr.outputs.registry }}/${{ secrets.AWS_ECR_NAME }}:latest
          push: true
```

**Key Concepts**:

- **Triggered on push to `main`** only, so images are published after merge, never from a PR. Combined with the CI gate, this means only reviewed, tested, lint-clean code becomes an image.
- **`configure-aws-credentials`** reads the repository secrets to obtain temporary AWS credentials for the job. Prefer OIDC (`role-to-assume`) in new repos over long-lived access keys; this project uses static keys for simplicity.
- **`amazon-ecr-login`** authenticates Docker to the account's registry and exposes the registry URL as an output (`steps.login-ecr.outputs.registry`). This is why the tag is built from an output rather than a hard-coded URL.
- **Two tags**: `:${{ github.sha }}` (immutable, traceable to a commit) and `:latest` (convenience). **Pin deployments to the SHA tag for reproducibility**; `latest` is only a human convenience.
- **`push: true`** builds and pushes in one step via Buildx. `context: .` and `file: ./Dockerfile` point at the repo's Dockerfile (Session 8.1).
- **Missing `platforms:`**: the workflow does not pass `--platform linux/amd64`. The build therefore uses the runner's native architecture (amd64 on `ubuntu-latest`), which happens to match SageMaker. On a self-hosted ARM runner you would need `platforms: linux/amd64`. This is a latent portability constraint worth making explicit.

### Required repository secrets

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | ECR auth |
| `AWS_SECRET_ACCESS_KEY` | ECR auth |
| `AWS_REGION` | ECR region (`eu-central-1` in the book) |
| `AWS_ECR_NAME` | Target repository name (created by ZenML Cloud) |

Set these under **Settings → Secrets and variables → Actions**. Never echo them in logs. GitHub masks secret values in logs, but that masking is defeated by transforming the value (base64, splitting) before printing.

### The image tag lifecycle

```
commit abc123 ──merge──► CI passed ──► CD build
                                        ├── repo:abc123   (immutable, pin here)
                                        └── repo:latest   (moves forward)
SageMaker parent_image ──► use :abc123 for reproducible runs
```

The ZenML `parent_image` in every `configs/*.yaml` points at `.../zenml-rlwlcs:latest`. That is the moving tag. For production reproducibility, replace it with a SHA tag and update it deliberately.

---

## 🔬 Deep Dive: The Gate Philosophy

```
       local                      CI                         CD
    ┌─────────┐            ┌──────────────┐           ┌──────────────┐
    │ gitleaks│            │ gitleaks     │           │ checkout     │
    │ ruff    │            │ ruff (check) │           │ buildx       │
    │ format  │  ──PR──►   │ format check │  ──merge──►│ aws creds    │
    │ pytest  │            │ pytest       │           │ ecr login    │
    │ poe *   │            │ (parallel)   │           │ push :sha    │
    └─────────┘            └──────────────┘           └──────────────┘
```

Four principles that explain every choice in these files:

1. **Fast feedback first.** `qa` (seconds) and `test` (slower) run in parallel, but within `qa` the cheapest, highest-severity check (gitleaks) runs first. A developer learns about a leaked key before waiting on the suite.
2. **Secrets before code.** A committed credential cannot be un-leaked by a later commit; rotating it and purging history is expensive. Catching it in the first CI step is the cheapest possible intervention.
3. **Format is enforced, not applied.** `--check` fails; the developer runs `poetry poe format-fix` locally. CI never pushes changes back to the branch.
4. **Immutable artifacts.** The CD image is addressed by commit SHA, so any deploy can be traced back to exact code. `latest` exists only for convenience.

### What is NOT in this pipeline (and why)

- **No build cache in CD.** `docker/build-push-action` supports `cache-from`/`cache-to`; this workflow omits them, so every CD build installs Chrome and dependencies from scratch. Adding `cache-from: type=gha` would cut build time significantly.
- **No vulnerability scan.** Gitleaks is a secret scanner, not an image/CVE scanner (e.g. Trivy, Grype). A production pipeline should add one.
- **No deployment step.** CD publishes an image; it does not deploy. Deployment is separate (ZenML runs pull the `parent_image` by tag).
- **No SHA pinning of actions.** `@v3`-style floating major tags can be moved by the action maintainer; production hardening pins to a commit SHA.

---

## 🛠️ Hands-On: Reproduce CI Locally

### Step 1: Install dev tooling

```bash
poetry install --only dev
poetry self add 'poethepoet[poetry_plugin]'
```

### Step 2: Run each gate in the CI order

```bash
poetry poe gitleaks-check
poetry poe lint-check
poetry poe format-check
poetry poe test
```

### Step 3: Fix what fails

```bash
poetry poe lint-fix
poetry poe format-fix
```

Then re-run the checks. A gate is only "fixed" when the `-check` variant passes.

### Step 4: Simulate CD with docker

```bash
docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile .
```

This is the same build the CD job performs (modulo the ECR push).

### Step 5: Inspect the workflow locally

If you have `act` installed, run the CI workflow in Docker without pushing:

```bash
act pull_request -W .github/workflows/ci.yaml
```

Caveats: `act` approximates GitHub runners and may not have all secrets or services. Treat it as a fast smoke test, not a substitute for CI.

---

## 📝 Exercise: Strengthen the Secret Gate

### Task

Gitleaks scans `llm_engineering/` only. Make the secret gate cover the whole repository.

1. Change the `gitleaks-check` Poe task to scan `.` (the whole mount) instead of `/src/llm_engineering`.
2. Deliberately plant a fake secret (a clearly fake AWS key) in a scratch file and confirm gitleaks fails locally.
3. Add a `.gitleaks.toml` allowlist for known false positives (for example, the example credentials in `docker-compose.yml`).
4. Confirm the allowlist suppresses only the intended entries and that the fake secret still fails.

**Goal**: Understand that a scanner's value is defined by its **scope** and its **allowlist**, not just its presence. A scanner that misses half the repo or one that is silenced into uselessness both fail the same way.

---

## 📝 Exercise 2: Add a Type-Check Job

### Task

Add a `pyright` (or `mypy`) type check to CI.

1. Add `pyright` to `[tool.poetry.group.dev.dependencies]`.
2. Add a Poe task `type-check = "poetry run pyright llm_engineering"`.
3. Add a step to the `qa` job: `- name: Type check [Python]` → `run: poetry poe type-check`.
4. Run it locally on the current code and note any existing type errors.

**Goal**: Experience adding a new gate to both local and CI in sync. Expect errors - the codebase is not fully typed, and that is exactly the point of the exercise. Decide whether to fix them, add an ignore list, or drop the gate.

---

## 🔬 Deep Dive: Supply-Chain Security and Build Performance

The pipeline as written is correct and simple. Two axes make it production-grade: **what third-party code runs**, and **how fast the image builds**.

### Supply chain

```
Risk                         Where it enters                Hardening
────────────────────────     ────────────────────────────   ─────────────────────────
Floating action tags         uses: ...@v3                   pin by commit SHA
Long-lived AWS keys          repo secrets                   use OIDC role-to-assume
Unpinned base image          python:3.11-slim-bullseye      pin by digest
Untrusted PR code            pull_request from forks        require approval; no secrets
Compromised action           any third-party action         allowlist actions; Dependabot
Image CVEs                   OS + pip packages              add Trivy/Grype scan
```

- **Pin actions to a full commit SHA** in hardened repos. A tag like `@v3` is mutable; even major versions can be moved. SHA pinning plus Dependabot updates gives both safety and freshness.
- **Prefer OIDC over static keys.** `aws-actions/configure-aws-credentials` supports `role-to-assume` with `id-token: write`; the job gets short-lived credentials and no long-lived secret exists to leak.
- **Least privilege for the ECR role.** The build role needs `ecr:GetAuthorizationToken` plus push/pull on the one repository, not `ecr:*` on `*`.
- **Fork PRs are untrusted.** They cannot read secrets, which is the correct default. Never restructure CI so that fork code runs with secrets.
- **Add an image scanner.** Gitleaks covers committed secrets; it says nothing about a vulnerable `pip` package. A Trivy step after build closes that gap.

### Build performance

```
                             No cache           With GHA cache
────────────────────────     ───────────────    ───────────────
Install Chrome + apt         every run          cached layer
poetry install (all deps)    every run          cached layer
COPY . (code)                every run          every run (cheap)
Wall-clock (order of mag.)   minutes            tens of seconds
```

`docker/build-push-action` accepts `cache-from` and `cache-to`. Adding:

```yaml
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

reuses layers between CD runs and can cut build time substantially. The tradeoff is cache storage and occasional stale-cache confusion; `mode=max` caches all layers, which is usually what you want for a dependency-heavy image.

### The cost of the parallel jobs

`qa` and `test` run in parallel, so wall-clock is roughly `max(qa, test)` plus setup, not the sum. Setup (checkout + Python + Poetry + install) dominates the fast `qa` job. If cost becomes a concern, a matrix job or a single job with sequential steps is cheaper in runner-minutes but slower; the current design deliberately favors feedback latency over minute cost.

---

## 🐛 Common Pitfalls

- **`gitleaks-check` mounts the whole repo but scans `llm_engineering/`**: a secret in `tools/`, `configs/`, or `tests/` is not caught. Widen the scan path deliberately.
- **Rotating vs deleting a leaked secret**: deleting the file or committing over it is **not enough**; the value remains in Git history. Rotate the secret immediately, then purge history if the repo is shared.
- **Format drift**: developers who do not run `format-fix` fail CI. Pre-commit hooks (`.pre-commit-config.yaml`) reduce this but must be installed (`pre-commit install`).
- **Hook lags action**: the pre-commit ruff revision (`v0.3.5`) can be older than the dev dependency (`^0.4.9`), so local passes can still fail CI. Pin them together.
- **Deprecated action versions**: `actions/checkout@v3`, `setup-python@v3`, and `configure-aws-credentials@v1` are older; new repos should use `@v4`/`@v5` and OIDC. Upgrade when convenient.
- **`latest` tag races**: two rapid merges can move `latest` twice. Deploy by SHA tag.
- **Secrets in forks**: PRs from forks do not receive secrets, so `configure-aws-credentials` steps fail there. Keep AWS steps only in CD.
- **No BuildKit cache**: CD rebuilds every dependency each run; adds minutes. Add `cache-from`/`cache-to: type=gha`.
- **`poetry install --only dev` is a trap for new deps**: if a dev tool starts importing a runtime package, this job breaks while the real code is fine.
- **`ENV_FILE=.env.testing` must exist**: the `test` task reads it; a missing file makes tests fail for the wrong reason.
- **Workflow YAML indent bug**: the `on:` block in `ci.yaml` has a trailing blank line under `pull_request:`. Harmless, but a reminder that workflow YAML is parsed strictly and a stray tab breaks the whole file.

---

## 🎓 Knowledge Check

1. **What triggers CI vs CD?**
   - Answer: CI on pull requests; CD on pushes to `main`.

2. **Which four gates run in the pipeline?**
   - Answer: gitleaks (secret scan), ruff lint, ruff format check, pytest.

3. **Why does the `qa` job install only dev dependencies?**
   - Answer: Lint/format/secret-scan do not need runtime packages, so it is faster.

4. **What does `format-check` do versus `format-fix`?**
   - Answer: `--check` fails CI if unformatted; `format-fix` rewrites files locally.

5. **What two tags does CD publish?**
   - Answer: the commit SHA and `latest`.

6. **Why is the image built for `linux/amd64`?**
   - Answer: SageMaker and most cloud runners are amd64; ARM builds would not run there.

7. **What does `concurrency.cancel-in-progress: true` do?**
   - Answer: Cancels the previous in-flight run of the same workflow + ref when a newer commit arrives, saving minutes and giving fresher feedback.

8. **Why does gitleaks run before lint in the `qa` job?**
   - Answer: A leaked secret is the highest-severity, highest-cost failure, so it is caught first.

9. **Why should deployments pin the SHA tag instead of `latest`?**
   - Answer: `latest` moves with every merge; the SHA tag is immutable and traceable to exact code.

10. **Which rule bans `print()` and how is it exempted where needed?**
    - Answer: `T20`; exempted per-line with `# noqa` in scripts meant to print.

11. **What is the security risk of PRs from forks in this setup?**
    - Answer: They do not receive repository secrets, so any step needing them fails; and running untrusted code with secrets would be dangerous.

12. **Why must `format-fix` run locally rather than in CI?**
    - Answer: CI should verify, not mutate code; auto-formatting hides the fact that a developer skipped the hook.

13. **What is the difference between Gitleaks and an image vulnerability scanner?**
    - Answer: Gitleaks finds committed secrets; an image scanner (Trivy/Grype) finds CVEs in packages and OS libraries.

14. **What does `steps.login-ecr.outputs.registry` provide, and why use it?**
    - Answer: The account's registry URL as an action output, avoiding a hard-coded URL that would differ per account/region.

15. **Where does the CD image get consumed downstream?**
    - Answer: ZenML's `settings.docker.parent_image` in each `configs/*.yaml` (Session 8.3).

---

## 📖 Glossary

| Term | Meaning |
|------|---------|
| **Workflow** | A YAML file in `.github/workflows/` describing triggers and jobs. |
| **Job** | A set of steps on one runner; jobs can run in parallel. |
| **Step** | A single shell command or action invocation within a job. |
| **Action** | A reusable unit (`actions/checkout@v3`) referenced by `uses:`. |
| **Runner** | The VM executing a job (`ubuntu-latest`). |
| **Concurrency group** | A named lock that lets new runs cancel older ones. |
| **Gitleaks** | Open-source secret scanner used as the security gate. |
| **Buildx** | Docker's build client; enables `--platform` and cache backends. |
| **ECR** | AWS Elastic Container Registry; stores the CD image. |
| **Immutable tag** | A tag (here, the SHA) that always points to the same image. |
| **Moving tag** | A tag (`latest`) that is reassigned on each build. |
| **OIDC** | Keyless AWS auth where GitHub issues a short-lived token; a safer alternative to static keys. |

---

## 📚 References

- [GitHub Actions](https://docs.github.com/en/actions)
- [GitHub Actions security hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Ruff](https://docs.astral.sh/ruff/)
- [Ruff rules reference](https://docs.astral.sh/ruff/rules/)
- [Gitleaks](https://github.com/gitleaks/gitleaks)
- [AWS ECR Login Action](https://github.com/aws-actions/amazon-ecr-login)
- [Docker build-push-action](https://github.com/docker/build-push-action)
- [pre-commit](https://pre-commit.com/)
- Book: *LLM Engineer's Handbook*, Chapter 11 (pages 463-480) - CI/CD.

---

## 🔗 Next Session

**Session 8.3**: ZenML Orchestration

We connect all the pipelines into a reproducible DAG with artifacts and caching.

---

## 📚 Additional Resources

- [GitHub Actions](https://docs.github.com/en/actions)
- [Ruff](https://docs.astral.sh/ruff/)
- [Gitleaks](https://github.com/gitleaks/gitleaks)
- [AWS ECR Login Action](https://github.com/aws-actions/amazon-ecr-login)
- Cross-links: [Session 8.1 Docker](session_8.1_docker.md), [Session 8.3 ZenML](session_8.3_zenml.md), [Session 6.1 FastAPI API](session_6.1_fastapi_api.md)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Session 8.1

**Outcome**: You can read, reproduce, and extend the CI/CD workflows, explain the gate philosophy and its tradeoffs, and reason about artifact immutability and secret safety.
