# Session 9.3: Security Best Practices

## 🎯 Learning Objectives

By the end of this session, you will:
- Inventory every secret and where it must (and must not) live
- Recognize the security gaps in the current API and local stack
- Apply secret scanning, least-privilege, and input validation
- Understand prompt-injection risk in RAG context
- Produce a prioritized hardening plan
- Read an IAM policy set and judge it against least privilege
- Distinguish a trust boundary from a network boundary

> This session is defensive: it audits this codebase and shows how to reduce risk. It is not a penetration-testing guide.

---

## 🏗️ Architecture Overview

Security is organized around **trust boundaries**: places where data or control crosses from a less-trusted zone to a more-trusted one. Each boundary is where a check belongs.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Trust boundaries                                    │
│                                                                            │
│  Secrets:  .env (local)  ──►  ZenML secret store (remote)                 │
│            SageMaker job: environment vars (NOT hyperparameters)          │
│            CI: GitHub repository secrets                                  │
│                                                                            │
│  API:      internet ──► /rag (no auth, no rate limit) ──► AWS SageMaker    │
│                     └── untrusted query ──► retriever ──► LLM prompt        │
│                                                                            │
│  Data:     MongoDB (default creds)   Qdrant (no auth locally)             │
│                                                                            │
│  Corpus:   crawled websites (UNTRUSTED) ──► context string ──► LLM          │
│                                                                            │
│  Supply chain: poetry.lock pinned  +  gitleaks in CI                      │
└──────────────────────────────────────────────────────────────────────────┘
```

The two boundaries most people miss:

1. **The corpus is untrusted input.** Crawled web pages flow into the generation prompt as context. An attacker who can get a page crawled can attempt prompt injection.
2. **The model's output is untrusted.** It is rendered to users and, in a feedback loop, could be re-ingested. Treat it as data.

| Boundary | Untrusted side | Trusted side | Control |
|----------|---------------|--------------|---------|
| Internet → API | client | service | auth, rate limit, input size |
| Corpus → prompt | crawled text | LLM context | delimit and escape context |
| Client → Mongo | any process that can reach port 27017 | warehouse | strong creds, private network |
| CI → cloud | PR author | AWS account | gitleaks, scoped IAM, no static keys |
| App → Opik | user data | third-party tracer | local Opik, redaction |

---

## 📁 Secret Management

### 1. Never commit secrets

```bash
# .env.example - template only, values are placeholders
OPENAI_MODEL_ID=gpt-4o-mini
OPENAI_API_KEY=str
HUGGINGFACE_ACCESS_TOKEN=str
COMET_API_KEY=str
DATABASE_HOST="mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
USE_QDRANT_CLOUD=false
QDRANT_CLOUD_URL=str
QDRANT_APIKEY=str
AWS_ARN_ROLE=str
AWS_REGION=eu-central-1
AWS_ACCESS_KEY=str
AWS_SECRET_KEY=str
```

- **`.env.example` holds placeholders**, not real values. Note the placeholder string is literally `str`, so a copied-and-unedited `.env` has useless values — a deliberate fail-loud design.
- **`.env` is gitignored**. Verified at `.gitignore:125` (`.env` in the "Environments" block). **`.env.testing` is not separately listed but matches no pattern**, so confirm it before relying on it; the safe check is `git check-ignore -v .env.testing`.
- **`gitleaks-check` runs in CI** and scans `llm_engineering/` for committed secrets (`.github/workflows/ci.yaml:34-35`). This is the mechanical backstop.
- **`.pre-commit-config.yaml` also runs gitleaks** locally (rev `v8.18.2`), catching a secret before it is ever committed. CI is the second gate; pre-commit is the first.

```yaml
# .pre-commit-config.yaml
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.2
    hooks:
      - id: gitleaks
```

**Rule:** if a secret ever reaches a commit, **rotate it immediately**. Removing the file does not remove it from history. A secret that was pushed must be treated as public forever, even after a force-push, because forks and clones retain it.

### 2. Remote runs use the ZenML secret store

```python
# llm_engineering/settings.py
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

    def export(self) -> None:
        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()
        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning(
                "Secret 'scope' already exists. Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )
```

- **`export()` pushes all settings to ZenML** so remote steps read secrets from the platform, not a shipped `.env`. Run it with `poetry poe export-settings-to-zenml`.
- **`settings.json`/files are never shipped to remote steps**; ZenML injects the secret values at runtime.
- **`load_settings()` degrades gracefully** to `.env`/defaults. That convenience is a **production risk**: if the ZenML lookup fails silently, a remote run uses weak defaults (e.g. the dev Mongo credentials) and the only signal is a warning log. Verify the "Loading settings from the ZenML secret store" line in production logs.
- **Two code-level smells worth noting** (not exploitable, but confusing):
  - `export()` iterates the module-global `settings`, not `self`. Inside the classmethod-free method, `self` is ignored, so calling `some_other_settings.export()` would export the global. It works because there is one global, but it is surprising.
  - The warning says `Secret 'scope' already exists`, but the secret is named `settings`. A copy-paste slip; the message is misleading during incident response.

### 3. Training/evaluation containers

```python
# llm_engineering/infrastructure/aws/deploy/sagemaker.py (deploy call)
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
        hyperparameters=hyperparameters,
```

- **Secrets go in `environment`, never `hyperparameters`.** Hyperparameters are logged, shown in the AWS console, and often persisted with the job definition; environment variables are not surfaced the same way.
- This is a deliberate, correct choice in the code. Do not "simplify" by moving a token into hyperparameters.

---

## 📁 API Security Gaps

### 1. No authentication or rate limiting

The entire API is one file with three functions:

```python
# llm_engineering/infrastructure/inference_pipeline_api.py
@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()
    return answer


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)
    answer = call_llm_service(query, context)
    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )
    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Gaps**:
- **No auth**: anyone who can reach the service can invoke the LLM, incurring cost.
- **No rate limiting**: an unauthenticated client can exhaust your OpenAI/SageMaker budget. A single loop hitting `/rag` also triggers 3× token-count lookups and a full retrieval each time.
- **`detail=str(e)`** leaks exception messages (and potentially internals) to clients.
- **`QueryRequest` validates type but not size** — an enormous `query` inflates retrieval and generation cost.
- **`k=3` is hard-coded**, so an attacker cannot ask for more context, but they *can* ask for many requests.

**Hardening**:

```python
from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)

API_KEY = settings.COMET_API_KEY  # or, better, a dedicated service key


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


async def require_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest, _=Depends(require_key)):
    try:
        return {"answer": rag(query=request.query)}
    except Exception:
        logger.exception("RAG failed")  # full detail server-side only
        raise HTTPException(status_code=500, detail="Internal error") from None
```

Add `slowapi` or an API gateway for rate limiting. The key lesson is the pattern: **authenticate at the boundary, validate shape and size, log detail server-side, return a generic message**.

> Using `settings.COMET_API_KEY` as the service key is expedient but wrong in spirit: it couples the observability credential to the API auth and cannot be rotated independently. A dedicated `SERVICE_API_KEY` setting is the correct fix.

### 2. Binding and transport

```python
# pyproject.toml poe task
run-inference-ml-service = "poetry run uvicorn tools.ml_service:app --host 0.0.0.0 --port 8000 --reload"
```

- **`0.0.0.0`** exposes the service on all interfaces. On an untrusted network, this is reachable by anyone on the LAN.
- **No TLS**: the API is plain HTTP. Terminate TLS at a reverse proxy or gateway.
- **`reload=True`** must be off in production. `--reload` watches files and restarts the server; it is a development convenience with a process-management overhead and a larger attack surface.

### 3. Input validation

- **`QueryRequest` validates shape** (`query: str`), but not size. Add `min_length`/`max_length` as above.
- **No CORS policy** is configured; add `CORSMiddleware` with an explicit allow-list if a browser client is added. Without it, the browser default is restrictive (good); do not add `allow_origins=["*"]` reflexively.
- **No output sanitization**: the answer is returned verbatim. If ever rendered as HTML, escape it (XSS).

### 4. Error handling and DoS

`HTTPException(detail=str(e))` on the broad `except Exception` converts every internal error into a 500 with an attacker-influenced message. Two problems: information disclosure and, if `str(e)` echoes part of the query, a reflected-content vector. The hardened version logs server-side and returns a constant.

---

## 📁 Data-Layer Security

### 1. Local databases have no auth protection beyond defaults

```yaml
# docker-compose.yml
    environment:
      MONGO_INITDB_ROOT_USERNAME: "llm_engineering"
      MONGO_INITDB_ROOT_PASSWORD: "llm_engineering"
```

- **Default credentials are public knowledge.** Acceptable only on an isolated dev host. Never expose ports 27017/6333 to a public network.
- **Qdrant has no authentication locally.** Do not bind it publicly.
- **Production**: put both behind a private network, use strong unique credentials, and enable TLS. Qdrant Cloud (via `USE_QDRANT_CLOUD`) provides managed auth.
- **The connection string embeds the password** in `DATABASE_HOST` (`mongodb://llm_engineering:llm_engineering@127.0.0.1:27017`). It will therefore appear in logs, traces, and error messages if the URI is ever printed. Use a secrets manager and a passwordless-in-logs pattern.

### 2. AWS least privilege

`create_execution_role.py` attaches **four `*FullAccess` managed policies**:

```python
# llm_engineering/infrastructure/aws/roles/create_execution_role.py
        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
            "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        ]

        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)
```

**Risk**: the role can touch far more than it needs. `AmazonS3FullAccess` reaches *every* bucket in the account; `AmazonEC2ContainerRegistryFullAccess` pushes and deletes *any* image. A compromised training job inherits all of it.

**Recommendation**: replace with scoped policies limited to the specific S3 buckets, ECR repositories, and SageMaker resources the pipeline uses. This is a textbook least-privilege improvement.

`create_sagemaker_role.py` is **even broader** and structurally worse: it creates a long-lived **IAM user with a static access key** and attaches five policies, two of which are account-wide administration:

```python
# llm_engineering/infrastructure/aws/roles/create_sagemaker_role.py
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AWSCloudFormationFullAccess",
        "arn:aws:iam::aws:policy/IAMFullAccess",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ]

    for policy in policies:
        iam.attach_user_policy(UserName=username, PolicyArn=policy)

    response = iam.create_access_key(UserName=username)
```

- `IAMFullAccess` lets the identity create and attach *any* policy — including granting itself more. `AWSCloudFormationFullAccess` lets it deploy arbitrary infrastructure. Combined, a leaked key is effectively account takeover.
- The script writes the secret key to `sagemaker_user_credentials.json`. That file is gitignored only by the `sagemaker_*.json` rule (`.gitignore:172`), which **does** match it — verify with `git check-ignore -v sagemaker_user_credentials.json`. Even so, the static key has an unbounded lifetime.
- **Prefer the execution-role pattern; avoid static user keys where possible** (use roles and OIDC federation).
- **Code smell**: the `logger.info("Credentials saved...")` at the very end is **unindented**, sitting at module scope after the `if __name__ == "__main__":` block (line 58). It executes on *import*, not just when run as a script. Harmless in effect (it logs a message) but it signals the file was not linted, and it makes the module print on import. The `run` script's `create-sagemaker-role` task would trigger it.

### 3. IAM policy comparison

| Script | Identity type | Policies | Worst-case power | Lifetime |
|--------|--------------|----------|------------------|----------|
| `create_execution_role.py` | IAM **role** (assumed by SageMaker) | 4 × FullAccess | broad data + compute | temporary STS creds |
| `create_sagemaker_role.py` | IAM **user** + static access key | 5 × FullAccess incl. IAM + CloudFormation | account takeover | permanent until rotated |

The role is strictly better even before scoping, because its credentials are short-lived.

---

## 📁 RAG-Specific Security: Prompt Injection

Retrieved documents are inserted into the generation prompt:

```python
# llm_engineering/model/inference/run.py
class InferenceExecutor:
    def __init__(self, llm, query, context=None, prompt=None) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}
            """
        else:
            self.prompt = prompt

    def execute(self) -> str:
        self.llm.set_payload(
            inputs=self.prompt.format(query=self.query, context=self.context),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
            },
        )
        answer = self.llm.inference()[0]["generated_text"]
        return answer
```

The dangerous call is `self.prompt.format(query=..., context=...)`: `str.format` interprets `{` and `}` in the *inserted* values only insofar as they are literal, but a `{` left unescaped anywhere the template does not expect it raises. More precisely, brace characters **inside the interpolated `context` string are fine** (they are values, not template), but braces in a *user-supplied `prompt` template* along with braces in `context` combine badly. The exploitable path in this code is any place a template string is built by concatenating untrusted text before `.format`, and the DoS in the harness below. The robust fix is to never `.format` untrusted text: build the prompt with an f-string over pre-escaped parts, or a templating engine with explicit escaping.

**Risks**:
- **Prompt injection**: if crawled content contains instructions like "ignore previous instructions", the model may follow them. The corpus is untrusted (arbitrary websites).
- **`str.format` fragility**: content containing `{` or `}` raises `KeyError`/`ValueError`. Because the corpus is attacker-influenced, crafted content can trigger a 500 — a denial-of-service vector. Code snippets are a common source of stray braces in crawl text.
- **Data exfiltration**: a malicious document could attempt to steer the model into revealing other context, or (combined with a shared cache) another user's data.

**Mitigations**:
- Treat retrieved content as data, not instructions. Use a template engine that does not interpret braces, or escape `{`/`}` in the context before formatting (`context.replace("{", "{{").replace("}", "}}")`).
- Delimit context clearly and instruct the model to use it only as reference material.
- Consider a guardrail/filter on inputs and outputs.
- Prefer `jinja2` (already a dependency) with `.format` semantics disabled, or a simple concatenation, for prompts that embed untrusted text.

---

## 📁 Observability and PII

Opik traces store full queries, retrieved context, and answers:

```python
    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )
```

**Risk**: personal data and proprietary content end up in a third-party trace store. The metadata here is limited to ids and counts (good), but `@opik.track` captures the function inputs and outputs by default, so the full `query`, `context`, and `answer` strings are recorded.

**Mitigations**:
- Use local Opik (`use_local=True`) for sensitive corpora.
- Redact or hash PII before it enters the trace.
- Set a retention policy in Opik.
- Use `@opik.track(ignore_arguments=...)` or manual spans to avoid logging raw content when it is sensitive.

---

## 📁 Supply Chain and Container Hygiene

- **`poetry.lock` pins dependencies**, and CI installs from it (`poetry install --only dev` for QA, full install for tests) — good reproducibility. Pin by hash for stronger guarantees.
- **`gitleaks` runs both pre-commit and in CI** — two gates.
- **Docker runs as root**: the `Dockerfile` never creates a non-root `USER`. Add one for defense in depth:

```dockerfile
RUN useradd -m appuser
USER appuser
```

- **Base image**: `python:3.11-slim-bullseye` is fine; pin an image digest for reproducibility and scan with Trivy/Dependabot. `poetry install` pulls from PyPI; a compromised transitive package executes at build time.
- **CI action pinning**: `.github/workflows/ci.yaml` uses `actions/checkout@v3` and `actions/setup-python@v3` by tag. Pin to a commit SHA in hardened environments.

---

## 🛠️ Hands-On: Audit and Harden

### Step 1: Confirm `.env` is ignored

```bash
git check-ignore -v .env
# should print: .gitignore:125:.env    .env
git check-ignore -v sagemaker_user_credentials.json
```

### Step 2: Run the secret scan locally

```bash
poetry poe gitleaks-check
```

### Step 3: Grep for risky patterns

```bash
git grep -n "str(e)" -- llm_engineering
git grep -n "0.0.0.0" -- pyproject.toml tools
git grep -n "FullAccess" -- llm_engineering
git grep -n "create_access_key" -- llm_engineering
```

### Step 4: Add API-key auth and a query length limit

Apply the snippets above, then verify:

```bash
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d "{\"query\":\"hi\"}"
# 401 without the key
curl -X POST http://localhost:8000/rag -H "x-api-key: <key>" -H "Content-Type: application/json" -d "{\"query\":\"hi\"}"
# 200 with the key
```

### Step 5: Understand where brace handling can bite

`InferenceExecutor.execute` calls `self.prompt.format(query=..., context=...)`. The default template is a hardcoded constant, so **braces inside `context` are safe** — they are values, not template syntax:

```python
template = "Context: {context}"
print(template.format(context="{unclosed"))  # prints "Context: {unclosed" — no error
```

The real risk appears when the *template* is attacker-influenced. `InferenceExecutor` accepts a `prompt` argument, and `str.format` resolves field names via attribute and item access:

```python
# If an attacker controls the template:
"{query.__class__.__mro__}".format(query="hello")   # leaks the class hierarchy
"{query.__init__.__globals__}".format(query="hello")  # can expose process internals
```

This is **format-string injection**. The mitigation is to never `.format` an untrusted template: keep templates as trusted constants (as the default here does), and if a custom template must be accepted, validate its allowed fields or use a non-executing substitution.

```python
def render(template: str, **values) -> str:
    # reject unexpected field names rather than executing attribute access
    allowed = set(values)
    import string
    for _, field, _, _ in string.Formatter().parse(template):
        if field and field not in allowed:
            raise ValueError(f"Disallowed template field: {field!r}")
    return template.format(**values)
```

Then verify the guard rejects the attack:

```python
try:
    render("{query.__class__}", query="x")
except ValueError as exc:
    print("blocked:", exc)
```

---

## 📝 Exercise 1: Write a Hardening Plan

### Task

Produce a prioritized remediation table for this repository.

| Priority | Finding | Impact | Mitigation |
|----------|---------|--------|------------|
| P0 | Unauthenticated `/rag` | Cost abuse | API key / gateway auth |
| P0 | Default DB credentials exposed | Data compromise | Strong creds + private network |
| P0 | `create_sagemaker_role.py` static key + `IAMFullAccess` | Account takeover | Delete user; use scoped execution role + OIDC |
| P1 | `detail=str(e)` leaks internals | Info disclosure | Log server-side, generic message |
| P1 | `*FullAccess` IAM policies | Over-privilege | Scoped policies |
| P1 | Prompt injection via context | Output manipulation | Treat context as data, escape braces |
| P1 | `str.format` brace DoS | Availability | Escape `{`/`}` or use safe templating |
| P2 | `0.0.0.0` + no TLS | Network exposure | Reverse proxy + TLS |
| P2 | Root container | Lateral movement | Non-root `USER` |
| P2 | PII in Opik traces | Privacy | Local Opik / redaction |
| P3 | Unpinned CI actions | Supply chain | Pin to commit SHAs |

Add any findings you discover yourself, then order by exploitability × impact.

**Goal**: Practice turning an audit into an actionable, prioritized plan.

---

## 📝 Exercise 2: Threat-Model One Boundary

### Task

Pick the **corpus → prompt** boundary and write a one-page threat model.

1. **Assets**: user queries, the model, the AWS account, other users' data.
2. **Entry points**: crawled pages, user queries, the `/feedback` endpoint (if added).
3. **Threats**: prompt injection, brace DoS, indirect injection via a poisoned document.
4. **Current controls**: `k=3` limits context size; the retriever filters by `author_id`.
5. **Gaps and mitigations**: escaping, delimiter hardening, output filtering, per-tenant cache keys.
6. **Residual risk**: what remains after mitigations.

**Goal**: Move from a checklist to reasoning about a specific attacker and control set. A threat model names the attacker; a checklist does not.

---

## 📁 STRIDE Threat Model for the RAG Service

STRIDE is a structured way to enumerate threats at each trust boundary. Apply it to the `/rag` service.

| STRIDE | Threat to this service | Concrete example | Control |
|--------|------------------------|------------------|---------|
| **S**poofing | Fake client identity | Anonymous caller impersonating a paying user | API key / JWT |
| **T**ampering | Modify data in transit or at rest | Edit a crawled page to inject instructions | TLS, input validation, treat corpus as data |
| **R**epudiation | Deny having made a request | No audit trail of who queried what | Request IDs, Opik traces |
| **I**nformation disclosure | Leak secrets or other users' data | `detail=str(e)`; shared answer cache across tenants | Generic errors; per-tenant cache key |
| **D**enial of service | Exhaust budget or crash the service | Unbounded `query`; `{` in context breaks `str.format` | Rate limit, max length, escape braces |
| **E**levation of privilege | Gain more rights than intended | `IAMFullAccess` on a static key | Least privilege, roles over users |

**The two highest-severity items**: Spoofing (unauthenticated endpoint) and Elevation of privilege (static IAM user with admin policies). Both are P0 because they turn a single flaw into account or budget compromise.

**Why STRIDE and not a checklist**: a checklist tells you what to check; STRIDE tells you *which attacker goal* each control addresses, so you can reason about coverage. Missing any STRIDE letter is a category of risk you have not considered.

---

## 📁 Secret Rotation Runbook

A leaked secret is an incident with a procedure. Write it before you need it.

```
1. CONTAIN
   - Revoke the exposed credential at the source (OpenAI, HF, AWS, Comet).
   - If AWS: deactivate the access key; do not merely delete the user.

2. ROTATE
   - Issue a new credential; update .env locally, ZenML secret store remotely,
     and GitHub repository secrets for CI.
   - For remote runs: poetry poe export-settings-to-zenml (after updating .env).

3. VERIFY
   - Run poetry poe call-inference-ml-service and confirm 200.
   - Check the "Loading settings from the ZenML secret store" log line.

4. PURGE (best-effort)
   - The secret is already public; purging git history reduces casual exposure
     but does not un-leak it. Treat rotation, not purge, as the fix.

5. REVIEW
   - How did it enter the repo? Add a targeted gitleaks rule or pre-commit hook.
```

**The non-negotiable step is rotation.** Everything else is damage control. `.env` and `.env.example` never hold real values; the real ones live in `.env` (gitignored) and the ZenML secret store.

---

## 🧪 Security Testing

Static checks catch classes of bug before runtime.

| Check | Tool | What it catches | Where it runs |
|-------|------|-----------------|---------------|
| Secret scan | gitleaks | committed keys/tokens | pre-commit + CI |
| Lint | ruff | unsafe patterns, unused imports | CI (`lint-check`) |
| Format | ruff format | inconsistent code (weak signal) | CI (`format-check`) |
| Dependency audit | `pip-audit` / Dependabot | known CVEs in pinned deps | add to CI |
| Container scan | Trivy | vulnerable base image / OS packages | add to CD |
| Auth test | pytest + TestClient | endpoint rejects missing key | add to `tests/` |

A minimal auth regression test (the missing test today):

```python
from fastapi.testclient import TestClient

from llm_engineering.infrastructure.inference_pipeline_api import app

client = TestClient(app)


def test_rag_requires_api_key():
    response = client.post("/rag", json={"query": "hi"})
    assert response.status_code == 401


def test_rag_rejects_oversized_query():
    response = client.post("/rag", json={"query": "a" * 10_000}, headers={"x-api-key": "test"})
    assert response.status_code in (401, 422)
```

**Why tests beat reviews for security**: a reviewer forgets; a test fails forever. Every control in the table above should eventually have a test or a gate.

---

## 🐛 Common Pitfalls

- **Deleting a leaked secret without rotating it**: the value remains in Git history. Rotate first.
- **Trusting the corpus**: crawled content is untrusted input. Never let it define instructions unescaped.
- **Assuming local defaults are safe**: `llm_engineering/llm_engineering` in Compose is a dev convenience, not a production credential.
- **Over-broad IAM for speed**: the `*FullAccess` policies are convenient but undermine least privilege; combined with a static key they are account-takeover grade.
- **Ignoring trace PII**: observability stores real user data; `@opik.track` logs inputs and outputs by default.
- **Reusing the Comet key as the API key**: couples two credentials that must rotate independently.
- **Reflexive `allow_origins=["*"]`**: an open CORS policy turns a browser into an open proxy to your LLM.
- **Forgetting `--reload`**: shipping a dev server flag to production invites file-watch overhead and restarts.
- **Trusting a `.gitignore` match without checking**: run `git check-ignore -v <file>`; patterns like `sagemaker_*.json` are easier to get wrong than `.env`.
- **Treating silent fallback as safe**: `load_settings` falls back to `.env`/dev defaults on failure; verify the log line in production.

---

## 🎓 Knowledge Check

1. **Why do secrets go in `environment` rather than `hyperparameters` for SageMaker?**
   - Answer: Hyperparameters are logged and visible; environment variables are not surfaced the same way.

2. **What is the API's biggest security gap?**
   - Answer: No authentication or rate limiting on `/rag`.

3. **Why is `detail=str(e)` risky?**
   - Answer: It can leak internal exception details to clients, and may reflect query-derived content.

4. **What is prompt injection in this project?**
   - Answer: Untrusted crawled content in the context steering the model's behavior.

5. **Why are the `*FullAccess` IAM policies a problem?**
   - Answer: They grant far more permission than the pipeline needs, violating least privilege.

6. **What is the first action after a committed secret is found?**
   - Answer: Rotate the secret; removing the file is not enough.

7. **Why is `create_sagemaker_role.py` worse than `create_execution_role.py`?**
   - Answer: It creates a permanent IAM user with a static access key and attaches `IAMFullAccess` and `AWSCloudFormationFullAccess`; the execution role uses temporary credentials and no user.

8. **What is the correct path to the execution-role script?**
   - Answer: `llm_engineering/infrastructure/aws/roles/create_execution_role.py` (there is no top-level `infrastructure/` directory).

9. **How can untrusted content cause a denial of service without any prompt injection?**
   - Answer: A `{` or `}` in the context makes `str.format` raise `KeyError`/`ValueError`, returning a 500.

10. **Why is the unindented `logger.info` at `create_sagemaker_role.py:58` notable?**
    - Answer: It sits at module scope after the `__main__` guard, so it executes on import, and it signals the file was not linted.

11. **What does `@opik.track` capture by default that is a privacy concern?**
    - Answer: The function's arguments and return value — here, the full query, context, and answer.

12. **Why not reuse `COMET_API_KEY` as the service API key?**
    - Answer: It couples the observability credential to API auth; neither can be rotated independently.

13. **Where should TLS terminate for this API?**
    - Answer: At a reverse proxy or API gateway in front of uvicorn; the app itself speaks plain HTTP.

14. **What are the two secret-scanning gates in the repo?**
    - Answer: The `gitleaks` pre-commit hook (`.pre-commit-config.yaml`) and the `gitleaks-check` CI job (`.github/workflows/ci.yaml`).

15. **Why is silently falling back to `.env` a production risk?**
    - Answer: A failed ZenML secret lookup lets a remote run use weak dev defaults while only logging a warning; you must verify the load line.

---

## 📖 Glossary

- **Trust boundary**: A point where data crosses between security zones; where controls belong.
- **Least privilege**: Granting only the permissions required, no more.
- **Prompt injection**: Untrusted text in a prompt that alters model behavior.
- **Indirect prompt injection**: Injection via retrieved/crawled content rather than direct user input.
- **gitleaks**: A secret-scanning tool run pre-commit and in CI.
- **IAM role vs user**: A role issues temporary credentials via STS; a user can hold a permanent access key.
- **OIDC federation**: Exchanging an external identity token for short-lived AWS credentials, avoiding static keys.
- **Static access key**: A long-lived AWS secret; a leak is permanent until rotated.
- **PII**: Personally identifiable information.
- **CORS**: Browser policy controlling which origins may call an API.
- **XSS**: Cross-site scripting; relevant if the answer is ever rendered as HTML.
- **Denial of service (DoS)**: Making a service unavailable; here, via crafted content or request volume.
- **TLS termination**: Ending the encrypted connection at a proxy/gateway.
- **SBOM / dependency scan**: Inventorying and checking third-party packages for vulnerabilities.

---

## 🔚 Course Complete

You have worked through all nine modules:

1. Foundations and DDD
2. Data engineering
3. Dataset generation
4. Advanced RAG
5. Training and fine-tuning
6. Inference and APIs
7. Monitoring and evaluation
8. Production deployment
9. Advanced topics

The capstone is to run `end_to_end_data`, train SFT then DPO, deploy the endpoint, and evaluate — then harden the system using this session's plan.

---

## 📚 Additional Resources

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [AWS IAM Least Privilege](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Gitleaks](https://github.com/gitleaks/gitleaks)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [AWS: IAM roles vs users](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html)
- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)

---

## 🔎 References

- **Book**: *LLM Engineer's Handbook* — cross-cutting security concerns across Chapters 5, 8, and 11 (deployment, safeguards, secrets).
- **Repo**: `llm_engineering/settings.py`, `.env.example`, `.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yaml`, `llm_engineering/infrastructure/inference_pipeline_api.py`, `llm_engineering/infrastructure/aws/roles/create_execution_role.py`, `llm_engineering/infrastructure/aws/roles/create_sagemaker_role.py`.
- **Related sessions**: [`session_5.3_sagemaker_deployment.md`](session_5.3_sagemaker_deployment.md) (IAM roles), [`session_6.1_fastapi_api.md`](session_6.1_fastapi_api.md) (the API), [`session_6.2_rag_inference_flow.md`](session_6.2_rag_inference_flow.md), [`session_8.2_cicd.md`](session_8.2_cicd.md) (gitleaks in CI).
- **Curriculum**: [`../CURRICULUM.md`](../CURRICULUM.md).

---

**Estimated Time**: 4-5 hours

**Prerequisites**: [Session 6.1](session_6.1_fastapi_api.md), [Session 6.2](session_6.2_rag_inference_flow.md), [Session 8.2](session_8.2_cicd.md)

**Outcome**: You can audit the project's trust boundaries and produce a prioritized hardening plan.
