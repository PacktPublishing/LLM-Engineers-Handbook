# Session 2.1: Web Crawling with Selenium, Git, and LangChain

> Book reference: Chapter 3, *Data Engineering* (pages 84-125).
> Repo: `llm_engineering/application/crawlers/*.py`, `steps/etl/*.py`, `pipelines/digital_data_etl.py`, `tools/data_warehouse.py`.

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand the ETL data-collection architecture and why it is split into crawler + dispatcher + ODM layers.
- Explain the builder pattern used by `CrawlerDispatcher` and its **regex-based domain registry**.
- Read every crawler in the repo: `MediumCrawler`, `LinkedInCrawler`, `GithubCrawler`, `CustomArticleCrawler`.
- Distinguish Selenium crawling (login + scroll) from plain HTTP crawling (LangChain loaders) and from `git clone` subprocess crawling.
- Understand the ODM base class (`NoSQLBaseDocument`) that gives every crawler its `save()` and `find()` methods.
- Run the `digital_data_etl` ZenML pipeline from a YAML config and inspect its artifacts.
- Diagnose the common failure modes: ChromeDriver mismatch, Selenium hangs, duplicate documents, and blocked pages.

---

## 🏗️ Architecture Overview

### The ETL data-collection pipeline

Data engineering is the first stage of the LLM Twin. It is deliberately "not a book on data engineering" — the goal is the minimum needed to produce raw data for fine-tuning and RAG.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ETL Data Collection Pipeline                        │
│                                                                        │
│  Input: user_full_name + list[URL]                                     │
│        │                                                               │
│        ▼                                                               │
│  get_or_create_user(user_full_name)   ──► UserDocument (MongoDB)       │
│        │  first_name, last_name                                        │
│        ▼                                                               │
│  crawl_links(user, links)                                              │
│        │  CrawlerDispatcher.build()                                    │
│        │      .register_linkedin().register_medium().register_github() │
│        ▼                                                               │
│  ┌──────────────┬──────────────┬──────────────┬──────────────────────┐ │
│  │ Medium       │ LinkedIn     │ GitHub       │ CustomArticle        │ │
│  │ (Selenium)   │ (Selenium)   │ (git clone)  │ (LangChain loaders)  │ │
│  └──────┬───────┴──────┬───────┴──────┬───────┴──────────┬───────────┘ │
│         │   article    │   post       │  repository      │  article    │
│         ▼              ▼              ▼                  ▼            │
│      ArticleDocument / PostDocument / RepositoryDocument (MongoDB)     │
│                                                                        │
│  Output: list[str] of the original links (the raw docs live in Mongo)  │
└──────────────────────────────────────────────────────────────────────┘
```

**Three data categories, four crawlers.** Every crawled page collapses into one of three formats: `post`, `article`, or `repository`. The crawler chooses the *format*, not a per-source class. This is the key scaling decision: adding a new source (X, Dev.to, a blog) means one new crawler that outputs an existing category, and nothing downstream changes.

| Crawler | Base | Data category | Mechanism | Login |
|---------|------|---------------|-----------|-------|
| `MediumCrawler` | `BaseSeleniumCrawler` | article | Selenium + BeautifulSoup | no (public) |
| `LinkedInCrawler` | `BaseSeleniumCrawler` | post | Selenium + BeautifulSoup | yes (deprecated) |
| `GithubCrawler` | `BaseCrawler` | repository | `subprocess` `git clone` + `os.walk` | no |
| `CustomArticleCrawler` | `BaseCrawler` | article | LangChain `AsyncHtmlLoader` | no (fallback) |

### Why MongoDB as a data warehouse

Using a transactional DB as a warehouse is unusual. The book justifies it explicitly:

- **Small data**: hundreds of documents. A full scan is trivial.
- **Unstructured source**: crawled text has no fixed schema; a NoSQL document store avoids migration pain.
- **Operational simplicity**: one Docker image locally, a freemium cloud tier.
- **The tradeoff**: at millions of documents, a dedicated warehouse (Snowflake, BigQuery) wins. The book states this threshold plainly.

### Pipeline decoupling

The ETL pipeline and the feature pipeline never call each other. They communicate **only** through MongoDB:

```
ETL pipeline ──writes──► MongoDB ──reads──► Feature pipeline (Chapter 4)
```

This lets them run on independent schedules. The ETL can crawl hourly while feature engineering runs daily.

---

## 📁 Key Files Explained

### 1. `llm_engineering/application/crawlers/base.py` — the two base classes

**Purpose**: define one interface every crawler obeys, plus a reusable Selenium driver.

```python
# llm_engineering/application/crawlers/base.py
import time
from abc import ABC, abstractmethod
from tempfile import mkdtemp

import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from llm_engineering.domain.documents import NoSQLBaseDocument

# Check if the current version of chromedriver exists
# and if it doesn't exist, download it automatically,
# then add chromedriver to path
chromedriver_autoinstaller.install()


class BaseCrawler(ABC):
    model: type[NoSQLBaseDocument]

    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ...


class BaseSeleniumCrawler(BaseCrawler, ABC):
    def __init__(self, scroll_limit: int = 5) -> None:
        options = webdriver.ChromeOptions()

        options.add_argument("--no-sandbox")
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        options.add_argument("--remote-debugging-port=9226")

        self.set_extra_driver_options(options)

        self.scroll_limit = scroll_limit
        self.driver = webdriver.Chrome(
            options=options,
        )

    def set_extra_driver_options(self, options: Options) -> None:
        pass

    def login(self) -> None:
        pass

    def scroll_page(self) -> None:
        """Scroll through the LinkedIn page based on the scroll limit."""
        current_scroll = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or (self.scroll_limit and current_scroll >= self.scroll_limit):
                break
            last_height = new_height
            current_scroll += 1
```

**Design notes and tradeoffs**:

- **`model` is a class attribute, not a parameter.** Each subclass binds `model = ArticleDocument` (or `PostDocument` / `RepositoryDocument`). The base class then calls `self.model(...)` and `self.model.find(...)` generically. This is polymorphism: the dispatcher calls `crawler.extract(...)` without knowing the concrete class.
- **`extract()` returns `None`.** Crawlers persist directly; they do not return documents. The pipeline's output is the *list of links*, and MongoDB is the real store.
- **`AbstractMethod` on `extract`** forces every crawler to implement it, so `BaseCrawler()` cannot be instantiated.
- **`chromedriver_autoinstaller.install()` runs at import time**, so importing the module mutates `PATH`. It downloads the ChromeDriver matching the installed Chrome.
- **`--headless=new`** is the modern headless mode (Chrome 112+); the old `--headless` is legacy. This distinction matters: `new` behaves closer to headed Chrome, which matters for sites that sniff headless.
- **`mkdtemp()` for `--user-data-dir`, `--data-path`, `--disk-cache-dir`** gives each crawl a throwaway profile. This avoids profile-lock errors when multiple crawlers run, at the cost of no shared cache.
- **`--remote-debugging-port=9226`** is hard-coded; two concurrent Selenium crawlers would collide on this port. In the repo they run serially, so it is fine.
- **`scroll_page()` sleeps 5 seconds per scroll step.** This is the dominant cost of Selenium crawling: with `scroll_limit=5`, a single crawl can sleep 25+ seconds.

> **Workstation note (RTX 5000, 16 GB)**: crawling is CPU/network-bound, not GPU-bound. Selenium's headless Chrome uses system RAM, not VRAM. On this 32 GB machine you can run several crawlers, but the fixed `--remote-debugging-port` means keep them serial.

### 2. `llm_engineering/application/crawlers/dispatcher.py` — routing by regex

**Purpose**: choose the crawler for a URL.

```python
# llm_engineering/application/crawlers/dispatcher.py
import re
from urllib.parse import urlparse

from loguru import logger

from .base import BaseCrawler
from .custom_article import CustomArticleCrawler
from .github import GithubCrawler
from .linkedin import LinkedInCrawler
from .medium import MediumCrawler


class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {}

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        dispatcher = cls()

        return dispatcher

    def register_medium(self) -> "CrawlerDispatcher":
        self.register("https://medium.com", MediumCrawler)

        return self

    def register_linkedin(self) -> "CrawlerDispatcher":
        self.register("https://linkedin.com", LinkedInCrawler)

        return self

    def register_github(self) -> "CrawlerDispatcher":
        self.register("https://github.com", GithubCrawler)

        return self

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc

        self._crawlers[r"https://(www\.)?{}/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logger.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.")

            return CustomArticleCrawler()
```

**Key concepts**:

- **Builder pattern**: `build()` returns an empty dispatcher; `register_*()` return `self`, so calls chain: `CrawlerDispatcher.build().register_linkedin().register_medium().register_github()`. This is exactly how `steps/etl/crawl_links.py` constructs it.
- **Registry keys are regex strings**, not plain domains. `register("https://github.com", ...)` becomes the pattern `https://(www\.)?github\.com/*`. The `(www\.)?` group makes `www.github.com` and `github.com` both match. `re.escape(domain)` prevents the dots from acting as wildcards.
- **`get_crawler` uses `re.match`**, which anchors at the start of the string. So only URLs beginning with `https://` (or `https://www.`) match. A URL given as `http://...` or `github.com/...` (no scheme) falls through to `CustomArticleCrawler`.
- **The `for ... else` idiom**: the `else` runs only if the loop completes with no `break`. Here there is no `break`, so `else` always runs after the loop — but if any pattern matched, the function already `return`ed inside the loop. Net effect: default to `CustomArticleCrawler` when nothing matched.
- **In registration order**: insertion order is preserved in Python dicts, so `linkedin` is tested before `medium` before `github`. This only matters if patterns overlap, which they do not.

**Worked routing examples**:

| Input URL | Matched pattern | Crawler |
|-----------|-----------------|---------|
| `https://medium.com/@a/post` | `https://(www\.)?medium\.com/*` | `MediumCrawler` |
| `https://www.linkedin.com/in/x` | `https://(www\.)?linkedin\.com/*` | `LinkedInCrawler` |
| `https://github.com/u/repo` | `https://(www\.)?github\.com/*` | `GithubCrawler` |
| `https://decodingml.substack.com/p/x` | none | `CustomArticleCrawler` |
| `http://medium.com/x` | none (no `s`) | `CustomArticleCrawler` |

### 3. `llm_engineering/application/crawlers/github.py` — repository crawler

**Purpose**: clone a repo, flatten its tree into a dict, save it.

```python
# llm_engineering/application/crawlers/github.py
import os
import shutil
import subprocess
import tempfile

from loguru import logger

from llm_engineering.domain.documents import RepositoryDocument

from .base import BaseCrawler


class GithubCrawler(BaseCrawler):
    model = RepositoryDocument

    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self._ignore = ignore

    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Repository already exists in the database: {link}")

            return

        logger.info(f"Starting scrapping GitHub repository: {link}")

        repo_name = link.rstrip("/").split("/")[-1]

        local_temp = tempfile.mkdtemp()

        try:
            os.chdir(local_temp)
            subprocess.run(["git", "clone", link])

            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])  # noqa: PTH118

            tree = {}
            for root, _, files in os.walk(repo_path):
                dir = root.replace(repo_path, "").lstrip("/")
                if dir.startswith(self._ignore):
                    continue

                for file in files:
                    if file.endswith(self._ignore):
                        continue
                    file_path = os.path.join(dir, file)  # noqa: PTH118
                    with open(os.path.join(root, file), "r", errors="ignore") as f:  # noqa: PTH123, PTH118
                        tree[file_path] = f.read().replace(" ", "")

            user = kwargs["user"]
            instance = self.model(
                content=tree,
                name=repo_name,
                link=link,
                platform="github",
                author_id=user.id,
                author_full_name=user.full_name,
            )
            instance.save()

        except Exception:
            raise
        finally:
            shutil.rmtree(local_temp)

        logger.info(f"Finished scrapping GitHub repository: {link}")
```

**Key concepts**:

- **Not Selenium and not GitHub API**: it shells out to the `git` executable via `subprocess.run(["git", "clone", link])`. This requires `git` on `PATH`.
- **`os.chdir(local_temp)` then clone**: `git clone` with no target creates a subdirectory named after the repo, and the code assumes it is `os.listdir(local_temp)[0]`. This is fragile if the temp dir already holds files (it does not, freshly created).
- **`tree` is a `dict[str, str]`**: `{relative_file_path: file_contents}`. The `RepositoryDocument.content` field is typed `dict`.
- **`.replace(" ", "")` strips every space** from file content. This is aggressive but consistent; downstream cleaning and chunking never need indentation.
- **`errors="ignore"`** on `open` means non-UTF-8 bytes are dropped rather than raising. Essential when cloning arbitrary repos with binary-ish text files.
- **`ignore` filter is doubled**: it skips directories whose relative path *starts with* an ignore token and files whose name *ends with* one. `.toml`, `.lock`, `.png` are extensions; `.git` is a directory.
- **Cleanup in `finally`**: `shutil.rmtree(local_temp)` always runs. But note it does **not** `os.chdir` back — after `extract` the process CWD remains the now-deleted temp dir. A second crawl re-`chdir`s, so it recovers, but other relative-path code in the same process could break. This is a latent bug worth noticing.

**Edge cases**:
- A repo whose clone directory is not the only entry (e.g., partial clone) breaks the `os.listdir[0]` assumption.
- A repo with symlinks may produce unreadable paths; `errors="ignore"` helps but directory symlink loops are possible in theory.
- Large repos blow up `tree` in memory and later one `RepositoryDocument` per repo. The book keeps proof-of-concept repos small.

### 4. `llm_engineering/application/crawlers/custom_article.py` — the LangChain fallback

**Purpose**: fetch any article URL and convert HTML to text with LangChain.

```python
# llm_engineering/application/crawlers/custom_article.py
from urllib.parse import urlparse

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from loguru import logger

from llm_engineering.domain.documents import ArticleDocument

from .base import BaseCrawler


class CustomArticleCrawler(BaseCrawler):
    model = ArticleDocument

    def __init__(self) -> None:
        super().__init__()

    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")

            return

        logger.info(f"Starting scrapping article: {link}")

        loader = AsyncHtmlLoader([link])
        docs = loader.load()

        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        doc_transformed = docs_transformed[0]

        content = {
            "Title": doc_transformed.metadata.get("title"),
            "Subtitle": doc_transformed.metadata.get("description"),
            "Content": doc_transformed.page_content,
            "language": doc_transformed.metadata.get("language"),
        }

        parsed_url = urlparse(link)
        platform = parsed_url.netloc

        user = kwargs["user"]
        instance = self.model(
            content=content,
            link=link,
            platform=platform,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()

        logger.info(f"Finished scrapping custom article: {link}")
```

**Key concepts**:

- **Two LangChain classes**: `AsyncHtmlLoader` fetches raw HTML; `Html2TextTransformer` converts it to Markdown-ish text. Both live in `langchain_community`.
- **`platform` is the full netloc** (e.g., `decodingml.substack.com`), not a fixed string. Contrast with `MediumCrawler` which hard-codes `"medium"`.
- **No login, no scroll**: this is the simplest crawler and the safety net for every unregistered domain.
- **Tradeoff**: the book notes this is "fast to implement but hard to customize", and cites it as a reason some developers avoid LangChain in production. The loader works "decently in most scenarios" but you inherit its parsing decisions.

### 5. `llm_engineering/application/crawlers/medium.py` — Selenium article crawler

```python
# llm_engineering/application/crawlers/medium.py
from bs4 import BeautifulSoup
from loguru import logger

from llm_engineering.domain.documents import ArticleDocument

from .base import BaseSeleniumCrawler


class MediumCrawler(BaseSeleniumCrawler):
    model = ArticleDocument

    def set_extra_driver_options(self, options) -> None:
        options.add_argument(r"--profile-directory=Profile 2")

    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")

            return

        logger.info(f"Starting scrapping Medium article: {link}")

        self.driver.get(link)
        self.scroll_page()

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        title = soup.find_all("h1", class_="pw-post-title")
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")

        data = {
            "Title": title[0].string if title else None,
            "Subtitle": subtitle[0].string if subtitle else None,
            "Content": soup.get_text(),
        }

        self.driver.close()

        user = kwargs["user"]
        instance = self.model(
            platform="medium",
            content=data,
            link=link,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()

        logger.info(f"Successfully scraped and saved article: {link}")
```

**Key concepts**:

- **`set_extra_driver_options` overrides the base hook** to select a Chrome profile. The base class calls it before creating the driver, so subclass customization slots in cleanly.
- **`scroll_page()`** loads lazy content before `page_source` is read.
- **`soup.get_text()`** as `"Content"` means the body includes nav, footer, and boilerplate. Cleaning (Session 2.2) is what removes most of that; this crawler optimizes for *complete* capture over *clean* capture.
- **`platform="medium"`** is hard-coded, unlike the custom crawler's netloc.

### 6. `llm_engineering/application/crawlers/linkedin.py` — deprecated Selenium crawler

```python
# llm_engineering/application/crawlers/linkedin.py (excerpt)
class LinkedInCrawler(BaseSeleniumCrawler):
    model = PostDocument

    def __init__(self, scroll_limit: int = 5, is_deprecated: bool = True) -> None:
        super().__init__(scroll_limit)

        self._is_deprecated = is_deprecated

    def set_extra_driver_options(self, options) -> None:
        options.add_experimental_option("detach", True)

    def login(self) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its security measures, the login() method is no longer supported."
            )
        ...

    def extract(self, link: str, **kwargs) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its feed structure, the extract() method is no longer supported."
            )
        ...
```

**Why it is deprecated**: LinkedIn hardened its login and feed DOM. The repo keeps the class and its helpers (`_scrape_section`, `_extract_image_urls`, `_extract_posts`, `_scrape_experience`, `_scrape_education`) but fails fast with `DeprecationWarning`. This is a **deliberate, honest failure mode**: the pipeline logs the error per-link and continues rather than silently producing garbage.

**Takeaway for your own work**: a crawler that depends on a third party's DOM is fragile. Prefer APIs or stable feeds when available.

### 7. `llm_engineering/domain/documents.py` + ODM (`domain/base/nosql.py`)

Every crawler binds `model` to a Pydantic document class. The document inherits all persistence from `NoSQLBaseDocument`:

```python
# llm_engineering/domain/documents.py
class UserDocument(NoSQLBaseDocument):
    first_name: str
    last_name: str

    class Settings:
        name = "users"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"


class Document(NoSQLBaseDocument, ABC):
    content: dict
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")


class RepositoryDocument(Document):
    name: str
    link: str

    class Settings:
        name = DataCategory.REPOSITORIES


class PostDocument(Document):
    image: Optional[str] = None
    link: str | None = None

    class Settings:
        name = DataCategory.POSTS


class ArticleDocument(Document):
    link: str

    class Settings:
        name = DataCategory.ARTICLES
```

The ODM base provides the CRUD used by crawlers:

- `save()` → `collection.insert_one(self.to_mongo())` in the collection named by `Settings.name`.
- `find(**filter_options)` → `collection.find_one(...)`, used for the duplicate guard.
- `get_or_create(**filter_options)` → find-or-create, used for users.
- `bulk_insert(documents)`, `bulk_find(**filter_options)`.
- `from_mongo(data)` / `to_mongo()` convert between the `_id` string and the `id` UUID4.

Because `content` is a `dict`, `ArticleDocument` stores `{"Title": ..., "Subtitle": ..., "Content": ...}` while `RepositoryDocument` stores `{file_path: file_contents}`.

---

## 🔧 Pipeline Integration

### `steps/etl/get_or_create_user.py`

```python
# steps/etl/get_or_create_user.py
@step
def get_or_create_user(user_full_name: str) -> Annotated[UserDocument, "user"]:
    logger.info(f"Getting or creating user: {user_full_name}")

    first_name, last_name = utils.split_user_full_name(user_full_name)

    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="user", metadata=_get_metadata(user_full_name, user))

    return user
```

The helper `_get_metadata` records the query and the retrieved user, which shows up on the ZenML `user` artifact.

### `steps/etl/crawl_links.py`

```python
# steps/etl/crawl_links.py
@step
def crawl_links(user: UserDocument, links: list[str]) -> Annotated[list[str], "crawled_links"]:
    dispatcher = CrawlerDispatcher.build().register_linkedin().register_medium().register_github()

    logger.info(f"Starting to crawl {len(links)} link(s).")

    metadata = {}
    successfull_crawls = 0
    for link in tqdm(links):
        successfull_crawl, crawled_domain = _crawl_link(dispatcher, link, user)
        successfull_crawls += successfull_crawl

        metadata = _add_to_metadata(metadata, crawled_domain, successfull_crawl)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)

    logger.info(f"Successfully crawled {successfull_crawls} / {len(links)} links.")

    return links
```

**Failure isolation**: `_crawl_link` wraps each crawl in `try/except`, logs the error, and returns `(False, domain)`. One broken link never aborts the batch. The `crawled_links` artifact records, per domain, `successful` and `total` counts — a built-in quality report.

### `pipelines/digital_data_etl.py`

```python
# pipelines/digital_data_etl.py
from zenml import pipeline

from steps.etl import crawl_links, get_or_create_user


@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    user = get_or_create_user(user_full_name)
    last_step = crawl_links(user=user, links=links)

    return last_step.invocation_id
```

The pipeline returns the crawl step's `invocation_id`, which `end_to_end_data` uses to chain feature engineering.

### `tools/data_warehouse.py` — export and import

```python
# tools/data_warehouse.py (excerpt)
def __export_data_category(data_dir: Path, category_class: type[NoSQLBaseDocument]) -> None:
    data = category_class.bulk_find()
    serialized_data = [d.to_mongo() for d in data]
    export_file = data_dir / f"{category_class.__name__}.json"

    logger.info(f"Exporting {len(serialized_data)} items of {category_class.__name__} to {export_file}...")
    with export_file.open("w") as f:
        json.dump(serialized_data, f)
```

This is the escape hatch: if Selenium fails, import the book's backup from `data/data_warehouse_raw_data` with `--import-raw-data`. The expected state after import is **88 articles and 3 users**.

---

## 🛠️ Hands-On

### Step 1: Confirm the dispatcher routes correctly (no network)

```python
from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher

d = CrawlerDispatcher.build().register_linkedin().register_medium().register_github()
for url in [
    "https://medium.com/@a/post",
    "https://www.linkedin.com/in/x",
    "https://github.com/u/repo",
    "https://decodingml.substack.com/p/x",
]:
    print(type(d.get_crawler(url)).__name__, "<-", url)
```

Expected:
```
MediumCrawler <- https://medium.com/@a/post
LinkedInCrawler <- https://www.linkedin.com/in/x
GithubCrawler <- https://github.com/u/repo
CustomArticleCrawler <- https://decodingml.substack.com/p/x
```

### Step 2: Inspect the registry keys

```python
d = CrawlerDispatcher.build().register_medium().register_github()
print(list(d._crawlers.keys()))
# ['https://(www\\.)?medium\\.com/*', 'https://(www\\.)?github\\.com/*']
```

### Step 3: Query the warehouse with the ODM

```python
from llm_engineering.domain.documents import ArticleDocument, UserDocument

user = UserDocument.get_or_create(first_name="Paul", last_name="Iusztin")
articles = ArticleDocument.bulk_find(author_id=str(user.id))
print(f"User: {user.first_name} {user.last_name}; articles: {len(articles)}")
```

### Step 4: Run the real pipeline

```bash
# bring up MongoDB (and Qdrant) locally
docker compose up -d

# run the ETL for a named config in configs/
python -m tools.run --run-etl --no-cache --etl-config-filename digital_data_etl_paul_iusztin.yaml
```

### Step 5: Open the ZenML dashboard and read the artifacts

Inspect the `user` artifact (query + retrieved user) and the `crawled_links` artifact (per-domain success/total). These are the primary debugging surface.

---

## 📝 Exercise 1: Add a crawler for a new domain

**Task**: add a `DevToCrawler` that treats `dev.to` articles as `ArticleDocument`.

Requirements:
1. Subclass `BaseCrawler` (dev.to serves static HTML, so no Selenium needed).
2. Set `model = ArticleDocument`.
3. Guard duplicates with `self.model.find(link=link)`.
4. Fetch HTML and parse it. You may reuse LangChain loaders or use `requests` + BeautifulSoup.
5. Save with `platform="dev.to"`.
6. Add `register_devto()` to `CrawlerDispatcher` and include it in the chain in `crawl_links.py`.
7. Verify `get_crawler("https://dev.to/x/y")` returns your crawler.

**Goal**: internalize the dispatcher + base-class contract.

Template:

```python
from bs4 import BeautifulSoup
import requests
from loguru import logger

from llm_engineering.domain.documents import ArticleDocument
from .base import BaseCrawler


class DevToCrawler(BaseCrawler):
    model = ArticleDocument

    def extract(self, link: str, **kwargs) -> None:
        if self.model.find(link=link) is not None:
            logger.info(f"Article already exists: {link}")
            return

        soup = BeautifulSoup(requests.get(link, timeout=20).text, "html.parser")
        data = {
            "Title": getattr(soup.find("h1"), "text", "").strip(),
            "Subtitle": "",
            "Content": soup.get_text(),
        }
        user = kwargs["user"]
        self.model(
            content=data,
            link=link,
            platform="dev.to",
            author_id=user.id,
            author_full_name=user.full_name,
        ).save()
```

---

## 📝 Exercise 2: Make the GitHub crawler fail-safe

**Task**: fix the latent CWD bug and the temp-dir assumption in `GithubCrawler`.

1. Do **not** `os.chdir`; instead pass an explicit destination to `git clone`:
   ```python
   subprocess.run(["git", "clone", link, os.path.join(local_temp, "repo")], check=True)
   repo_path = os.path.join(local_temp, "repo")
   ```
2. Add `check=True` so a failed clone raises immediately and the `finally` still cleans up.
3. Bound the tree with a max file size (skip files over, say, 1 MB) so a huge repo does not exhaust memory.
4. Write a unit test that patches `subprocess.run` and asserts `shutil.rmtree` is called even when the clone fails.

**Goal**: reason about resource cleanup and subprocess error handling. Deletes are never captured by timestamp CDC (Session 4.3), so correctness at the ETL layer matters.

---

## 🐛 Common Pitfalls

- **ChromeDriver mismatch**: `chromedriver_autoinstaller.install()` usually fixes it, but a system Chrome updated mid-session can break. Bypass by commenting out Medium URLs in `configs/digital_data_etl_*.yaml`; Substack/blogs still work via `CustomArticleCrawler`.
- **Selenium hangs**: `scroll_page` sleeps 5s per step. Five steps is 25s per article. Do not crawl hundreds of Selenium URLs in one run.
- **Port collision**: `--remote-debugging-port=9226` is fixed. Running two Selenium crawlers concurrently fails. Keep them serial.
- **CWD mutation**: `GithubCrawler` calls `os.chdir` and never restores it. Code that relies on relative paths after a GitHub crawl can misbehave.
- **`http://` vs `https://`**: the registry patterns only match `https://`. An `http://` link silently routes to `CustomArticleCrawler`.
- **Duplicate documents**: the `find(link=...)` guard prevents re-inserting the same URL. Re-running ETL is safe for already-crawled links.
- **Deprecated LinkedIn**: `extract` raises `DeprecationWarning`; expect per-link failures and a lower `successful` count in metadata.
- **Blocked / paywalled Medium**: the book notes only non-paywalled Medium articles work.
- **`os.listdir(local_temp)[0]` assumption** in the GitHub crawler: breaks if the clone creates more than one entry.

---

## 🎓 Knowledge Check

1. **What pattern does `CrawlerDispatcher` use, and why?**
   - Answer: the **builder** pattern. `register_*` methods return `self`, enabling `build().register_linkedin().register_medium().register_github()` chaining.

2. **How are crawler dispatch keys stored?**
   - Answer: as regex pattern strings of the form `https://(www\.)?<domain>/*`, built in `register()` from the parsed netloc.

3. **Why `re.match` and not `re.search`?**
   - Answer: `re.match` anchors at the start, so only URLs that begin with `https://` (optionally `www.`) match; anything else falls back to `CustomArticleCrawler`.

4. **What is the role of the `model` class attribute on `BaseCrawler`?**
   - Answer: it binds each crawler to its document category, letting the base logic call `self.model(...)`, `.find()`, and `.save()` generically.

5. **Why is `BaseSeleniumCrawler` separate from `BaseCrawler`?**
   - Answer: only some sites need a real browser (login, scroll, JS). GitHub and most articles do not, so they use the lighter `BaseCrawler`.

6. **What does `chromedriver_autoinstaller.install()` do, and when does it run?**
   - Answer: downloads the ChromeDriver matching the installed Chrome and adds it to `PATH`; it runs at module import time.

7. **Why `--headless=new` instead of `--headless`?**
   - Answer: `new` is the modern headless mode with behavior closer to a headed browser, which reduces bot detection and rendering differences.

8. **How does the GitHub crawler avoid using Selenium?**
   - Answer: it shells out to `git clone` via `subprocess`, then walks the tree with `os.walk` — no browser needed.

9. **What does `RepositoryDocument.content` hold?**
   - Answer: a dict mapping relative file path to file contents (spaces stripped).

10. **Why does `CustomArticleCrawler` set `platform` to the netloc?**
    - Answer: it is generic and has no fixed source, so the domain itself is the platform identifier.

11. **How is a failed crawl handled in `crawl_links`?**
    - Answer: `_crawl_link` catches the exception, logs it, and returns `(False, domain)`; the loop continues and the failure is recorded in metadata.

12. **What is the duplicate guard in every crawler?**
    - Answer: `self.model.find(link=link)`; if a document with that link exists, the crawler returns early.

13. **Why store data in MongoDB rather than a traditional warehouse here?**
    - Answer: small volume, unstructured text, no fixed schema, and simple local/cloud operation. At millions of docs the book recommends Snowflake/BigQuery.

14. **What two classes does `CustomArticleCrawler` use from LangChain, and what is the tradeoff?**
    - Answer: `AsyncHtmlLoader` and `Html2TextTransformer`; fast to write, works decently, but hard to customize.

15. **What is the expected dataset after importing the backup?**
    - Answer: 88 articles and 3 users.

---

## 📖 Glossary

- **ETL**: Extract, Transform, Load — the three-phase data-collection process.
- **ODM**: Object-Document Mapper — maps Python objects to NoSQL documents (MongoDB). The repo's `NoSQLBaseDocument` is a hand-written ODM.
- **ORB / ORM**: Object-Relational Mapper, the SQL analogue (SQLAlchemy, SQLModel).
- **Builder pattern**: a creational pattern where methods return `self` to chain configuration, ending in a `build()`.
- **Dispatcher**: the routing layer that maps an input (URL) to the correct handler (crawler).
- **Headless browser**: a browser with no visible UI, driven programmatically.
- **ChromeDriver**: the executable Selenium uses to control Chrome/Chromium.
- **Netloc**: the network location in a URL (host:port), e.g. `medium.com`.
- **Duplicate guard**: the `find(link=...)` check that prevents re-inserting a crawled URL.

---

## 🔗 Next Session

**Session 2.2: Text Preprocessing Pipeline**

We take the raw documents stored in MongoDB and run them through the cleaning and chunking dispatchers.

---

## 📚 Additional Resources

- [Scrapy](https://github.com/scrapy/scrapy) — a mature crawling framework.
- [Crawl4AI](https://github.com/unclecode/crawl4ai) — crawling specialized for LLM data.
- [Selenium docs](https://www.selenium.dev/documentation/)
- [Builder pattern](https://refactoring.guru/design-patterns/builder)
- [MongoDB VS Code plugin](https://www.mongodb.com/products/tools/vs-code)
- [SQLAlchemy](https://www.sqlalchemy.org/) · [SQLModel](https://github.com/fastapi/sqlmodel)
- [pymongo errors](https://pymongo.readthedocs.io/en/stable/api/pymongo/errors.html)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Session 1.1, Session 1.2, basic HTML knowledge, Chrome installed, MongoDB running (`docker compose up -d`)

**Outcome**: You can read, extend, and debug the crawler layer; you understand the dispatcher's regex routing, the two base classes, all four crawlers, and the ODM document classes. You can run the `digital_data_etl` pipeline from a YAML config and interpret its artifacts.
