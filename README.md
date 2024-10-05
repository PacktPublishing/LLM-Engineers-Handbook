# LLM-Engineering

Repository that contains all the code used throughout the [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

![Book Cover](/images/book_cover.png)

# Dependencies

## Local dependencies

To install and run the project locally, you need the following dependencies (the code was tested with the specified versions of the dependencies):

- [pyenv 2.3.36](https://github.com/pyenv/pyenv) (optional: for installing multiple Python versions on your machine)
- [Python 3.11](https://www.python.org/downloads/)
- [Poetry 1.8.3](https://python-poetry.org/docs/#installation)
- [Docker 27.1.1](https://docs.docker.com/engine/install/)

## Cloud services

The code also uses and depends on the following cloud services. For now, you don't have to do anything. We will guide you in the installation and deployment sections on how to use them:

- [HuggingFace](https://huggingface.com/): Model registry
- [Comet ML](https://www.comet.com/site/): Experiment tracker
- [Opik](https://www.comet.com/site/products/opik/): LLM evaluation and prompt monitoring
- [ZenML](https://www.zenml.io/): Orchestrator
- [AWS](https://aws.amazon.com/): Compute and storage
- [MongoDB](https://www.mongodb.com/): NoSQL database
- [Qdrant](https://qdrant.tech/): Vector database

In the [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/), Chapter 2 will walk you through each tool, and in Chapters 10 and 11, you will have step-by-step guides on how to set everything you need.  

# Install

## Install Python 3.11 using pyenv (Optional)

If you have a different global Python version than Python 3.11, you can use pyenv to install Python 3.11 at the project level. Verify your Python version with:
```shell
python --version
```

First, verify that you have pyenv installed:
```shell
pyenv --version
# Output: pyenv 2.3.36
```

Install Python 3.11: 
```shell
pyenv install 3.11
```

From the root of your repository, run the following to verify that everything works fine:
```shell
pyenv versions
# Output: 
# system
# * 3.11.8 (set by <path/to/repo>/LLM-Engineers-Handbook/.python-version)
```

Because we defined a `.python-version` file within the repository, pyenv will know to pick up the version from that file and use it locally whenever you are working within that folder. To double-check that, run the following command while you are in the repository:
```shell
python --version
# Output: Python 3.11.8
```

If you move out of this repository, both `pyenv versions` and `python --version`, might output different Python versions.

## Install project dependences

The first step is to verify that you have Poetry installed:
```shell
poetry --version
# Output: Poetry (version 1.8.3)
```

Use Poetry to install all the project's requirements to run it locally. Thus, we don't need to install any AWS dependencies. Also, we install Poe the Poet as a Poetry plugin to manage our CLI commands and pre-commit to verify our code before committing changes to git:
```shell
poetry install --without aws
poetry self add 'poethepoet[poetry_plugin]==0.29.0'
pre-commit install
```

We run all the scripts using [Poe the Poet](https://poethepoet.natn.io/index.html). You don't have to do anything else but install Poe the Poet as a Poetry plugin, as described above: `poetry self add 'poethepoet[poetry_plugin]'`

To activate the environment created by Poetry, run:
```shell
poetry shell
```

## Set up .env settings file (for local development)

After you have installed all the dependencies, you must create and fill a `.env` file with your credentials to properly interact with other services and run the project.

First, copy our example by running the following:
```shell
cp .env.example .env # The file must be at your repository's root!
```

Now, let's understand how to fill in all the variables inside the `.env` file to get you started.

We will begin by reviewing the mandatory settings we must complete when working locally or in the cloud.

### OpenAI

To authenticate to OpenAI's API, you must fill out the `OPENAI_API_KEY` env var with an authentication token.

→ Check out this [tutorial](https://platform.openai.com/docs/quickstart) to learn how to provide one from OpenAI.

### HuggingFace

To authenticate to HuggingFace, you must fill out the `HUGGINGFACE_ACCESS_TOKEN` env var with an authentication token.

→ Check out this [tutorial](https://huggingface.co/docs/hub/en/security-tokens) to learn how to provide one from HuggingFace.

### Comet ML

Comet ML is required only during training.

To authenticate to Comet ML, you must fill out the `COMET_API_KEY` and `COMET_WORKSPACE` env vars with an authentication token and workspace name.

→ Check out this [tutorial](https://www.comet.com/docs/v2/api-and-sdk/rest-api/overview/) to learn how to fill the Comet ML variables from above.

### Opik

> Soon


## Set up .env settings file (for deployment)

when deploying the project to the cloud, we must set additional settings for Mongo, Qdrant, and AWS.

If you are just working localy, the default values of these env vars will work out-of-the-box.

We will just highlight what has to be configured, as in **Chapter 11** of the [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/) we provide step-by-step details on how to deploy the whole system to the cloud.

### MongoDB

We must change the `DATABASE_HOST` env var with the URL pointing to the cloud MongoDB cluster.

### Qdrant

Change `USE_QDRANT_CLOUD` to `True` and `QDRANT_CLOUD_URL` with the URL and `QDRANT_APIKEY` with the API KEY of your cloud Qdrant cluster.

To work with Qdrant cloud, the env vars will look like this:
```env
USE_QDRANT_CLOUD=true
QDRANT_CLOUD_URL="<your_qdrant_cloud_url>"
QDRANT_APIKEY="<your_qdrant_api_key>"
```

### AWS



> [!IMPORTANT]
> Find more configuration options in the [settings.py](https://github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/settings.py) file. Every variable from the `Settings` class can be configured through the `.env` file. 


# Run the project locally 

## Local infrastructure

> [!WARNING]
> You need Docker installed (v27.1.1 or higher)


Start:
```shell
poetry poe local-infrastructure-up
```

Stop:
```shell
poetry poe local-infrastructure-down
```

> [!WARNING]  
> When running on MacOS, before starting the server, export the following environment variable:
> `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
> Otherwise, the connection between the local server and pipeline will break. 🔗 More details in [this issue](https://github.com/zenml-io/zenml/issues/2369).

### ZenML is now accessible at:

Web UI: [localhost:8237](localhost:8237)

Default credentials:
    - `username`: default
    - `password`: 

→🔗 [More on ZenML](https://docs.zenml.io/)

### Qdrant is now accessible at:

REST API: [localhost:6333](localhost:6333)
Web UI: [localhost:6333/dashboard](localhost:6333/dashboard)
GRPC API: [localhost:6334](localhost:6334)

→🔗 [More on Qdrant](https://qdrant.tech/documentation/quick-start/)

### MongoDB is now accessible at:

database URI: `mongodb://decodingml:decodingml@127.0.0.1:27017`
database name: `twin`


### AWS Infrastructure

We will fill this section in the future. So far it is available only in the 11th Chapter of the book.


## Run Pipelines

All the pipelines will be orchestrated behind the scenes by ZenML.

To see the pipelines running and their results:
- go to your ZenML dashboard
- go to the `Pipelines` section
- click on a specific pipeline (e.g., `feature_engineering`)
- click on a specific run (e.g., `feature_engineering_run_2024_06_20_18_40_24`)
- click on a specific step or artifact to find more details about the run

**But first, let's understand how we can run all our ML pipelines** ↓

### Data pipelines

Run the data collection ETL:
```shell
poetry poe run-digital-data-etl
```

> [!WARNING]
> You must have Chrome installed on your system for the LinkedIn and Medium crawlers to work (which use Selenium under the hood). Based on your Chrome version, the Chromedriver will be automatically installed to enable Selenium support. Note that you can run everything using our Docker image if you don't want to install Chrome. For example, to run all the pipelines combined you can run `poetry poe run-docker-end-to-end-data-pipeline`. Note that the command can be tweaked to support any other pipeline.

> [!IMPORTANT]
> To add additional links to collect from, go to `configs_digital_data_etl_[your_name].yaml` and add them to the `links` field. Also, you can create a completely new file and specify it at run time, like this: `python -m llm_engineering.interfaces.orchestrator.run --run-etl --etl-config-filename configs_digital_data_etl_[your_name].yaml`

Run the feature engineering pipeline:
```shell
poetry poe run-feature-engineering-pipeline
```

Run the dataset generation pipeline:
```shell
poetry poe run-generate-instruct-datasets-pipeline
```

Run all of the above compressed into a single pipeline:
```shell
poetry poe run-end-to-end-data-pipeline
```


### Utility pipelines

Export ZenML artifacts to JSON:
```shell
poetry poe run-export-artifact-to-json-pipeline
```

### Training pipelines

```shell
poetry poe run-training-pipeline
```

### Linting & Formatting (QA)

Check and fix your linting issues:
```shell
poetry poe lint-check
poetry poe lint-fix
```

Check and fix your formatting issues:
```shell
poetry poe format-check
poetry poe format-fix
```
