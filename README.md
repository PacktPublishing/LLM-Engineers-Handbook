# LLM-Engineering

## Dependencies

- Python 3.11
- Poetry 1.8.3
- Docker 26.0.0

## Install

```shell
poetry install --without aws
poetry self add 'poethepoet[poetry_plugin]'
pre-commit install
```

We run all the scripts using [Poe the Poet](https://poethepoet.natn.io/index.html). You don't have to do anything else but install it as a Poetry plugin.

### Configure sensitive information
After you have installed all the dependencies, you must create a `.env` file with sensitive credentials to run the project.

First, copy our example by running the following:
```shell
cp .env.example .env # The file has to be at the root of your repository!
```

Now, let's understand how to fill in all the variables inside the `.env` file to get you started.

### OpenAI

To authenticate to OpenAI, you must fill out the `OPENAI_API_KEY` env var with an authentication token.

→ Check out this [tutorial](https://platform.openai.com/docs/quickstart) to learn how to provide one from OpenAI.

### HuggingFace

To authenticate to HuggingFace, you must fill out the `HUGGINGFACE_ACCESS_TOKEN` env var with an authentication token.

→ Check out this [tutorial](https://huggingface.co/docs/hub/en/security-tokens) to learn how to provide one from HuggingFace.


### LinkedIn Crawling [Optional]
This step is optional. You can finish the project without this step.

But in case you want to enable LinkedIn crawling, you have to fill in your username and password:
```shell
LINKEDIN_USERNAME = "str"
LINKEDIN_PASSWORD = "str"
```

For this to work, you also have to:
- disable 2FA
- disable suspicious activity

We also recommend to:
- create a dummy profile for crawling
- crawl only your data


> [!IMPORTANT]
> Find more configuration options in the [settings.py](https://github.com/PacktPublishing/LLM-Engineering/blob/main/llm_engineering/settings.py) file. Every variable from the `Settings` class can be configured through the `.env` file. 


## Run Locally 

### Local Infrastructure

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

#### ZenML is now accessible at:

Web UI: [localhost:8237](localhost:8237)

Default credentials:
    - `username`: default
    - `password`: 

→🔗 [More on ZenML](https://docs.zenml.io/)

#### Qdrant is now accessible at:

REST API: [localhost:6333](localhost:6333)
Web UI: [localhost:6333/dashboard](localhost:6333/dashboard)
GRPC API: [localhost:6334](localhost:6334)

→🔗 [More on Qdrant](https://qdrant.tech/documentation/quick-start/)

#### MongoDB is now accessible at:

database URI: `mongodb://decodingml:decodingml@127.0.0.1:27017`
database name: `twin`


### AWS Infrastructure

We will fill this section in the future. So far it is available only in the 11th Chapter of the book.


### Run Pipelines

All the pipelines will be orchestrated behind the scenes by ZenML.

To see the pipelines running and their results:
- go to your ZenML dashboard
- go to the `Pipelines` section
- click on a specific pipeline (e.g., `feature_engineering`)
- click on a specific run (e.g., `feature_engineering_run_2024_06_20_18_40_24`)
- click on a specific step or artifact to find more details about the run

**But first, let's understand how we can run all our ML pipelines** ↓

#### Data pipelines

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


#### Utility pipelines

Export ZenML artifacts to JSON:
```shell
poetry poe run-export-artifact-to-json-pipeline
```

#### Training pipelines

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
