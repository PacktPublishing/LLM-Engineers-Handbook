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
After you have installed all the dependencies, you have to fill a `.env` file.

First, copy our example:
```shell
cp .env.example .env
```

Now, let's understand how to fill it.

### Selenium Drivers

You must download the Selenium Chrome driver to run the data collection pipeline. To proceed, use the links below:
* https://www.selenium.dev/documentation/webdriver/troubleshooting/errors/driver_location/
* https://googlechromelabs.github.io/chrome-for-testing/#stable

> [!WARNING]
> For MacOS users, after downloading the driver, run the following command to give permissions for the driver to be accessible: `xattr -d com.apple.quarantine /path/to/your/driver/chromedriver`

The last step is to add the path to the downloaded driver in your `.env` file:
```
SELENIUM_BROWSER_DRIVER_PATH = "str"
```

### LinkedIn Crawling

For crawling LinkedIn, you have to fill in your username and password:
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

### OpenAI

You also have to configure the standard `OPENAI_API_KEY`.


> [!IMPORTANT]
> Find more configuration options in the [settings.py](https://github.com/PacktPublishing/LLM-Engineering/blob/main/llm_engineering/settings.py) file.


## Run Locally 

### Local Infrastructure

> [!WARNING]
> You need Docker installed.

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

Web UI: localhost:8237

Default credentials:
    - username: default
    - password: 

🔗 [More on ZenML](https://docs.zenml.io/)

#### Qdrant is now accessible at:

REST API: localhost:6333
Web UI: localhost:6333/dashboard
GRPC API: localhost:6334

🔗 [More on Qdrant](https://qdrant.tech/documentation/quick-start/)

#### MongoDB is now accessible at:

database URI: mongodb://decodingml:decodingml@127.0.0.1:27017
database name: twin

### Run Pipelines

All the pipelines will be orchestrated behind the scenes by ZenML.

To see the pipelines running and their results & metadata:
- go to your ZenML dashboard
- go to the `Pipelines` section
- click on a specific pipeline (e.g., `feature_engineering`)
- click on a specific run (e.g., `feature_engineering_run_2024_06_20_18_40_24`)
- click on a specific step or artifact to find more details about the run

#### General Utilities

Export ZenML artifacts to JSON:
```shell
poetry poe run-export-artifact-to-json-pipeline
```

#### Data Preprocessing

Run the data collection ETL:
```shell
poetry poe run-digital-data-etl
```

> [!IMPORTANT]
> To add additional links to collect, go to `configs_digital_data_etl_[your_name].yaml` and add them to the `links` field. Also, you can create a completely new file and specify it at run time, like this: `python -m llm_engineering.interfaces.orchestrator.run --run-etl --etl-config-filename configs_digital_data_etl_[your_name].yaml`

Run the feature engineering pipeline:
```shell
poetry poe run-feature-engineering-pipeline
```

Run the dataset generation pipeline:
```shell
poetry poe run-generate-instruct-datasets-pipeline
```

Run all of the above:
```shell
poetry poe run-preprocessing-pipeline
```

#### Training

```shell
poetry poe run-training-pipeline
```

#### QA

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