# LLM-Engineering

## Dependencies

- Python 3.11
- Poetry 1.8.3
- Docker 26.0.0

## Install

```shell
poetry install
poetry self add 'poethepoet[poetry_plugin]'
```

### How to install drivers for Selenium

... Explain based on docs below

* https://www.selenium.dev/documentation/webdriver/troubleshooting/errors/driver_location/
* https://www.selenium.dev/documentation/webdriver/troubleshooting/errors/driver_location/
* https://googlechromelabs.github.io/chrome-for-testing/#stable

> [!WARNING]
> For MacOS users, after downloading the driver run the following command to give permissions for the driver to be accesible: `xattr -d com.apple.quarantine /path/to/your/driver/chromedriver`


## Run Locally 

### ZenML Server

Start the local ZenML server:
```shell
poetry poe start-local-zenml-server
```

Stop the local ZenML server:
```shell
poetry poe stop-local-infrastructure
```

> [!WARNING]  
> When running on MacOS, before starting the server export the following environment variable:
> `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
> Otherwise, the connection between the local server and pipeline will break. 🔗 More details in [this issue](https://github.com/zenml-io/zenml/issues/2369).

### Local Infrastructure

Start the local infrastructure using Docker:
```shell
poetry poe start-local-infrastructure
```

Stop the local infrastructure
```shell
poetry poe stop-local-infrastructure
```

#### Qdrant

Under the default configuration all data will be stored in the ./qdrant_storage directory. This will also be the only directory that both the Container and the host machine can both see.

**Qdrant is now accessible:**

REST API: localhost:6333
Web UI: localhost:6333/dashboard
GRPC API: localhost:6334

**NOTE:** [More on Qdrant](https://qdrant.tech/documentation/quick-start/)