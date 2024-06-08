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


## Run

> [!WARNING]  
> When running on MacOS, before starting the server export the following environment variable:
> `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
> Otherwise, the connection between the local server and pipeline will break. 🔗 More details in [this issue](https://github.com/zenml-io/zenml/issues/2369).