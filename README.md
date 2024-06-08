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

## Run

> Warning
> When running on MacOS, before starting the server export the following environment variable:
> `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
> Otherwise, the connection between the local server and pipeline will break. 🔗 More details in [this issue](https://github.com/zenml-io/zenml/issues/2369).