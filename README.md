<div align="center">
  <h1>👷 LLM Engineer's Handbook</h1>
  <p class="tagline">Official repository of the <a href="https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/">LLM Engineer's Handbook</a> by <a href="https://github.com/iusztinpaul">Paul Iusztin</a> and <a href="https://github.com/mlabonne">Maxime Labonne</a></p>
</div>
</br>

<p align="center">
  <a href="https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/">
    <img src="images/cover_plus.png" alt="Book cover">
  </a>
</p>

<p align="center">
  Find the book on <a href="https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/">Amazon</a> or <a href="https://www.packtpub.com/en-us/product/llm-engineers-handbook-9781836200062">Packt</a>
</p>

## 🌟 Features

The goal of this book is to create your own end-to-end LLM-based system using best practices:

- 📝 Data collection & generation
- 🔄 LLM training pipeline
- 📊 Simple RAG system
- 🚀 Production-ready AWS deployment
- 🔍 Comprehensive monitoring
- 🧪 Testing and evaluation framework

You can download and use the final trained model on [Hugging Face](https://huggingface.co/mlabonne/TwinLlama-3.1-8B-DPO).

> [!IMPORTANT]
> The code in this GitHub repository is actively maintained and may contain updates not reflected in the book. **Always refer to this repository for the latest version of the code.**

## 🔗 Dependencies

### Local dependencies

To install and run the project locally, you need the following dependencies.

| Tool | Version | Purpose | Installation Link |
|------|---------|---------|------------------|
| pyenv | ≥2.3.36 | Multiple Python versions (optional) | [Install Guide](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) |
| Python | 3.11 | Runtime environment | [Download](https://www.python.org/downloads/) |
| Poetry | >= 1.8.3 and < 2.0 | Package management | [Install Guide](https://python-poetry.org/docs/#installation) |
| Docker | ≥27.1.1 | Containerization | [Install Guide](https://docs.docker.com/engine/install/) |
| AWS CLI | ≥2.15.42 | Cloud management | [Install Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| Git | ≥2.44.0 | Version control | [Download](https://git-scm.com/downloads) |

### Cloud services

The code also uses and depends on the following cloud services. For now, you don't have to do anything. We will guide you in the installation and deployment sections on how to use them:

| Service | Purpose |
|---------|---------|
| [HuggingFace](https://huggingface.com/) | Model registry |
| [Comet ML](https://www.comet.com/site/products/opik/?utm_source=llm_handbook&utm_medium=github&utm_campaign=opik) | Experiment tracker |
| [Opik](https://www.comet.com/site/products/opik/?utm_source=llm_handbook&utm_medium=github&utm_campaign=opik) | Prompt monitoring |
| [ZenML](https://www.zenml.io/) | Orchestrator and artifacts layer |
| [AWS](https://aws.amazon.com/) | Compute and storage |
| [MongoDB](https://www.mongodb.com/) | NoSQL database |
| [Qdrant](https://qdrant.tech/) | Vector database |
| [GitHub Actions](https://github.com/features/actions) | CI/CD pipeline |

In the [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/), Chapter 2 will walk you through each tool. Chapters 10 and 11 provide step-by-step guides on how to set up everything you need.

## 🗂️ Project Structure

Here is the directory overview:

```bash
.
├── code_snippets/       # Standalone example code
├── configs/             # Pipeline configuration files
├── llm_engineering/     # Core project package
│   ├── application/    
│   ├── domain/         
│   ├── infrastructure/ 
│   ├── model/         
├── pipelines/           # ML pipeline definitions
├── steps/               # Pipeline components
├── tests/               # Test examples
├── tools/               # Utility scripts
│   ├── run.py
│   ├── ml_service.py
│   ├── rag.py
│   ├── data_warehouse.py
```

`llm_engineering/`  is the main Python package implementing LLM and RAG functionality. It follows Domain-Driven Design (DDD) principles:

- `domain/`: Core business entities and structures
- `application/`: Business logic, crawlers, and RAG implementation
- `model/`: LLM training and inference
- `infrastructure/`: External service integrations (AWS, Qdrant, MongoDB, FastAPI)

The code logic and imports flow as follows: `infrastructure` → `model` → `application` → `domain`

`pipelines/`: Contains the ZenML ML pipelines, which serve as the entry point for all the ML pipelines. Coordinates the data processing and model training stages of the ML lifecycle.

`steps/`: Contains individual ZenML steps, which are reusable components for building and customizing ZenML pipelines. Steps perform specific tasks (e.g., data loading, preprocessing) and can be combined within the ML pipelines.

`tests/`: Covers a few sample tests used as examples within the CI pipeline.

`tools/`: Utility scripts used to call the ZenML pipelines and inference code:
- `run.py`: Entry point script to run ZenML pipelines.
- `ml_service.py`: Starts the REST API inference server.
- `rag.py`: Demonstrates usage of the RAG retrieval module.
- `data_warehouse.py`: Used to export or import data from the MongoDB data warehouse through JSON files.

`configs/`: ZenML YAML configuration files to control the execution of pipelines and steps.

`code_snippets/`: Independent code examples that can be executed independently.

## 💻 Installation

> [!NOTE]
> If you are experiencing issues while installing and running the repository, consider checking the [Issues](https://github.com/PacktPublishing/LLM-Engineers-Handbook/issues) GitHub section for other people who solved similar problems or directly asking us for help.

### 1. Clone the Repository

Start by cloning the repository and navigating to the project directory:

```bash
git clone https://github.com/PacktPublishing/LLM-Engineers-Handbook.git
cd LLM-Engineers-Handbook 
```

Next, we have to prepare your Python environment and its adjacent dependencies. 

### 2. Set Up Python Environment

The project requires Python 3.11. You can either use your global Python installation or set up a project-specific version using pyenv.

#### Option A: Using Global Python (if version 3.11 is installed)

Verify your Python version:

```bash
python --version  # Should show Python 3.11.x
```

#### Option B: Using pyenv (recommended)

1. Verify pyenv installation:

```bash
pyenv --version   # Should show pyenv 2.3.36 or later
```

2. Install Python 3.11.8:

```bash
pyenv install 3.11.8
```

3. Verify the installation:

```bash
python --version  # Should show Python 3.11.8
```

4. Confirm Python version in the project directory:

```bash
python --version
# Output: Python 3.11.8
```

> [!NOTE]  
> The project includes a `.python-version` file that automatically sets the correct Python version when you're in the project directory.

### 3. Install Dependencies

The project uses Poetry for dependency management.

1. Verify Poetry installation:

```bash
poetry --version  # Should show Poetry version 1.8.3 or later
```

2. Set up the project environment and install dependencies:

```bash
poetry env use 3.11
poetry install --without aws
poetry run pre-commit install
```

This will:

- Configure Poetry to use Python 3.11
- Install project dependencies (excluding AWS-specific packages)
- Set up pre-commit hooks for code verification

### 4. Activate the Environment

As our task manager, we run all the scripts using [Poe the Poet](https://poethepoet.natn.io/index.html).

1. Start a Poetry shell:

```bash
poetry shell
```

2. Run project commands using Poe the Poet:

```bash
poetry poe ...
```

<details>
<summary>🔧 Troubleshooting Poe the Poet Installation</summary>

### Alternative Command Execution

If you're experiencing issues with `poethepoet`, you can still run the project commands directly through Poetry. Here's how:

1. Look up the command definition in `pyproject.toml`
2. Use `poetry run` with the underlying command

#### Example:
Instead of:
```bash
poetry poe local-infrastructure-up
```
Use the direct command from pyproject.toml:
```bash
poetry run <actual-command-from-pyproject-toml>
```
Note: All project commands are defined in the [tool.poe.tasks] section of pyproject.toml
</details>

Now, let's configure our local project with all the necessary credentials and tokens to run the code locally.

### 5. Local Development Setup

After you have installed all the dependencies, you must create and fill a `.env` file with your credentials to appropriately interact with other services and run the project. Setting your sensitive credentials in a `.env` file is a good security practice, as this file won't be committed to GitHub or shared with anyone else. 

1. First, copy our example by running the following:

```bash
cp .env.example .env # The file must be at your repository's root!
```

2. Now, let's understand how to fill in all the essential variables within the `.env` file to get you started. The following are the mandatory settings we must complete when working locally:

#### OpenAI

To authenticate to OpenAI's API, you must fill out the `OPENAI_API_KEY` env var with an authentication token.

```env
OPENAI_API_KEY=your_api_key_here
```

→ Check out this [tutorial](https://platform.openai.com/docs/quickstart) to learn how to provide one from OpenAI.

#### Hugging Face

To authenticate to Hugging Face, you must fill out the `HUGGINGFACE_ACCESS_TOKEN` env var with an authentication token.

```env
HUGGINGFACE_ACCESS_TOKEN=your_token_here
```

→ Check out this [tutorial](https://huggingface.co/docs/hub/en/security-tokens) to learn how to provide one from Hugging Face.

#### Comet ML & Opik

To authenticate to Comet ML (required only during training) and Opik, you must fill out the `COMET_API_KEY` env var with your authentication token.

```env
COMET_API_KEY=your_api_key_here
```

→ Check out this [tutorial](https://www.comet.com/docs/opik/?utm_source=llm_handbook&utm_medium=github&utm_campaign=opik) to learn how to get started with Opik. You can also access Opik's dashboard using 🔗[this link](https://www.comet.com/opik?utm_source=llm_handbook&utm_medium=github&utm_content=opik).

### 6. Deployment Setup

When deploying the project to the cloud, we must set additional settings for Mongo, Qdrant, and AWS. If you are just working locally, the default values of these env vars will work out of the box. Detailed deployment instructions are available in Chapter 11 of the [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

#### MongoDB

We must change the `DATABASE_HOST` env var with the URL pointing to your cloud MongoDB cluster.

```env
DATABASE_HOST=your_mongodb_url
```

→ Check out this [tutorial](https://www.mongodb.com/resources/products/fundamentals/mongodb-cluster-setup) to learn how to create and host a MongoDB cluster for free.

#### Qdrant

Change `USE_QDRANT_CLOUD` to `true`, `QDRANT_CLOUD_URL` with the URL point to your cloud Qdrant cluster, and `QDRANT_APIKEY` with its API key.

```env
USE_QDRANT_CLOUD=true
QDRANT_CLOUD_URL=your_qdrant_cloud_url
QDRANT_APIKEY=your_qdrant_api_key
```

→ Check out this [tutorial](https://qdrant.tech/documentation/cloud/create-cluster/) to learn how to create a Qdrant cluster for free

#### AWS

For your AWS set-up to work correctly, you need the AWS CLI installed on your local machine and properly configured with an admin user (or a user with enough permissions to create new SageMaker, ECR, and S3 resources; using an admin user will make everything more straightforward).

Chapter 2 provides step-by-step instructions on how to install the AWS CLI, create an admin user on AWS, and get an access key to set up the `AWS_ACCESS_KEY` and `AWS_SECRET_KEY` environment variables. If you already have an AWS admin user in place, you have to configure the following env vars in your `.env` file:

```bash
AWS_REGION=eu-central-1 # Change it with your AWS region.
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
```

AWS credentials are typically stored in `~/.aws/credentials`. You can view this file directly using `cat` or similar commands:

```bash
cat ~/.aws/credentials
```

> [!IMPORTANT]
> Additional configuration options are available in [settings.py](https://github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/settings.py). Any variable in the `Settings` class can be configured through the `.env` file. 

## 🏗️ Infrastructure

### Local infrastructure (for testing and development)

When running the project locally, we host a MongoDB and Qdrant database using Docker. Also, a testing ZenML server is made available through their Python package.

> [!WARNING]
> You need Docker installed (>= v27.1.1)

For ease of use, you can start the whole local development infrastructure with the following command:
```bash
poetry poe local-infrastructure-up
```

Also, you can stop the ZenML server and all the Docker containers using the following command:
```bash
poetry poe local-infrastructure-down
```

> [!WARNING]  
> When running on MacOS, before starting the server, export the following environment variable:
> `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
> Otherwise, the connection between the local server and pipeline will break. 🔗 More details in [this issue](https://github.com/zenml-io/zenml/issues/2369).
> This is done by default when using Poe the Poet.

Start the inference real-time RESTful API:
```bash
poetry poe run-inference-ml-service
```

> [!IMPORTANT]
> The LLM microservice, called by the RESTful API, will work only after deploying the LLM to AWS SageMaker.

#### ZenML

Dashboard URL: `localhost:8237`

Default credentials:
  - `username`: default
  - `password`: 

→ Find out more about using and setting up [ZenML](https://docs.zenml.io/).

#### Qdrant

REST API URL: `localhost:6333`

Dashboard URL: `localhost:6333/dashboard`

→ Find out more about using and setting up [Qdrant with Docker](https://qdrant.tech/documentation/quick-start/).

#### MongoDB

Database URI: `mongodb://llm_engineering:llm_engineering@127.0.0.1:27017`

Database name: `twin`

Default credentials:
  - `username`: llm_engineering
  - `password`: llm_engineering

→ Find out more about using and setting up [MongoDB with Docker](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-community-with-docker).

You can search your MongoDB collections using your **IDEs MongoDB plugin** (which you have to install separately), where you have to use the database URI to connect to the MongoDB database hosted within the Docker container: `mongodb://llm_engineering:llm_engineering@127.0.0.1:27017`

> [!IMPORTANT]
> Everything related to training or running the LLMs (e.g., training, evaluation, inference) can only be run if you set up AWS SageMaker, as explained in the next section on cloud infrastructure.

### Cloud infrastructure (for production)

Here we will quickly present how to deploy the project to AWS and other serverless services. We won't go into the details (as everything is presented in the book) but only point out the main steps you have to go through.

First, reinstall your Python dependencies with the AWS group:
```bash
poetry install --with aws
```

#### AWS SageMaker

> [!NOTE]
> Chapter 10 provides step-by-step instructions in the section "Implementing the LLM microservice using AWS SageMaker".

By this point, we expect you to have AWS CLI installed and your AWS CLI and project's env vars (within the `.env` file) properly configured with an AWS admin user.

To ensure best practices, we must create a new AWS user restricted to creating and deleting only resources related to AWS SageMaker. Create it by running:
```bash
poetry poe create-sagemaker-role
```
It will create a `sagemaker_user_credentials.json` file at the root of your repository with your new `AWS_ACCESS_KEY` and `AWS_SECRET_KEY` values. **But before replacing your new AWS credentials, also run the following command to create the execution role (to create it using your admin credentials).**

To create the IAM execution role used by AWS SageMaker to access other AWS resources on our behalf, run the following:
```bash
poetry poe create-sagemaker-execution-role
```
It will create a `sagemaker_execution_role.json` file at the root of your repository with your new `AWS_ARN_ROLE` value. Add it to your `.env` file. 

Once you've updated the `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, and `AWS_ARN_ROLE` values in your `.env` file, you can use AWS SageMaker. **Note that this step is crucial to complete the AWS setup.**

#### Training

We start the training pipeline through ZenML by running the following:
```bash
poetry poe run-training-pipeline
```
This will start the training code using the configs from `configs/training.yaml` directly in SageMaker. You can visualize the results in Comet ML's dashboard.

We start the evaluation pipeline through ZenML by running the following:
```bash
poetry poe run-evaluation-pipeline
```
This will start the evaluation code using the configs from `configs/evaluating.yaml` directly in SageMaker. You can visualize the results in `*-results` datasets saved to your Hugging Face profile.

#### Inference

To create an AWS SageMaker Inference Endpoint, run:
```bash
poetry poe deploy-inference-endpoint
```
To test it out, run:
```bash
poetry poe test-sagemaker-endpoint
```
To delete it, run:
```bash
poetry poe delete-inference-endpoint
```

#### AWS: ML pipelines, artifacts, and containers

The ML pipelines, artifacts, and containers are deployed to AWS by leveraging ZenML's deployment features. Thus, you must create an account with ZenML Cloud and follow their guide on deploying a ZenML stack to AWS. Otherwise, we provide step-by-step instructions in **Chapter 11**, section **Deploying the LLM Twin's pipelines to the cloud** on what you must do.  

#### Qdrant & MongoDB

We leverage Qdrant's and MongoDB's serverless options when deploying the project. Thus, you can either follow [Qdrant's](https://qdrant.tech/documentation/cloud/create-cluster/) and [MongoDB's](https://www.mongodb.com/resources/products/fundamentals/mongodb-cluster-setup) tutorials on how to create a freemium cluster for each or go through **Chapter 11**, section **Deploying the LLM Twin's pipelines to the cloud** and follow our step-by-step instructions.

#### GitHub Actions

We use GitHub Actions to implement our CI/CD pipelines. To implement your own, you have to fork our repository and set the following env vars as Actions secrets in your forked repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ECR_NAME`
- `AWS_REGION`

Also, we provide instructions on how to set everything up in **Chapter 11**, section **Adding LLMOps to the LLM Twin**.

#### Comet ML & Opik

You can visualize the results on their self-hosted dashboards if you create a Comet account and correctly set the `COMET_API_KEY` env var. As Opik is powered by Comet, you don't have to set up anything else along Comet:
- [Comet ML (for experiment tracking)](https://www.comet.com/?utm_source=llm_handbook&utm_medium=github&utm_campaign=opik)
- [Opik (for prompt monitoring)](https://www.comet.com/opik?utm_source=llm_handbook&utm_medium=github&utm_campaign=opik)

### 💰 Running the Project Costs

We will mostly stick to free tiers for all the services except for AWS and OpenAI's API, which are both pay-as-you-go services. The cost of running the project once, with our default values, will be roughly ~$25 (most of it comes from using AWS SageMaker for training and inference).

## ⚡ Pipelines

All the ML pipelines will be orchestrated behind the scenes by [ZenML](https://www.zenml.io/). A few exceptions exist when running utility scrips, such as exporting or importing from the data warehouse.

The ZenML pipelines are the entry point for most processes throughout this project. They are under the `pipelines/` folder. Thus, when you want to understand or debug a workflow, starting with the ZenML pipeline is the best approach.

To see the pipelines running and their results:
- go to your ZenML dashboard
- go to the `Pipelines` section
- click on a specific pipeline (e.g., `feature_engineering`)
- click on a specific run (e.g., `feature_engineering_run_2024_06_20_18_40_24`)
- click on a specific step or artifact of the DAG to find more details about it

Now, let's explore all the pipelines you can run. From data collection to training, we will present them in their natural order to go through the LLM project end-to-end.

### Data pipelines

Run the data collection ETL:
```bash
poetry poe run-digital-data-etl
```

> [!WARNING]
> You must have Chrome (or another Chromium-based browser) installed on your system for LinkedIn and Medium crawlers to work (which use Selenium under the hood). Based on your Chrome version, the Chromedriver will be automatically installed to enable Selenium support. Another option is to run everything using our Docker image if you don't want to install Chrome. For example, to run all the pipelines combined you can run `poetry poe run-docker-end-to-end-data-pipeline`. Note that the command can be tweaked to support any other pipeline.
>
> If, for any other reason, you don't have a Chromium-based browser installed and don't want to use Docker, you have two other options to bypass this Selenium issue:
> - Comment out all the code related to Selenium, Chrome and all the links that use Selenium to crawl them (e.g., Medium), such as the `chromedriver_autoinstaller.install()` command from [application.crawlers.base](https://github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/application/crawlers/base.py) and other static calls that check for Chrome drivers and Selenium.
> - Install Google Chrome using your CLI in environments such as GitHub Codespaces or other cloud VMs using the same command as in our [Docker file](https://github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/Dockerfile#L10).

To add additional links to collect from, go to `configs/digital_data_etl_[author_name].yaml` and add them to the `links` field. Also, you can create a completely new file and specify it at run time, like this: `python -m llm_engineering.interfaces.orchestrator.run --run-etl --etl-config-filename configs/digital_data_etl_[your_name].yaml`

Run the feature engineering pipeline:
```bash
poetry poe run-feature-engineering-pipeline
```

Generate the instruct dataset:
```bash
poetry poe run-generate-instruct-datasets-pipeline
```

Generate the preference dataset:
```bash
poetry poe run-generate-preference-datasets-pipeline
```

Run all of the above compressed into a single pipeline:
```bash
poetry poe run-end-to-end-data-pipeline
```

### Utility pipelines

Export the data from the data warehouse to JSON files:
```bash
poetry poe run-export-data-warehouse-to-json
```

Import data to the data warehouse from JSON files (by default, it imports the data from the `data/data_warehouse_raw_data` directory):
```bash
poetry poe run-import-data-warehouse-from-json
```

Export ZenML artifacts to JSON:
```bash
poetry poe run-export-artifact-to-json-pipeline
```

This will export the following ZenML artifacts to the `output` folder as JSON files (it will take their latest version):
- cleaned_documents.json
- instruct_datasets.json
- preference_datasets.json
- raw_documents.json

You can configure what artifacts to export by tweaking the `configs/export_artifact_to_json.yaml` configuration file.

### Training pipelines

Run the training pipeline:
```bash
poetry poe run-training-pipeline
```

Run the evaluation pipeline:
```bash
poetry poe run-evaluation-pipeline
```

> [!WARNING]
> For this to work, make sure you properly configured AWS SageMaker as described in [Set up cloud infrastructure (for production)](#set-up-cloud-infrastructure-for-production).

### Inference pipelines

Call the RAG retrieval module with a test query:
```bash
poetry poe call-rag-retrieval-module
```

Start the inference real-time RESTful API:
```bash
poetry poe run-inference-ml-service
```

Call the inference real-time RESTful API with a test query:
```bash
poetry poe call-inference-ml-service
```

Remember that you can monitor the prompt traces on [Opik](https://www.comet.com/opik).

> [!WARNING]
> For the inference service to work, you must have the LLM microservice deployed to AWS SageMaker, as explained in the setup cloud infrastructure section.

### Linting & formatting (QA)

Check or fix your linting issues:
```bash
poetry poe lint-check
poetry poe lint-fix
```

Check or fix your formatting issues:
```bash
poetry poe format-check
poetry poe format-fix
```

Check the code for leaked credentials:
```bash
poetry poe gitleaks-check
```

### Tests

Run all the tests using the following command:
```bash
poetry poe test
```

## 🏃 Run project

Based on the setup and usage steps described above, assuming the local and cloud infrastructure works and the `.env` is filled as expected, follow the next steps to run the LLM system end-to-end:

### Data

1. Collect data: `poetry poe run-digital-data-etl`

2. Compute features: `poetry poe run-feature-engineering-pipeline`

3. Compute instruct dataset: `poetry poe run-generate-instruct-datasets-pipeline`

4. Compute preference alignment dataset: `poetry poe run-generate-preference-datasets-pipeline`

### Training

> [!IMPORTANT]
> From now on, for these steps to work, you need to properly set up AWS SageMaker, such as running `poetry install --with aws` and filling in the AWS-related environment variables and configs.

5. SFT fine-tuning Llamma 3.1: `poetry poe run-training-pipeline`

6. For DPO, go to `configs/training.yaml`, change `finetuning_type` to `dpo`, and run `poetry poe run-training-pipeline` again

7. Evaluate fine-tuned models: `poetry poe run-evaluation-pipeline`

### Inference

> [!IMPORTANT]
> From now on, for these steps to work, you need to properly set up AWS SageMaker, such as running `poetry install --with aws` and filling in the AWS-related environment variables and configs.

8. Call only the RAG retrieval module: `poetry poe call-rag-retrieval-module`

9. Deploy the LLM Twin microservice to SageMaker: `poetry poe deploy-inference-endpoint`

10. Test the LLM Twin microservice: `poetry poe test-sagemaker-endpoint`

11. Start end-to-end RAG server: `poetry poe run-inference-ml-service`

12. Test RAG server: `poetry poe call-inference-ml-service`

## 📄 License

This course is an open-source project released under the MIT license. Thus, as long you distribute our LICENSE and acknowledge our work, you can safely clone or fork this project and use it as a source of inspiration for whatever you want (e.g., university projects, college degree projects, personal projects, etc.).
