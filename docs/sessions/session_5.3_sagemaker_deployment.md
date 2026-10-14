# Session 5.3: AWS SageMaker Deployment

## 🎯 Learning Objectives

By the end of this session, you will:
- Explain the three deployment types and when to use each
- Justify the LLM Twin's real-time microservice choice
- Create the IAM roles SageMaker needs and understand the trust relationship
- Launch a training job on a managed GPU instance
- Deploy a Hugging Face LLM to a real-time endpoint
- Configure inference via environment variables (quantization, token limits)
- Add target-tracking autoscaling and clean up the endpoint
- Read the `ResourceManager` / `DeploymentService` / `DeploymentStrategy` design
- Diagnose common deployment failures without burning money

> ⚠️ **Cost warning**: `ml.g5.2xlarge` and larger instances bill by the second.
> An `ml.g5.2xlarge` costs roughly \$1.0-1.5 per hour on demand. Do not leave an
> endpoint running. Always run the delete script when finished.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AWS account                                   │
│                                                                        │
│  IAM roles (create_execution_role.py / create_sagemaker_role.py)      │
│        │                                                               │
│        ▼                                                               │
│  SageMaker Training Job (model/finetuning/sagemaker.py)               │
│        │  HuggingFace estimator, entry_point=finetune.py             │
│        │  instance ml.g5.2xlarge, role=AWS_ARN_ROLE                  │
│        ▼                                                               │
│  Model on Hugging Face Hub  {workspace}/TwinLlama-3.1-8B-DPO         │
│        │                                                               │
│        ▼                                                               │
│  SageMaker Endpoint (deploy/huggingface/run.py)                       │
│        │  HuggingFaceModel(image=get_huggingface_llm_image_uri)      │
│        │  env: quantize=bitsandbytes, max tokens, SM_NUM_GPUS        │
│        ▼                                                               │
│  Autoscaling (autoscaling_sagemaker_endpoint.py)                      │
│        └─ target tracking on InvocationsPerCopy                       │
└──────────────────────────────────────────────────────────────────────┘
```

### The four requirements, then the choice

The book frames every serving decision against four requirements:

| Requirement | Question | LLM Twin answer |
|-------------|----------|-----------------|
| Throughput | Requests per second the system must sustain | Low to moderate, spiky |
| Latency | Time from request to answer | Must be low (chat, content creation) |
| Data | Input/output format, size, complexity | Text in, text out |
| Infrastructure | Hardware, network, cost | One GPU instance, scale to six |

Throughput is requests per second (RPS). Latency is the round-trip time for one
request: network I/O plus serialization plus inference. A lower latency usually
raises throughput, but batching breaks that rule: processing 20 batched requests
in 100 ms gives 100 ms latency and 200 RPS, while 60 requests in 200 ms gives
200 ms latency and 300 RPS. Higher latency can buy higher throughput.

### Three deployment types

```
Online real-time          Asynchronous              Offline batch
─────────────────         ──────────────            ──────────────
client ─HTTP─► server     client ─► queue ─► worker  storage ─► job ─► storage
client waits              client polls/pushed        client reads later
low latency               medium latency             high latency
high cost per idle GPU    cost efficient             cheapest
chatbots, RAG             summarization, keyword     reporting, analytics
                          extraction, deepfake
```

- **Online real-time inference**: a server reachable over HTTP (REST) or gRPC.
  The client waits synchronously. REST with JSON is more accessible but slower;
  gRPC with protobuf is faster but less flexible and is usually internal. The
  LLM Twin uses REST over SageMaker.
- **Asynchronous inference**: the request is queued and acknowledged; the client
  does not block. Good when jobs take minutes, or when spikes must be absorbed
  without scaling 10x. Costs more complexity and latency.
- **Offline batch transform**: the service pulls data from storage, processes a
  large batch, and writes results back. Highest throughput, cheapest, highest
  latency. Unsuitable for interactive chat.

**Decision for the LLM Twin**: an online real-time service with a strong latency
emphasis, because it is a content-creation chatbot.

### Monolithic versus microservices

```
Monolith                          Microservices
┌───────────────────────┐         ┌──────────┐     ┌──────────────┐
│ REST + RAG + LLM      │         │ REST API │────►│ LLM service  │
│ one machine, GPU+CPU  │         │ CPU only │     │ GPU only     │
└───────────────────────┘         └──────────┘     └──────────────┘
GPU idle during business logic    each scales independently
```

A monolith bundles the model and business logic in one deployable unit. It is
easy to start but couples GPU-bound inference with CPU/I/O-bound RAG, so the GPU
sits idle during retrieval and the CPU sits idle during generation. A
microservices split lets each side use the right machine and scale
independently: the business logic runs on a cheap CPU box, the LLM on a GPU box.

**Decision for the LLM Twin**: microservices. The FastAPI business service does
RAG retrieval and augmentation (CPU and network I/O); the SageMaker endpoint does
generation (GPU). A 30B upgrade would touch only the LLM service.

### Strategy Pattern for Deployment

```
DeploymentStrategy (domain/inference.py, ABC)
        ▲
        │ implements
SagemakerHuggingfaceStrategy  ──delegates──►  DeploymentService
   (public API)                                 (boto3 + HuggingFaceModel)
```

The strategy is swappable: a different cloud could implement `DeploymentStrategy`
without touching the deploy script.

### Inference flow at runtime

```
User ──HTTP──► FastAPI /rag
                  │
                  ├─ ContextRetriever.search (Qdrant + embeddings, CPU)
                  ├─ build prompt from query + context
                  ├─ HTTP ─► SageMaker endpoint (GPU, TGI)
                  └─ trace to prompt monitoring, then return answer
```

---

## 📁 Key Files Explained

### 1. `infrastructure/aws/roles/create_execution_role.py` - Execution Role

**Purpose**: The role SageMaker assumes to run jobs and read S3/ECR.

```python
# llm_engineering/infrastructure/aws/roles/create_execution_role.py
def create_sagemaker_execution_role(role_name: str):
    assert settings.AWS_REGION, "AWS_REGION is not set."
    assert settings.AWS_ACCESS_KEY, "AWS_ACCESS_KEY is not set."
    assert settings.AWS_SECRET_KEY, "AWS_SECRET_KEY is not set."

    iam = boto3.client(
        "iam",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )

    trust_relationship = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}
        ],
    }

    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_relationship),
            Description="Execution role for SageMaker",
        )

        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
            "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        ]
        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)

        logger.info(f"Role '{role_name}' created successfully.")
        logger.info(f"Role ARN: {role['Role']['Arn']}")
        return role["Role"]["Arn"]

    except iam.exceptions.EntityAlreadyExistsException:
        logger.warning(f"Role '{role_name}' already exists. Fetching its ARN...")
        role = iam.get_role(RoleName=role_name)
        return role["Role"]["Arn"]
```

The `__main__` block creates the role and writes its ARN to disk:

```python
if __name__ == "__main__":
    role_arn = create_sagemaker_execution_role("SageMakerExecutionRoleLLM")
    logger.info(role_arn)

    with Path("sagemaker_execution_role.json").open("w") as f:
        json.dump({"RoleArn": role_arn}, f)
```

**Key Concepts**:
- **Trust relationship** lets `sagemaker.amazonaws.com` assume the role
  (`sts:AssumeRole`). This is what allows SageMaker, not your laptop, to act with
  the role's permissions while the job runs.
- **Four managed policies**: SageMaker (control plane), S3 (training data and
  model output), CloudWatch Logs (job logs), and ECR (pull the Docker image).
- **Idempotent**: on `EntityAlreadyExistsException`, it fetches the existing ARN
  instead of failing, so re-running the command is safe.
- **The role name is hard-coded** to `SageMakerExecutionRoleLLM` in the
  `__main__` block.
- **The ARN is written to `sagemaker_execution_role.json`** and copied into
  `.env` as `AWS_ARN_ROLE`. The settings field is `AWS_ARN_ROLE`, not
  `ARN_ROLE`.

### IAM user versus IAM role: the flow

```
Step 1: create_sagemaker_role.py  (broad, admin credentials)
   creates IAM USER "sagemaker-deployer"
   attaches 5 policies, mints an access key
   writes sagemaker_user_credentials.json
        │
        ▼  you paste the new key/secret into .env
Step 2: create_execution_role.py  (narrow deployer credentials)
   creates IAM ROLE "SageMakerExecutionRoleLLM"
   trusts sagemaker.amazonaws.com
   writes sagemaker_execution_role.json
        │
        ▼  you paste RoleArn into .env as AWS_ARN_ROLE
Step 3: deploy / train
   SageMaker assumes AWS_ARN_ROLE at runtime
```

> ⚠️ **Fact**: `create_sagemaker_role.py` in this repo creates a full IAM **user**
> (`sagemaker-deployer`) with an **access key**, not a role, despite the file
> name. It grants `IAMFullAccess` and `AWSCloudFormationFullAccess` on top of
> SageMaker, CloudFormation, ECR, and S3 access. Prefer the execution role for
> deployments; the user exists so you can stop using admin credentials for
> day-to-day SageMaker work.

```python
# llm_engineering/infrastructure/aws/roles/create_sagemaker_role.py
def create_sagemaker_user(username: str):
    iam = boto3.client("iam", ...)
    iam.create_user(UserName=username)

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
    access_key = response["AccessKey"]
    return {"AccessKeyId": access_key["AccessKeyId"], "SecretAccessKey": access_key["SecretAccessKey"]}
```

The `__main__` block creates the user and writes credentials to
`sagemaker_user_credentials.json`.

### 2. `model/finetuning/sagemaker.py` - Training Job

**Purpose**: Submit `finetune.py` as a SageMaker training job.

```python
# llm_engineering/model/finetuning/sagemaker.py
finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def run_finetuning_on_sagemaker(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")
    if not finetuning_requirements_path.exists():
        raise FileNotFoundError(f"The file {finetuning_requirements_path} does not exist.")

    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]

    hyperparameters = {
        "finetuning_type": finetuning_type,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "dataset_huggingface_workspace": dataset_huggingface_workspace,
        "model_output_huggingface_workspace": huggingface_user,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True

    huggingface_estimator = HuggingFace(
        entry_point="finetune.py",
        source_dir=str(finetuning_dir),
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        role=settings.AWS_ARN_ROLE,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        requirements_file=finetuning_requirements_path,
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
    )

    huggingface_estimator.fit()
```

**Key Concepts**:
- **`entry_point="finetune.py"` + `source_dir`**: SageMaker packages the
  finetuning folder and runs the script in a managed container. `source_dir` is
  `Path(__file__).resolve().parent`, so the whole folder ships.
- **`hyperparameters`** are passed as CLI flags to `finetune.py`; this is why the
  script uses `argparse`.
- **`environment`** injects secrets (HF token, Comet key) into the container as
  env vars - never as hyperparameters, which are logged to CloudWatch.
- **`requirements_file`** installs `unsloth==2024.9.post2`, `trl==0.9.6`,
  `transformers==4.43.3`, `torch==2.4.0`, and the rest at container startup. The
  pinned versions matter: the managed image ships older libraries, and the
  requirement file upgrades them.
- **`transformers_version`/`pytorch_version`/`py_version`** pin the base image to
  `transformers==4.36`, `pytorch==2.1`, `py310`. Unsloth may require newer
  versions than the SageMaker-managed image provides. If the job fails on the
  Unsloth install, use a custom container image.
- **`huggingface_user`** is discovered from `whoami`; the output model is pushed
  to `{huggingface_user}/TwinLlama-3.1-8B`, so you must be authenticated.

### 3. `deploy/huggingface/run.py` - Create the Endpoint

```python
# llm_engineering/infrastructure/aws/deploy/huggingface/run.py
def create_endpoint(endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED) -> None:
    assert settings.AWS_ARN_ROLE is not None, "AWS_ARN_ROLE is not set in the .env file."

    logger.info(f"Creating endpoint with endpoint_type = {endpoint_type} and model_id = {settings.HF_MODEL_ID}")

    llm_image = get_huggingface_llm_image_uri("huggingface", version="2.2.0")

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=settings.AWS_ARN_ROLE,
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        endpoint_config_name=settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
        gpu_instance_type=settings.GPU_INSTANCE_TYPE,
        resources=model_resource_config,
        endpoint_type=endpoint_type,
    )


if __name__ == "__main__":
    create_endpoint(endpoint_type=EndpointType.MODEL_BASED)
```

**Key Concepts**:
- **`get_huggingface_llm_image_uri("huggingface", version="2.2.0")`** returns the
  AWS-owned Deep Learning Container (DLC) purpose-built for Hugging Face LLM
  inference. It includes the TGI serving stack, so you do not build an image.
- **Two endpoint types**:
  - `MODEL_BASED`: one model per endpoint.
  - `INFERENCE_COMPONENT_BASED`: supports multiple models and per-component
    autoscaling. Required for the autoscaling script.
- **Type mismatch trap**: the function default is
  `INFERENCE_COMPONENT_BASED`, but the `__main__` block explicitly passes
  `MODEL_BASED`. Running the file directly creates a model-based endpoint, which
  the autoscaling script cannot scale. Call `create_endpoint()` from code (or
  change the `__main__` argument) to get component-based.
- **`ResourceManager`** and **`DeploymentService`** are injected, keeping the
  strategy testable.

### 4. `deploy/huggingface/config.py` - Runtime Environment

```python
# llm_engineering/infrastructure/aws/deploy/huggingface/config.py
hugging_face_deploy_config = {
    "HF_MODEL_ID": settings.HF_MODEL_ID,
    "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
    "SM_NUM_GPUS": json.dumps(settings.SM_NUM_GPUS),
    "MAX_INPUT_LENGTH": json.dumps(settings.MAX_INPUT_LENGTH),
    "MAX_TOTAL_TOKENS": json.dumps(settings.MAX_TOTAL_TOKENS),
    "MAX_BATCH_TOTAL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "MAX_BATCH_PREFILL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "HF_MODEL_QUANTIZE": "bitsandbytes",
}

model_resource_config = ResourceRequirements(
    requests={
        "copies": settings.COPIES,
        "num_accelerators": settings.GPUS,
        "num_cpus": settings.CPUS,
        "memory": 5 * 1024,
    },
)
```

**Key Concepts**:
- **`HF_MODEL_ID`** points at the merged model on the Hub. The default is
  `mlabonne/TwinLlama-3.1-8B-DPO` (`settings.py:46`).
- **`HF_MODEL_QUANTIZE="bitsandbytes"`** loads the model in 8-bit on the
  endpoint, cutting VRAM enough to fit within `ml.g5.2xlarge` (24 GB) after the
  merged 16-bit weights are loaded.
- **`MAX_BATCH_PREFILL_TOKENS` is set equal to `MAX_BATCH_TOTAL_TOKENS`** in this
  repo (both resolve to 4096). The book snippet shows `10000`, but the repo is
  the source of truth. TGI requires the prefill budget not to exceed the total
  batch budget.
- **Token limits** map onto TGI scheduling:
  - `MAX_INPUT_LENGTH` = 2048, the longest accepted prompt.
  - `MAX_TOTAL_TOKENS` = 4096, input plus generation.
  - `MAX_BATCH_TOTAL_TOKENS` = 4096, aggregate tokens across the batch.
  - `MAX_BATCH_PREFILL_TOKENS` = 4096, same value here.
- **`json.dumps`** is applied to every numeric value because TGI reads env vars
  as strings and validates JSON-typed numbers.
- **`model_resource_config`** declares per-replica resources used with inference
  components: `copies=1`, `num_accelerators=1` (GPUs), `num_cpus=2`, and
  `memory=5120` MB.

### 5. `sagemaker_huggingface.py` - Strategy and Service

```python
# llm_engineering/infrastructure/aws/deploy/huggingface/sagemaker_huggingface.py
class SagemakerHuggingfaceStrategy(DeploymentStrategy):
    def __init__(self, deployment_service) -> None:
        self.deployment_service = deployment_service

    def deploy(self, role_arn, llm_image, config, endpoint_name, endpoint_config_name,
               gpu_instance_type, resources=None, endpoint_type=EndpointType.MODEL_BASED) -> None:
        try:
            self.deployment_service.deploy(
                role_arn=role_arn, llm_image=llm_image, config=config,
                endpoint_name=endpoint_name, endpoint_config_name=endpoint_config_name,
                gpu_instance_type=gpu_instance_type, resources=resources, endpoint_type=endpoint_type,
            )
            logger.info("Deployment completed successfully.")
        except Exception as e:
            logger.error(f"Error during deployment: {e}")
            raise


class DeploymentService:
    def __init__(self, resource_manager):
        self.sagemaker_client = boto3.client("sagemaker", region_name=settings.AWS_REGION, ...)
        self.resource_manager = resource_manager

    @staticmethod
    def prepare_and_deploy_model(role_arn, llm_image, config, endpoint_name, update_endpoint,
                                 gpu_instance_type, resources=None, endpoint_type=EndpointType.MODEL_BASED) -> None:
        huggingface_model = HuggingFaceModel(role=role_arn, image_uri=llm_image, env=config)

        huggingface_model.deploy(
            instance_type=gpu_instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            update_endpoint=update_endpoint,
            resources=resources,
            tags=[{"Key": "task", "Value": "model_task"}],
            endpoint_type=endpoint_type,
            container_startup_health_check_timeout=900,
        )
```

The `deploy` method checks for an existing config before deploying:

```python
    def deploy(self, ..., endpoint_type=EndpointType.MODEL_BASED) -> None:
        try:
            if self.resource_manager.endpoint_config_exists(endpoint_config_name=endpoint_config_name):
                logger.info(f"Endpoint configuration {endpoint_config_name} exists. Using existing configuration...")
            else:
                logger.info(f"Endpoint configuration{endpoint_config_name} does not exist.")

            self.prepare_and_deploy_model(
                role_arn=role_arn, llm_image=llm_image, config=config,
                endpoint_name=endpoint_name, update_endpoint=False,
                resources=resources, endpoint_type=endpoint_type,
                gpu_instance_type=gpu_instance_type,
            )
        except Exception as e:
            logger.error(f"Failed to deploy model to SageMaker: {e}")
            raise
```

**Key Concepts**:
- **`container_startup_health_check_timeout=900`** allows 15 minutes for the
  container to download the model and start TGI. LLM endpoints are slow to boot;
  the default timeout is too short.
- **`ResourceManager.endpoint_config_exists` / `endpoint_exists`** check state
  with boto3 before deploying, so the flow is idempotent. `endpoint_exists` is
  implemented in `model/utils.py:31`.
- **`SagemakerHuggingfaceStrategy` implements the `DeploymentStrategy` ABC** from
  `domain/inference.py`, keeping the domain contract independent of boto3.
- **The strategy logs and re-raises**: on failure the error propagates, so the
  CLI exits non-zero rather than silently succeeding.

### 5b. `model/utils.py` - `ResourceManager`

```python
# llm_engineering/model/utils.py
class ResourceManager:
    def __init__(self) -> None:
        self.sagemaker_client = boto3.client("sagemaker", region_name=settings.AWS_REGION, ...)

    def endpoint_config_exists(self, endpoint_config_name: str) -> bool:
        try:
            self.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            return True
        except ClientError:
            return False

    def endpoint_exists(self, endpoint_name: str) -> bool:
        try:
            self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            return True
        except self.sagemaker_client.exceptions.ResourceNotFoundException:
            return False
```

Note the asymmetry: `endpoint_config_exists` catches the broad `ClientError`,
while `endpoint_exists` catches the specific `ResourceNotFoundException`. Both
work, but the specific exception is the safer pattern because it will not swallow
a permissions error as "does not exist."

### 6. Autoscaling and Teardown

```python
# llm_engineering/infrastructure/aws/deploy/autoscaling_sagemaker_endpoint.py
class AutoscalingSagemakerEndpoint:
    def __init__(
        self,
        auto_scaling_client,
        inference_component_name: str,
        endpoint_name: str,
        initial_copy_count: int = 1,
        max_copy_count: int = 6,
        target_value: float = 4.0,
    ):
        self.service_namespace = "sagemaker"
        self.scalable_dimension = "sagemaker:inference-component:DesiredCopyCount"
        self.resource_id = f"inference-component/{self.inference_component_name}"

    def setup_autoscaling(self):
        scalable_target = ScalableTarget(
            auto_scaling_client=self.auto_scaling_client,
            service_namespace=self.service_namespace,
            resource_id=self.resource_id,
            scalable_dimension=self.scalable_dimension,
            min_capacity=self.initial_copy_count,
            max_capacity=self.max_copy_count,
        )
        scalable_target.register()

        policy = TargetTrackingScalingPolicy(
            auto_scaling_client=self.auto_scaling_client,
            policy_name=self.endpoint_name,
            service_namespace=self.service_namespace,
            resource_id=self.resource_id,
            scalable_dimension=self.scalable_dimension,
            target_value=self.target_value + 1,  # Example adjustment, should be based on specific use case
            scale_in_cooldown=200,
            scale_out_cooldown=200,
        )
        policy.apply_policy()
```

The policy itself uses a predefined metric:

```python
class TargetTrackingScalingPolicy(ScalingPolicyStrategy):
    def apply_policy(self):
        self.aas_client.put_scaling_policy(
            PolicyName=self.policy_name,
            PolicyType="TargetTrackingScaling",
            ServiceNamespace=self.service_namespace,
            ResourceId=self.resource_id,
            ScalableDimension=self.scalable_dimension,
            TargetTrackingScalingPolicyConfiguration={
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerInferenceComponentInvocationsPerCopy",
                },
                "TargetValue": self.target_value,
                "ScaleInCooldown": self.scale_in_cooldown,
                "ScaleOutCooldown": self.scale_out_cooldown,
            },
        )
```

**Key Concepts**:
- **Register vs policy**: registering a scalable target says *what* to scale and
  the min/max bounds; the policy says *when* to scale. Both steps are required.
- **Target tracking** on `SageMakerInferenceComponentInvocationsPerCopy` scales
  the number of copies to hold a target load per copy. Application Auto Scaling
  creates and manages the CloudWatch alarms for you, like a thermostat.
- **The code adds 1 to the target** (`self.target_value + 1`), an explicit
  "example adjustment" in the repo. With the default `target_value=4.0`, the
  effective target is 5.0 invocations per copy.
- **Cooldowns (200s)** prevent flapping. They are generous because GPU endpoints
  take minutes to become healthy, so reacting quickly would be counterproductive.
- **Only works with `INFERENCE_COMPONENT_BASED` endpoints**; model-based
  endpoints have no inference component to scale.
- **Bounds**: `MinCapacity=self.initial_copy_count` (default 1), so at least one
  copy always serves. `MaxCapacity=self.max_copy_count` (default 6) caps cost.

### Target-tracking worked example

Assume the endpoint has 1 copy, each copy can comfortably handle about 5
invocations per second (the effective target), and the ALB spreads requests.

| Incoming RPS | Copies needed | Action |
|--------------|---------------|--------|
| 3 | 1 | hold (below target) |
| 5 | 1 | hold (at target) |
| 10 | 2 | scale out |
| 30 | 6 | scale out to max |
| 0 (idle) | 1 | scale in to min |

The system converges to `copies ≈ RPS / target`. Over-scaling (too low a target,
too short a cooldown) wastes money on idle GPUs; under-scaling (too high a
target) produces timeouts. Bounds and the target must be tuned against a stress
test, like hyperparameters.

```python
# delete_sagemaker_endpoint.py
def delete_endpoint_and_config(endpoint_name) -> None:
    sagemaker_client = boto3.client("sagemaker", ...)

    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    config_name = response["EndpointConfigName"]

    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    response = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    model_name = response["ProductionVariants"][0]["ModelName"]

    sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
    sagemaker_client.delete_model(ModelName=model_name)
```

Deletes the endpoint, then its config, then the model. Run this to stop billing.

> ⚠️ **Repo quirk (flag it)**: on the line that fetches the model name, the code
> calls `describe_endpoint_config(EndpointConfigName=endpoint_name)` even though
> it already has `config_name`. It also calls `describe_endpoint_config` with
> `EndpointName=`-style naming. If the endpoint name and the config name differ
> (SageMaker auto-generates a config name when none is passed to
> `prepare_and_deploy_model`), this call raises `ClientError`, the handler logs
> "Error getting model name.", and `model_name` is never assigned. The following
> `delete_model(ModelName=model_name)` then raises `NameError`, which is **not**
> caught by `except ClientError`. The endpoint and its config are still deleted,
> but the model may be orphaned or the script may crash at the end. Check the
> SageMaker console after running the delete script.

---

## 🛠️ Hands-On: Deploy and Call the Endpoint

### Step 1: One-time IAM setup

```bash
poetry poe create-sagemaker-role
# creates sagemaker_user_credentials.json → paste keys into .env as
# AWS_ACCESS_KEY / AWS_SECRET_KEY

poetry poe create-sagemaker-execution-role
# creates sagemaker_execution_role.json → paste RoleArn into .env as AWS_ARN_ROLE
```

Or directly:

```bash
python -m llm_engineering.infrastructure.aws.roles.create_sagemaker_role
python -m llm_engineering.infrastructure.aws.roles.create_execution_role
```

### Step 2: Create the endpoint

```bash
poetry poe deploy-inference-endpoint
# equivalent to:
python -m llm_engineering.infrastructure.aws.deploy.huggingface.run
```

This takes 10-15 minutes (model download + container boot). The health check
timeout is 900 seconds, so give it time before assuming failure.

### Step 3: Call the endpoint

```python
import json
import boto3
from llm_engineering.settings import settings

client = boto3.client("sagemaker-runtime", region_name=settings.AWS_REGION)
payload = {
    "inputs": "Below is an instruction...\n### Instruction:\nWhat is an LLM Twin?\n### Response:\n",
    "parameters": {"max_new_tokens": 150, "temperature": 0.01, "top_p": 0.9},
}

response = client.invoke_endpoint(
    EndpointName=settings.SAGEMAKER_ENDPOINT_INFERENCE,
    ContentType="application/json",
    Body=json.dumps(payload),
)
print(json.loads(response["Body"].read()))
```

Or use the repo's own client:

```bash
poetry poe test-sagemaker-endpoint
# equivalent to: python -m llm_engineering.model.inference.test
```

### Step 4: Add autoscaling (component-based only)

```python
import boto3
from llm_engineering.infrastructure.aws.deploy.autoscaling_sagemaker_endpoint import (
    AutoscalingSagemakerEndpoint,
)

client = boto3.client("application-autoscaling", region_name=settings.AWS_REGION)
AutoscalingSagemakerEndpoint(
    auto_scaling_client=client,
    inference_component_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
    endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
).setup_autoscaling()
```

### Step 5: Delete the endpoint (do not skip)

```bash
poetry poe delete-inference-endpoint
# equivalent to:
python -m llm_engineering.infrastructure.aws.deploy.delete_sagemaker_endpoint
```

Then open the SageMaker console and confirm the endpoint, config, and model are
gone.

---

## 📝 Exercise 1: Scale Policy Tuning

### Task

Reason about autoscaling without burning money.

1. Read `TargetTrackingScalingPolicy.apply_policy`.
2. Set a target of **2 invocations per copy** vs **10 per copy** (remember the
   code adds 1).
3. Predict which configuration scales out sooner and which uses more idle GPU
   time.
4. Explain the cooldown values in terms of cold-start time (model boot is slow).
5. Compute the number of copies for 40 RPS under both targets.

**Goal**: Connect autoscaling parameters to cost and latency for LLM endpoints.

---

## 📝 Exercise 2: Right-Size the Endpoint Config

### Task

1. Start from `settings.py` defaults: `SM_NUM_GPUS=1`, `MAX_INPUT_LENGTH=2048`,
   `MAX_TOTAL_TOKENS=4096`, `MAX_BATCH_TOTAL_TOKENS=4096`, `COPIES=1`,
   `GPUS=1`, `CPUS=2`.
2. For each change below, predict the effect on VRAM, latency, throughput, and
   cost, then justify:
   - Raise `COPIES` from 1 to 4.
   - Raise `MAX_TOTAL_TOKENS` from 4096 to 8192.
   - Remove `HF_MODEL_QUANTIZE` (run in 16-bit).
3. Which single change is most likely to OOM `ml.g5.2xlarge` (24 GB)?
4. Which change most directly raises throughput without raising latency?

**Goal**: Internalize the throughput/latency/cost triangle from the book's four
requirements.

---

## 🐛 Common Pitfalls

- **Forgetting to delete the endpoint**: the top cost risk. Always run the delete
  script and verify in the console.
- **Role not set**: `create_endpoint` asserts `AWS_ARN_ROLE`. Create the
  execution role first and paste the ARN into `.env`.
- **Wrong setting name**: the settings field is `AWS_ARN_ROLE`; the book text
  occasionally writes `ARN_ROLE`. Use the repo name.
- **Unsloth in the SageMaker image**: the managed HuggingFace image may not
  satisfy Unsloth's version requirements. If training fails, build a custom
  container. The repo pins `unsloth==2024.9.post2` in `requirements.txt`.
- **Endpoint type mismatch**: autoscaling requires `INFERENCE_COMPONENT_BASED`.
  The `run.py` `__main__` block uses `MODEL_BASED`, so pass the type explicitly
  in code when you need autoscaling.
- **Secrets in hyperparameters**: hyperparameters are logged in CloudWatch. Put
  tokens in `environment`, as the code does.
- **`MAX_BATCH_PREFILL_TOKENS` > `MAX_BATCH_TOTAL_TOKENS`**: TGI rejects the
  config. The repo keeps them equal (4096).
- **Delete-script quirk**: see the caution above; `describe_endpoint_config` is
  called with `endpoint_name`, which can leave `model_name` unbound.
- **Non-idempotent delete**: deleting an endpoint that does not exist logs an
  error but continues; always confirm in the console.
- **Cold-start timeouts**: an LLM endpoint can take 10+ minutes. Do not lower
  `container_startup_health_check_timeout` below 900.
- **Region mismatch**: `AWS_REGION` defaults to `eu-central-1`; the DLC image URI
  is resolved for that region. A mismatched region fails at deploy time.

---

## 🎓 Knowledge Check

1. **What does the trust relationship in the execution role allow?**
   - Answer: `sagemaker.amazonaws.com` to assume the role via `sts:AssumeRole`.

2. **Why is `container_startup_health_check_timeout` 900 seconds?**
   - Answer: LLM containers must download multi-GB weights and boot TGI before
     serving traffic.

3. **What does `HF_MODEL_QUANTIZE="bitsandbytes"` accomplish?**
   - Answer: Loads the model quantized (8-bit) so it fits the endpoint GPU's
     VRAM.

4. **Which endpoint type can be autoscaled?**
   - Answer: `INFERENCE_COMPONENT_BASED`.

5. **Why are HF and Comet tokens passed via `environment`, not `hyperparameters`?**
   - Answer: Hyperparameters are logged to CloudWatch; environment variables are
     not surfaced the same way.

6. **What three resources does the delete script remove?**
   - Answer: The endpoint, its endpoint configuration, and the model.

7. **What is the difference between registering a scalable target and creating a
   scaling policy?**
   - Answer: The target defines the resource and its min/max bounds; the policy
     defines the metric and thresholds that trigger scaling.

8. **Why is `MAX_BATCH_PREFILL_TOKENS` set equal to `MAX_BATCH_TOTAL_TOKENS`?**
   - Answer: The prefill budget must not exceed the total batch budget; TGI
     rejects an invalid combination.

9. **What metric does the autoscaling policy track?**
   - Answer: `SageMakerInferenceComponentInvocationsPerCopy`.

10. **Why 200-second cooldowns?**
    - Answer: GPU endpoints are slow to become healthy, so short cooldowns would
      cause flapping.

11. **What is `DeploymentService.prepare_and_deploy_model` responsible for?**
    - Answer: Constructing the `HuggingFaceModel` and calling `.deploy()` with
      the instance type, count, resources, and endpoint type.

12. **Which class implements the `DeploymentStrategy` ABC, and why does that
    matter?**
    - Answer: `SagemakerHuggingfaceStrategy`; the domain depends on the
      interface, so another cloud could be swapped in.

13. **Why does the training job use `source_dir=str(finetuning_dir)`?**
    - Answer: SageMaker packages the whole folder (including `finetune.py` and
      `requirements.txt`) and runs it in the managed container.

14. **What does `get_huggingface_llm_image_uri` return?**
    - Answer: The AWS-owned Hugging Face LLM DLC image URI, which includes the
      TGI serving stack.

15. **Name two Hugging Face DLC/TGI features that help LLM serving.**
    - Answer: Continuous/dynamic batching and tensor parallelism (also 8-bit
      quantization, safetensors loading, token streaming).

---

## 📖 Glossary

- **DLC (Deep Learning Container)**: A prebuilt AWS Docker image with frameworks
  and the TGI serving stack; no image build needed.
- **TGI (Text Generation Inference)**: Hugging Face's serving engine for LLMs,
  with batching, tensor parallelism, quantization, and streaming.
- **Endpoint**: A scalable HTTP API SageMaker hosts for real-time predictions.
- **Endpoint configuration**: The hardware/software spec (instance type, count)
  used when creating an endpoint.
- **Inference component**: The binding of a model and a resource config to an
  endpoint; enables multi-model and per-component scaling.
- **Execution role**: The IAM role SageMaker assumes to access S3, ECR, and
  CloudWatch on your behalf.
- **Trust relationship**: The policy that names which principal may assume a
  role.
- **ARN (Amazon Resource Name)**: A unique identifier for an AWS resource.
- **Target tracking**: An autoscaling policy that adjusts capacity to hold a
  metric at a target value.
- **Cooldown**: The pause after a scaling action before another can occur.
- **Prefill**: The phase where the prompt is processed before token generation.
- **Quantization**: Reducing weight precision to cut memory (e.g. 8-bit
  bitsandbytes).

---

## 🔗 Next Session

**Session 6.1**: FastAPI REST API

We build the local inference API that wraps the model (SageMaker endpoint or
local vLLM), calling this endpoint through the `Inference` interface.

See [Session 6.1: FastAPI REST API](session_6.1_fastapi_api.md).

---

## 📚 Additional Resources

- [SageMaker Hugging Face Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/)
- [Deploy LLMs with Hugging Face on SageMaker](https://huggingface.co/docs/sagemaker/inference)
- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference)
- [Application Auto Scaling for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
- [SageMaker autoscaling prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-prerequisites.html)
- Related sessions: [Session 5.2 DPO](session_5.2_dpo.md),
  [Session 6.1 FastAPI REST API](session_6.1_fastapi_api.md)

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Sessions 5.1, 5.2, AWS account

**Outcome**: You can provision IAM roles, run a SageMaker training job, deploy a
quantized LLM endpoint on the correct endpoint type, autoscale it with target
tracking, call it, and tear it down safely.
