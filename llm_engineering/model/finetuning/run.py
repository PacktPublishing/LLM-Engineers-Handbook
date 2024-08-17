from pathlib import Path

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

from llm_engineering.settings import settings

# Set up the SageMaker session
sagemaker_boto3_session = boto3.Session(
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
    region_name=settings.AWS_REGION,
)
sagemaker_session = sagemaker.Session(boto_session=sagemaker_boto3_session)

# Set up paths
base_dir = Path("/Users/vesaalexandru/Workspaces/cube/LLM-Engineering")
script_dir = base_dir / "llm_engineering/model/finetuning"
script_path = script_dir / "finetune.py"
requirements_path = script_dir / "requirements.txt"
bucket = sagemaker_session.default_bucket()
input_data = f"s3://{bucket}/dummy-fine-tuning-data"

# Verify that the necessary files exist
for file_path in [script_path, requirements_path]:
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

# Create the HuggingFace estimator
huggingface_estimator = HuggingFace(
    entry_point="finetune.py",
    source_dir=str(script_dir),
    instance_type=settings.GPU_INSTANCE_TYPE,
    instance_count=1,
    role=settings.AWS_ARN_ROLE,
    transformers_version="4.28.1",
    pytorch_version="2.0.0",
    py_version="py310",
    hyperparameters={
        "epochs": 1,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "learning_rate": 3e-4,
        "model_id": "facebook/opt-125m",
        "use_qlora": True,
        "max_seq_length": 512,
    },
    requirements_file=requirements_path,
)

# Start the training job
huggingface_estimator.fit({"training": input_data})
