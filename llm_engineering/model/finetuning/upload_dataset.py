import boto3
import sagemaker

from llm_engineering.settings import settings

# Initialize the SageMaker session
sagemaker_boto3_session = boto3.Session(
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
    region_name="eu-central-1",  # Optional, specify your AWS region
)

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session(boto_session=sagemaker_boto3_session)

# Get the default S3 bucket for this SageMaker session
bucket = sagemaker_session.default_bucket()

# Upload the dataset to S3
input_data = sagemaker_session.upload_data(
    path="/Users/vesaalexandru/Workspaces/cube/LLM-Engineering/dummy_dataset",
    bucket=bucket,
    key_prefix="dummy-fine-tuning-data",
)

print(f"Data uploaded to: {input_data}")
