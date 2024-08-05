import logging

import boto3
from botocore.exceptions import ClientError

from llm_engineering.settings import settings


class ResourceManager:
    def __init__(self):
        # self.session = boto3.Session(profile_name="decodingml")
        # self.sagemaker_client = self.session.client("sagemaker", region_name="eu-central-1")
        self.sagemaker_client = boto3.client(
            "sagemaker",
            region_name="eu-central-1",
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.logger = logging.getLogger(__name__)

    def endpoint_config_exists(self, endpoint_config_name: str) -> bool:
        """Check if the SageMaker endpoint configuration exists."""
        try:
            self.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            self.logger.info(f"Endpoint configuration '{endpoint_config_name}' exists.")
            return True
        except ClientError:
            self.logger.info(f"Endpoint configuration '{endpoint_config_name}' does not exist.")
            return False

    def endpoint_exists(self, endpoint_name: str) -> bool:
        """Check if the SageMaker endpoint exists."""
        try:
            self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            self.logger.info(f"Endpoint '{endpoint_name}' exists.")
            return True
        except self.sagemaker_client.exceptions.ResourceNotFoundException:
            self.logger.info(f"Endpoint '{endpoint_name}' does not exist.")
            return False
