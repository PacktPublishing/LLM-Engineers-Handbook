import logging
import sys

import boto3
from botocore.exceptions import ClientError

from llm_engineering.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def delete_endpoint_and_config(endpoint_name):
    """
    Deletes an AWS SageMaker endpoint and its associated configuration.
    Args:
    endpoint_name (str): The name of the SageMaker endpoint to delete.
    Returns:
    None
    """
    try:
        sagemaker_client = boto3.client(
            "sagemaker",
            region_name="eu-central-1",
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
    except Exception as e:
        logger.error(f"Error creating SageMaker client: {e}")
        return

    # Delete the endpoint
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint '{endpoint_name}' deletion initiated.")
    except ClientError as e:
        logger.error(f"Error deleting endpoint: {e}")
        return

    # Get the endpoint configuration name
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = response["EndpointConfigName"]
    except ClientError as e:
        logger.error(f"Error getting endpoint configuration name: {e}")
        return

    # Delete the endpoint configuration
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
        logger.info(f"Endpoint configuration '{config_name}' deleted.")
    except ClientError as e:
        logger.error(f"Error deleting endpoint configuration: {e}")


def run():
    if len(sys.argv) != 2:
        logger.error("Usage: python script_name.py <endpoint_name>")
        sys.exit(1)

    endpoint_name = sys.argv[1]
    logger.info(f"Attempting to delete endpoint: {endpoint_name}")
    delete_endpoint_and_config(endpoint_name)


if __name__ == "__main__":
    run()
