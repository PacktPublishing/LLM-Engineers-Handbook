import enum
from typing import Optional

from loguru import logger

try:
    import boto3
    from sagemaker.enums import EndpointType
    from sagemaker.huggingface import HuggingFaceModel
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS or SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.domain.inference import DeploymentStrategy
from llm_engineering.settings import settings


class SagemakerHuggingfaceStrategy(DeploymentStrategy):
    def __init__(self, deployment_service) -> None:
        """
        Initializes the deployment strategy with the necessary services.

        :param deployment_service: The service handling the deployment details.
        :param logger: Logger for logging information and errors.
        """
        self.deployment_service = deployment_service

    def deploy(
        self,
        role_arn: str,
        llm_image: str,
        config: dict,
        endpoint_name: str,
        endpoint_config_name: str,
        gpu_instance_type: str,
        resources: Optional[dict] = None,
        endpoint_type: enum.Enum = EndpointType.MODEL_BASED,
    ) -> None:
        """
        Initiates the deployment process for a HuggingFace model on AWS SageMaker.

        :param role_arn: AWS role ARN with permissions for SageMaker deployment.
        :param llm_image: URI for the HuggingFace model Docker image.
        :param config: Configuration settings for the model environment.
        :param endpoint_name: Name of the SageMaker endpoint.
        :param endpoint_config_name: Name of the SageMaker endpoint configuration.
        :param resources: Optional resources for the model deployment (used for multi model endpoints)
        :param endpoint_type: can be EndpointType.MODEL_BASED (without inference component)
                or EndpointType.INFERENCE_COMPONENT (with inference component)

        """

        logger.info("Starting deployment using Sagemaker Huggingface Strategy...")
        logger.info(
            f"Deployment parameters: nb of replicas: {settings.COPIES}, nb of gpus:{settings.GPUS}, instance_type:{settings.GPU_INSTANCE_TYPE}"
        )
        try:
            # Delegate to the deployment service to handle the actual deployment details
            self.deployment_service.deploy(
                role_arn=role_arn,
                llm_image=llm_image,
                config=config,
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name,
                gpu_instance_type=gpu_instance_type,
                resources=resources,
                endpoint_type=endpoint_type,
            )
            logger.info("Deployment completed successfully.")
        except Exception as e:
            logger.error(f"Error during deployment: {e}")
            raise


class DeploymentService:
    def __init__(self, resource_manager):
        """
        Initializes the DeploymentService with necessary dependencies.

        :param resource_manager: Manages resources and configurations for deployments.
        :param settings: Configuration settings for deployment.
        :param logger: Optional logger for logging messages. If None, the standard logging module will be used.
        """

        self.sagemaker_client = boto3.client(
            "sagemaker",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.resource_manager = resource_manager

    def deploy(
        self,
        role_arn: str,
        llm_image: str,
        config: dict,
        endpoint_name: str,
        endpoint_config_name: str,
        gpu_instance_type: str,
        resources: Optional[dict] = None,
        endpoint_type: enum.Enum = EndpointType.MODEL_BASED,
    ) -> None:
        """
        Handles the deployment of a model to SageMaker, including checking and creating
        configurations and endpoints as necessary.

        :param role_arn: The ARN of the IAM role for SageMaker to access resources.
        :param llm_image: URI of the Docker image in ECR for the HuggingFace model.
        :param config: Configuration dictionary for the environment variables of the model.
        :param endpoint_name: The name for the SageMaker endpoint.
        :param endpoint_config_name: The name for the SageMaker endpoint configuration.
        :param resources: Optional resources for the model deployment (used for multi model endpoints)
        :param endpoint_type: can be EndpointType.MODEL_BASED (without inference component)
                or EndpointType.INFERENCE_COMPONENT (with inference component)
        :param gpu_instance_type: The instance type for the SageMaker endpoint.
        """

        try:
            # Check if the endpoint configuration exists
            if self.resource_manager.endpoint_config_exists(endpoint_config_name=endpoint_config_name):
                logger.info(f"Endpoint configuration {endpoint_config_name} exists. Using existing configuration...")
            else:
                logger.info(f"Endpoint configuration{endpoint_config_name} does not exist.")

            # Prepare and deploy the HuggingFace model
            self.prepare_and_deploy_model(
                role_arn=role_arn,
                llm_image=llm_image,
                config=config,
                endpoint_name=endpoint_name,
                update_endpoint=False,
                resources=resources,
                endpoint_type=endpoint_type,
                gpu_instance_type=gpu_instance_type,
            )

            logger.info(f"Successfully deployed/updated model to endpoint {endpoint_name}.")
        except Exception as e:
            logger.error(f"Failed to deploy model to SageMaker: {e}")

            raise

    @staticmethod
    def prepare_and_deploy_model(
        role_arn: str,
        llm_image: str,
        config: dict,
        endpoint_name: str,
        update_endpoint: bool,
        gpu_instance_type: str,
        resources: Optional[dict] = None,
        endpoint_type: enum.Enum = EndpointType.MODEL_BASED,
    ) -> None:
        """
        Prepares and deploys/updates the HuggingFace model on SageMaker.

        :param role_arn: The ARN of the IAM role.
        :param llm_image: The Docker image URI for the HuggingFace model.
        :param config: Configuration settings for the model.
        :param endpoint_name: The name of the endpoint.
        :param update_endpoint: Boolean flag to update an existing endpoint.
        :param gpu_instance_type: The instance type for the SageMaker endpoint.
        :param resources: Optional resources for the model deployment(used for multi model endpoints)
        :param endpoint_type: can be EndpointType.MODEL_BASED (without inference component)
                or EndpointType.INFERENCE_COMPONENT (with inference component)
        """

        huggingface_model = HuggingFaceModel(
            role=role_arn,
            image_uri=llm_image,
            env=config,
        )

        # Deploy or update the model based on the endpoint existence
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
