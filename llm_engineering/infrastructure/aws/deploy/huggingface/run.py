from sagemaker.enums import EndpointType
from sagemaker.huggingface import get_huggingface_llm_image_uri

from llm_engineering.model.deploy.huggingface.config import (
    hugging_face_deploy_config,
    model_resource_config,
)
from llm_engineering.model.deploy.huggingface.sagemaker_huggingface import (
    DeploymentService,
    SagemakerHuggingfaceStrategy,
)
from llm_engineering.model.utils import (
    ResourceManager,
)
from llm_engineering.settings import settings


def create_huggingface_endpoint(task_name, endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED):
    gpu_instance_type = settings.GPU_INSTANCE_TYPE

    if task_name == "inference":
        endpoint_name = settings.SAGEMAKER_ENDPOINT_INFERENCE
        endpoint_config_name = settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE

    else:
        raise ValueError("Invalid task name")

    llm_image = get_huggingface_llm_image_uri("huggingface", version=None)

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    """
    Deploy endpoint without inference component EndpointType.MODEL_BASED
    """
    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=settings.ARN_ROLE,
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=endpoint_name,
        endpoint_config_name=endpoint_config_name,
        gpu_instance_type=gpu_instance_type,
        resources=model_resource_config,
        endpoint_type=endpoint_type,
    )


if __name__ == "__main__":
    task_name = "inference"
    create_huggingface_endpoint(task_name, endpoint_type=EndpointType.MODEL_BASED)
