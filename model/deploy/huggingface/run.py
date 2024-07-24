from sagemaker.enums import EndpointType
from sagemaker.huggingface import get_huggingface_llm_image_uri

from ai_document_tasks.core.aws.utils import authenticate_with_aws_vault
from ai_document_tasks.model.deploy.huggingface.config import (
    hugging_face_deploy_config,
    model_resource_config,
)
from ai_document_tasks.model.deploy.huggingface.sagemaker_huggingface import (
    DeploymentService,
    SagemakerHuggingfaceStrategy,
)
from ai_document_tasks.model.utils import (
    ResourceManager,
)
from ai_document_tasks.settings import settings


def create_huggingface_endpoint(task_name, endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED):

    gpu_instance_type = settings.GPU_INSTANCE_TYPE

    if task_name == "summarization":
        endpoint_name = settings.SAGEMAKER_ENDPOINT_SUMMARIZATION
        endpoint_config_name = settings.SAGEMAKER_ENDPOINT_CONFIG_SUMMARIZATION
    elif task_name == "classification":
        endpoint_name = settings.SAGEMAKER_ENDPOINT_CLASSIFICATION
        endpoint_config_name = settings.SAGEMAKER_ENDPOINT_CONFIG_CLASSIFICATION
    elif task_name == "extraction":
        endpoint_name = settings.SAGEMAKER_ENDPOINT_EXTRACTION
        endpoint_config_name = settings.SAGEMAKER_ENDPOINT_CONFIG_EXTRACTION
    elif task_name == "evaluation":
        endpoint_name = settings.SAGEMAKER_ENDPOINT_EVALUATION
        endpoint_config_name = settings.SAGEMAKER_ENDPOINT_CONFIG_EVALUATION
        hugging_face_deploy_config["HF_MODEL_ID"] = settings.EVALUATION_HF_MODEL_ID
        gpu_instance_type = settings.EVALUATION_GPU_INSTANCE_TYPE
    else:
        raise ValueError("Invalid task name")

    role_arn = settings.ARN_ROLE
    llm_image = get_huggingface_llm_image_uri("huggingface", version=None)

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    """
    Deploy endpoint without inference component EndpointType.MODEL_BASED
    """
    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=role_arn,
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=endpoint_name,
        endpoint_config_name=endpoint_config_name,
        gpu_instance_type=gpu_instance_type,
        resources=model_resource_config,
        endpoint_type=endpoint_type,
    )


if __name__ == "__main__":
    authenticate_with_aws_vault("epostbox.development")
    task_name = "extraction"
    create_huggingface_endpoint(task_name, endpoint_type=EndpointType.MODEL_BASED)
