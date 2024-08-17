import json

from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements

from llm_engineering.settings import settings

hugging_face_deploy_config = {
    "HF_MODEL_ID": settings.HF_MODEL_ID,
    "SM_NUM_GPUS": json.dumps(settings.SM_NUM_GPUS),  # Number of GPU used per replica
    "MAX_INPUT_LENGTH": json.dumps(settings.MAX_INPUT_LENGTH),  # Max length of input text
    "MAX_TOTAL_TOKENS": json.dumps(settings.MAX_TOTAL_TOKENS),  # Max length of the generation (including input text)
    "MAX_BATCH_TOTAL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "HUGGING_FACE_HUB_TOKEN": settings.HUGGING_FACE_HUB_TOKEN,
    "MAX_BATCH_PREFILL_TOKENS": "10000",
    # 'HF_MODEL_QUANTIZE': "bitsandbytes",
}


model_resource_config = ResourceRequirements(
    requests={
        "copies": settings.COPIES,  # Number of replicas
        "num_accelerators": settings.GPUS,  # Number of GPUs
        "num_cpus": settings.CPUS,  # Number of CPU cores  96 // num_replica - more for management
        "memory": 5 * 1024,  # Minimum memory in MB 1152 // num_replica - more for management
    },
)
