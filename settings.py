import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env"))


class CommonSettings(BaseSettings):
    ARN_ROLE: str
    PASSPHRASE_AWS_VAULT: str = "test"
    HUGGING_FACE_HUB_TOKEN: str
    OPENAI_API_KEY: str
    COMET_ML_API_KEY: str
    COMET_ML_WORKSPACE: str
    DYNAMO_TABLE: str = "ai-results"
    STREAMLIT_TABLE: str = "dev-epostbox-streamlit_human_check"


class ModelDeploySettings(CommonSettings):
    HF_MODEL_ID: str = "test"
    GPU_INSTANCE_TYPE: str = "test"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 20000
    MAX_TOTAL_TOKENS: int = 32000
    MAX_BATCH_TOTAL_TOKENS: int = 12000
    COPIES: int = 4  # Number of replicas
    GPUS: int = 1  # Number of GPUs
    CPUS: int = 8  # Number of CPU cores  96 // num_replica - more for management
    RETURN_FULL_TEXT: bool = False


class ExtractionSettings(CommonSettings):
    SAGEMAKER_ENDPOINT_CONFIG_EXTRACTION: str = "test"
    SAGEMAKER_INFERENCE_COMPONENT_EXTRACTION: str = "test"
    SAGEMAKER_ENDPOINT_EXTRACTION: str = "test"
    SAGEMAKER_MODEL_EXTRACTION: str = "test"
    TEMPERATURE_EXTRACTION: float = 0.1
    TOP_P_EXTRACTION: float = 0.9
    MAX_NEW_TOKENS_EXTRACTION: int = 400


class SummarizationSettings(CommonSettings):
    SAGEMAKER_ENDPOINT_CONFIG_SUMMARIZATION: str
    SAGEMAKER_INFERENCE_COMPONENT_SUMMARIZATION: str
    SAGEMAKER_ENDPOINT_SUMMARIZATION: str
    SAGEMAKER_MODEL_SUMMARIZATION: str
    TEMPERATURE_SUMMARY: float = 0.8
    TOP_P_SUMMARY: float = 0.9
    MAX_NEW_TOKENS_SUMMARY: int = 150


class ClassificationSettings(CommonSettings):
    SAGEMAKER_ENDPOINT_CONFIG_CLASSIFICATION: str = "test"
    SAGEMAKER_INFERENCE_COMPONENT_CLASSIFICATION: str = "test"
    SAGEMAKER_ENDPOINT_CLASSIFICATION: str = "test"
    SAGEMAKER_MODEL_CLASSIFICATION: str = "test"
    TEMPERATURE_CLASSIFICATION: float = 0.01
    TOP_P_CLASSIFICATION: float = 0.9
    MAX_NEW_TOKENS_CLASSIFICATION: int = 150


class EvaluationSettings(CommonSettings):
    EVALUATION_HF_MODEL_ID: str = "test"
    EVALUATION_GPU_INSTANCE_TYPE: str = "test"
    SAGEMAKER_ENDPOINT_CONFIG_EVALUATION: str = "test"
    SAGEMAKER_INFERENCE_COMPONENT_EVALUATION: str = "test"
    SAGEMAKER_ENDPOINT_EVALUATION: str = "test"
    SAGEMAKER_MODEL_EVALUATION: str = "test"
    TEMPERATURE_EVALUATION: float = 0.01
    TOP_P_EVALUATION: float = 0.9
    MAX_NEW_TOKENS_EVALUATION: int = 150


class SentenceTransformerSettings(CommonSettings):
    MODEL_NAME_SENTENCE_TRANSFORMER: str = "test"
    SAGEMAKER_BUCKET: str = "test"
    SENTENCE_TRANSFORMER_INSTANCE_TYPE: str = "test"


class Settings(
    ModelDeploySettings,
    ExtractionSettings,
    SummarizationSettings,
    ClassificationSettings,
    EvaluationSettings,
    SentenceTransformerSettings,
):
    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "../.env")


settings = Settings()
