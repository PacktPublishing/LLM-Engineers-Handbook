from loguru import logger
from zenml import pipeline

from llm_engineering.application.steps import create_dataset as cd_steps


@pipeline
def create_dataset() -> None:
    logger.info("Creating instruct dataset for fine-tuning.")

    cleaned_documents = cd_steps.query_feature_store()

    logger.info("Feature engineering completed.")
