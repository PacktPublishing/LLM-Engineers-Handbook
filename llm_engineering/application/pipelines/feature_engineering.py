from loguru import logger
from zenml import pipeline

from llm_engineering.application.steps.feature_engineering import query_data_warehouse


@pipeline
def feature_engineering(user_full_name: str) -> None:
    raw_data = query_data_warehouse(user_full_name)

