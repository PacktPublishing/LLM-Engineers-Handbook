from loguru import logger
from zenml import pipeline

from llm_engineering.application.steps import feature_engineering as fe_steps


@pipeline
def feature_engineering(user_full_name: str) -> None:
    logger.info(f"Computing features for user: {user_full_name}")
    
    # TODO: Fix passed types between steps
    # TODO: Is it ok to handle all data types and collections through the DataType enum?
    # TODO: Recheck data models and handlers class hierachy.
    # TODO: Refactor the DB module
    # TODO: Add metadata to the artifacts
    
    raw_documents = fe_steps.query_data_warehouse(user_full_name)
    cleaned_documents = fe_steps.clean_documents(raw_documents)
    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    
    logger.info("Feature engineering completed.")
    
    
    

