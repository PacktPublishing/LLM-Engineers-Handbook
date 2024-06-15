from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.domain.base import VectorBaseDocument


@step
def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> None:
    logger.info(f"Loading # documents: {len(documents)}")

    grouped_documents = VectorBaseDocument.group_by_collection(documents)
    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")
        
        document_class.bulk_insert(documents)
