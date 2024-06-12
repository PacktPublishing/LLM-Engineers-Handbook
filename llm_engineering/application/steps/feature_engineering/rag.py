from typing_extensions import Annotated
from zenml import step

from llm_engineering.application.preprocessing import (
    ChunkingDispatcher,
    EmbeddingDispatcher,
)
from llm_engineering.domain.base import DataModel


@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    embedded_documents = []
    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        for chunk in chunks:
            # TODO: Create method to handle a batch of chunks.
            embedded_chunk = EmbeddingDispatcher.dispatch(chunk)
            embedded_documents.append(embedded_chunk)

    return embedded_documents
