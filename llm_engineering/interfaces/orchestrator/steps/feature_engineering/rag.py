from typing_extensions import Annotated
from zenml import step

from llm_engineering.application import utils
from llm_engineering.application.preprocessing import ChunkingDispatcher, EmbeddingDispatcher


@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    embedded_documents = []
    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_documents.extend(batched_embedded_chunks)

    return embedded_documents
