from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from llm_engineering.application.networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()

def chunk_text(text: str) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"], chunk_size=500, chunk_overlap=0
    )
    text_split = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=50,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )
    chunks = []

    for section in text_split:
        chunks.extend(token_splitter.split_text(section))

    return chunks
