from numpy.typing import NDArray
from sentence_transformers.SentenceTransformer import SentenceTransformer

from llm_engineering.settings import settings


def embedd_text(text: str) -> NDArray:
    model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_ID, device=settings.RAG_MODEL_DEVICE)

    return model.encode(text)
