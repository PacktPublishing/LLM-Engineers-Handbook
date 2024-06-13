from InstructorEmbedding import INSTRUCTOR
from numpy.typing import NDArray
from sentence_transformers.SentenceTransformer import SentenceTransformer

from llm_engineering.settings import settings


def embedd_text(text: str) -> NDArray:
    model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_ID)

    return model.encode(text)


def embedd_repositories(text: str) -> NDArray:
    model = INSTRUCTOR(settings.CODE_EMBEDDING_MODEL_ID)
    sentence = text
    # TODO: Improve instruction.
    instruction = "Represent the structure of a code repository."

    return model.encode([instruction, sentence])
