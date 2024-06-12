from InstructorEmbedding import INSTRUCTOR
from sentence_transformers.SentenceTransformer import SentenceTransformer

from llm_engineering.settings import settings


def embedd_text(text: str):
    model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_ID)

    return model.encode(text)


def embedd_repositories(text: str):
    model = INSTRUCTOR(settings.CODE_EMBEDDING_MODEL_ID)
    sentence = text
    instruction = "Represent the structure of the repository"

    return model.encode([instruction, sentence])
