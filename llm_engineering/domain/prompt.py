from langchain_core.prompts import PromptTemplate

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.types import DataCategory


class Prompt(VectorBaseDocument):
    template: PromptTemplate
    input_variables: dict
    content: str
    num_tokens: int | None = None

    class Config:
        category = DataCategory.PROMPT


class GenerateDatasetSamplesPrompt(Prompt):
    data_category: DataCategory
    document: CleanedDocument
