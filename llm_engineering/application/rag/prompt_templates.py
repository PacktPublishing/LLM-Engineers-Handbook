from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from pydantic import BaseModel


class BasePromptTemplate(ABC, BaseModel):
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass


class QueryExpansionTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to generate {expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "expand_to_n": expand_to_n,
            },
        )


class SelfQueryTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the user or author id. 
    Your response should consists of only the extracted id (e.g. 1345256), nothing else.
    If the user question does not contain any author id, you should return the following token: none.
    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])


class RerankingTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to rerank passages related to a query
    based on their relevance. 
    The most relevant passages should be put at the beginning. 
    You should only pick at max {keep_top_k} passages.
    The provided and reranked documents are separated by '{separator}'.
    
    The following are passages related to this query: {question}.
    
    Passages: 
    {passages}
    """

    def create_template(self, keep_top_k: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question", "passages"],
            partial_variables={"keep_top_k": keep_top_k, "separator": self.separator},
        )

    @property
    def separator(self) -> str:
        return "\n#next-document#\n"
