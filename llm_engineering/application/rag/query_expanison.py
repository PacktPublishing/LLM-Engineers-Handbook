from langchain_openai import ChatOpenAI

from llm_engineering.settings import settings

from .chain import GeneralChain
from .prompt_templates import QueryExpansionTemplate


class QueryExpansion:
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock

    def generate(self, query: str, expand_to_n: int) -> list[str]:
        if self._mock:
            return [query for _ in range(expand_to_n)]
        
        query_expansion_template = QueryExpansionTemplate()
        prompt_template = query_expansion_template.create_template(expand_to_n)
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)

        chain = GeneralChain().get_chain(
            llm=model, output_key="expanded_queries", template=prompt_template
        )

        response = chain.invoke({"question": query})
        result = response["expanded_queries"]

        queries = result.strip().split(query_expansion_template.separator)
        stripped_queries = [
            stripped_item for item in queries if (stripped_item := item.strip())
        ]

        return stripped_queries
