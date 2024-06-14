from langchain_openai import ChatOpenAI

from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .prompt_templates import QueryExpansionTemplate


class QueryExpansion:
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock

    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
        assert expand_to_n > 0, f"'expand_to_n' should be greater than 0. Got {expand_to_n}."
        
        if self._mock:
            return [query for _ in range(expand_to_n)]

        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1)
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)
        
        chain = prompt | model

        response = chain.invoke({"question": query})
        result = response.content

        queries_content = result.strip().split(query_expansion_template.separator)
        
        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]

        return queries
