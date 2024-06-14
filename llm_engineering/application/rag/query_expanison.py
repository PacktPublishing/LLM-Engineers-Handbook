from langchain_openai import ChatOpenAI

from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .chain import GeneralChain
from .prompt_templates import QueryExpansionTemplate


class QueryExpansion:
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock

    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
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

        queries_content = result.strip().split(query_expansion_template.separator)
        stripped_queries = [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]

        return stripped_queries
