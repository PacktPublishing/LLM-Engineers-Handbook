from langchain_openai import ChatOpenAI

from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .chain import GeneralChain
from .prompt_templates import SelfQueryTemplate


class SelfQuery:
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock
        
    def generate(self, query: Query) -> Query:
        if self._mock:
            return query
        
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)

        chain = GeneralChain().get_chain(
            llm=model, output_key="metadata_filter_value", template=prompt
        )

        response = chain.invoke({"question": query})
        author_id = response["metadata_filter_value"]

        if author_id == "none":
            author_id = None

        query.author_id = author_id

        return query
