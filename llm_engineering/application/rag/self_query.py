from langchain_openai import ChatOpenAI

from llm_engineering.settings import settings

from .chain import GeneralChain
from .prompt_templates import SelfQueryTemplate


class SelfQuery:
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock
        
    def generate(self, query: str) -> str | None:
        if self._mock:
            return None
        
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)

        chain = GeneralChain().get_chain(
            llm=model, output_key="metadata_filter_value", template=prompt
        )

        response = chain.invoke({"question": query})
        result = response["metadata_filter_value"]

        if result == "none":
            return None

        return result
