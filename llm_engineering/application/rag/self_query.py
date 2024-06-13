from langchain_openai import ChatOpenAI

from .chain import GeneralChain
from .prompt_templates import SelfQueryTemplate
from llm_engineering.settings import settings


class SelfQuery:
    @staticmethod
    def generate_response(query: str) -> str | None:
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
