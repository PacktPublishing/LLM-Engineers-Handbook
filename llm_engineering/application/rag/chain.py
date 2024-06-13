from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


class GeneralChain:
    @staticmethod
    def get_chain(llm, template: PromptTemplate, output_key: str, verbose=True):
        return LLMChain(
            llm=llm, prompt=template, output_key=output_key, verbose=verbose
        )
