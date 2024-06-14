from langchain_openai import ChatOpenAI

from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument
from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .prompt_templates import SelfQueryTemplate


class SelfQuery:
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock
        
    def generate(self, query: Query) -> Query:
        if self._mock:
            return query
        
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)

        chain = prompt | model

        response = chain.invoke({"question": query})
        username_or_id = response.content
        username_or_id = username_or_id.strip("\n ")

        if username_or_id == "none":
            username_or_id = None
        else:
            first_name, last_name = utils.split_user_full_name(username_or_id)
            user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
            username_or_id = user.id

        query.author_id = username_or_id

        return query
