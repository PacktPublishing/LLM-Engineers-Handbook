from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument


@step
def get_or_create_user(user_full_name: str) -> Annotated[UserDocument, "user"]:
    logger.info(f"Getting or creating user: {user_full_name}")

    first_name, last_name = utils.split_user_full_name(user_full_name)

    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    return user
