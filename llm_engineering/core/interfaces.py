from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel


class LLMInterface(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def get_answer(self, prompt: str, *args, **kwargs):
        pass


class BasePromptTemplate(ABC, BaseModel):
    @abstractmethod
    def create_template(self, *args) -> str:
        pass


class DeploymentStrategy(ABC):
    @abstractmethod
    def deploy(self, model, endpoint_name: str, endpoint_config_name: str) -> None:
        pass


class Inference(ABC):
    """An abstract class for performing inference."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def set_payload(self, inputs, parameters=None):
        pass

    @abstractmethod
    def inference(self):
        pass


class Summarize(ABC):
    """A class for summarizing documents."""

    def __init__(self, llm: Inference):
        self.llm = llm

    @abstractmethod
    def summarize(self, document_structure: dict):
        pass


class Task:
    """An abstract class for performing a task."""

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the task."""
        raise NotImplementedError
