from abc import ABC, abstractmethod


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
