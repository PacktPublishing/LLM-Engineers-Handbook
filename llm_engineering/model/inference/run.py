from __future__ import annotations

import logging

from llm_engineering.core.interfaces import Inference
from llm_engineering.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_engineering.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InferenceExecutor:
    def __init__(self, llm: Inference, text: str, prompt: str):
        self.llm = llm
        self.text = text
        self.prompt = prompt

    def execute(self) -> str:
        """Extracts entities from a text."""
        self.llm.set_payload(
            inputs=self.prompt.format(TEXT=self.text),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
            },
        )
        extraction = self.llm.inference()[0]["generated_text"]
        return extraction


if __name__ == "__main__":
    text = "The weather in Berlin is nice today."
    prompt = 'Continue the following text: "{TEXT}"'
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    InferenceExecutor(llm, text, prompt).execute()
