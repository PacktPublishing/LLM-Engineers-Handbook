from __future__ import annotations

import logging

from llm_engineering.core.interfaces import Inference
from llm_engineering.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_engineering.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        context: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
You are a content creator. Write a paragraph about what the user asked you to while using the provided context.
User query: {query}
Context: {context}
            """
        else:
            self.prompt = prompt

    def execute(self) -> str:
        self.llm.set_payload(
            inputs=self.prompt.format(query=self.query, context=self.context),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
            },
        )
        answer = self.llm.inference()[0]["generated_text"]

        return answer


if __name__ == "__main__":
    text = "The weather in Berlin is nice today."
    prompt = 'Continue the following text: "{TEXT}"'
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    InferenceExecutor(llm, text, prompt).execute()
