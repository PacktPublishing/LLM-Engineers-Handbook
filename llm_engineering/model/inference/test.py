from loguru import logger

from llm_engineering.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_engineering.model.inference.run import InferenceExecutor
from llm_engineering.settings import settings

if __name__ == "__main__":
    text = "The weather in Berlin is nice today."
    prompt = 'Continue the following text: "{TEXT}"'
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, text, prompt).execute()

    logger.info(answer)
