import json
import logging
from typing import Any, Dict, Optional

import boto3
from langchain_community.llms import SagemakerEndpoint

from llm_engineering.core.interfaces import Inference
from llm_engineering.core.langchain_utils import ContentHandler
from llm_engineering.settings import settings


class LLMInferenceSagemakerEndpoint(Inference):
    """
    Class for performing inference using a SageMaker endpoint for LLM (Language Model) schemas.
    """

    def __init__(
        self,
        endpoint_name: str,
        default_payload: Optional[Dict[str, Any]] = None,
        inference_component_name: Optional[str] = None,
    ):
        super().__init__()

        self.client = boto3.client(
            "sagemaker-runtime",
            region_name="eu-central-1",
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.endpoint_name = endpoint_name
        self.payload = default_payload if default_payload else self._default_payload()
        self.inference_component_name = inference_component_name
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _default_payload(self) -> Dict[str, Any]:
        """
        Generates the default payload for the inference request.

        Returns:
            dict: The default payload.
        """
        return {
            "inputs": "How is the weather?",
            "parameters": {
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "top_p": settings.TOP_P_INFERENCE,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "return_full_text": False,
            },
        }

    def set_payload(self, inputs: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets the payload for the inference request.

        Args:
            inputs (str): The input text for the inference.
            parameters (dict, optional): Additional parameters for the inference. Defaults to None.
        """
        self.payload["inputs"] = inputs
        if parameters:
            self.payload["parameters"].update(parameters)

    def inference(self) -> Dict[str, Any]:
        """
        Performs the inference request using the SageMaker endpoint.

        Returns:
            dict: The response from the inference request.
        Raises:
            Exception: If an error occurs during the inference request.
        """
        try:
            logging.info(f"Inference request sent with parameters: {self.payload['parameters']}")
            invoke_args = {
                "EndpointName": self.endpoint_name,
                "ContentType": "application/json",
                "Body": json.dumps(self.payload),
            }
            if self.inference_component_name not in ["None", None]:
                invoke_args["InferenceComponentName"] = self.inference_component_name
            response = self.client.invoke_endpoint(**invoke_args)
            response_body = response["Body"].read().decode("utf8")

            return json.loads(response_body)

        except Exception as e:
            logging.error(f"An error occurred during inference: {e}")
            raise


class LLMLangchainSagemakerInference(Inference):
    """
    Class for performing inference using a SageMaker endpoint for LLMLangchain model.
    """

    def __init__(self, endpoint_name: str, inference_component_name: Optional[str] = None):
        super().__init__()
        endpoint_kwargs = {"CustomAttributes": "accept_eula=true"}
        if inference_component_name != "None":
            endpoint_kwargs["InferenceComponentName"] = inference_component_name
        self.model = SagemakerEndpoint(
            endpoint_name=endpoint_name,
            region_name="eu-central-1",
            model_kwargs={
                "max_new_tokens": settings.MAX_NEW_TOKENS_EXTRACTION,
                "top_p": settings.TOP_P_EXTRACTION,
                "temperature": settings.TEMPERATURE_SUMMARY,
                "do_sample": True,
            },
            endpoint_kwargs=endpoint_kwargs,
            content_handler=ContentHandler(),
        )
        self.inputs = None
        self.parameters = None

    def set_payload(self, inputs, parameters=None):
        """
                Set the payload for inference.
        qqq
                Args:
                    inputs: The input data for inference.
                    parameters: Additional parameters for the endpoint (optional).
        """
        self.inputs = inputs
        self.parameters = parameters

    def inference(self):
        """
        Perform inference using the set payload.

        Returns:
            The inference result.
        Raises:
            ValueError: If no inputs are set for inference.
        """
        if self.inputs is None:
            raise ValueError("No inputs set for inference")

        # If additional parameters are needed for the endpoint, modify the call accordingly
        return self.model(self.inputs, **(self.parameters or {}))
