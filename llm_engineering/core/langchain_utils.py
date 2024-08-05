import json

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        # Structure the payload according to your inference example

        input_payload = {"inputs": prompt, "parameters": model_kwargs}
        input_str = json.dumps(input_payload)
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        if isinstance(response_json, list) and len(response_json) > 0:
            full_text = response_json[0].get("generated_text", "")

            # Split the text based on a unique delimiter (e.g., "SUMMARY:")
            parts = full_text.split("SUMMARY:")
            if len(parts) > 1:
                # Return only the part after the delimiter
                generated_summary = parts[1]
                return generated_summary.strip()
            else:
                print("Delimiter 'SUMMARY:' not found in the response")
                return ""
        else:
            print("Unexpected response format or empty response:", response_json)
            return ""


class GeneralChain:
    @staticmethod
    def get_chain(llm, template: str, input_variables=None, verbose=True, output_key=""):
        prompt_template = PromptTemplate(input_variables=input_variables, template=template, verbose=verbose)
        return LLMChain(
            llm=llm,
            prompt=prompt_template,
            output_key=output_key,
            verbose=verbose,
        )
