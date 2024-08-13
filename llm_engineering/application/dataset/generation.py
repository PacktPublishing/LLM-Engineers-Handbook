import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering import domain
from llm_engineering.application import utils
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domain.types import DataCategory
from llm_engineering.settings import settings

from . import constants, splits
from .output_parsers import ListPydanticOutputParser


class DatasetGenerator:
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)

    system_prompt_template: str = "You are a helpful assistant who \
            generates instruction-answer pairs based on the given context. \
            You will imitate the casual tone and writing style of the context."
    prompt_template_str: str = """I want to create an AI assistant that can write paragraphs and \
{{ data_category }} about machine learning topics. Based on the following extract, \
generate three instruction-answer pairs. Each instruction should ask \
to write about a specific topic contained in the context, and each answer \
should provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Copy the writing style from the extract to imitate the author's personality. \
Do not use markdown.

Extract:
{{ extract }}

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), a list of objects only with fields called instruction and content.
Do not add any extra characters and make sure it is a list with objects in valid json format following exactly the next structure:\n
'```json\n[{"instruction": "<generated instruction>"}, {"instruction": "<generated instruction here>"}, ...]\n```'

Output JSON format. Make sure that you generate exactly three instruction-answer pairs, and the keys of the JSON object are 'instruction' and 'answer':
[
    {"instruction": "<generated instruction> 1", "answer": "<generated answer> 1"},
    {"instruction": "<generated instruction> 2", "answer": "<generated answer> 2"},
    {"instruction": "<generated instruction> 3", "answer": "<generated answer> 3"}
]
"""

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        return Prompt(
            template=cls.system_prompt_template,
            input_variables={},
            content=cls.system_prompt_template,
        )

    @classmethod
    def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        grouped_prompts = {}
        grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
        for category, category_documents in grouped_cleaned_documents.items():
            category_prompts = [cls.get_prompt(document) for document in category_documents]
            grouped_prompts[category] = category_prompts

        return grouped_prompts

    @classmethod
    def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
        data_category = document.get_category()

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )
        input_variables = {
            "data_category": data_category,
            "extract": document.content,
        }
        prompt = prompt_template.format(**input_variables)
        prompt_tokens = cls.tokenizer.encode(prompt)
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)

        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            data_category=data_category,
            document=document,
        )

        return prompt

    @classmethod
    def generate(
        cls,
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> domain.dataset.TrainTestSplit:
        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt,
        ) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

            return messages

        if mock:
            llm = FakeListLLM(responses=[constants.MOCKED_RESPONSE])
        else:
            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, max_tokens=800, temperature=0.7, n=1
            )
        parser = ListPydanticOutputParser(pydantic_object=domain.dataset.InstructDatasetSample)

        chain = llm | parser

        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]
            batches = utils.misc.batch(langchain_category_prompts, size=4)

            flattened_instruct_dataset_samples = []
            for batch in batches:
                try:
                    batched_instruct_dataset_samples = chain.batch(batch, stop=None)
                except OutputParserException:
                    logger.error(f"Failed to parse the output JSON for a batch for category {category}")

                for instruct_dataset_sample_batch in batched_instruct_dataset_samples:
                    flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)

            dataset = domain.dataset.InstructDataset(category=category, samples=flattened_instruct_dataset_samples)
            datasets[category] = dataset

        train_test_split = splits.create_train_test_split(datasets, test_size=test_size, random_state=42)

        return train_test_split
