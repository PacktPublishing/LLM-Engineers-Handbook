from typing import Generator

import tiktoken
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from llm_engineering import domain
from llm_engineering.application import utils
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domain.types import DataCategory
from llm_engineering.settings import settings

from .output_parsers import ListPydanticOutputParser


class DatasetGenerator:
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)

    system_prompt_template: str = "You are a technical writer creating posts and articles about AI and MLOps."
    prompt_template_str: str = """I will give you batches of contents of {{ data_category }}. Generate me exactly 1 instruction for each of them. The {{ data_category }} text
for which you have to generate the instructions is under 'Content number' x lines. 

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), a list of objects only with fields called instruction and content.
Do not add any extra characters and make sure it is a list with objects in valid json format following exactly the next structure:\n
'```json\n[{"instruction": "<generated instruction>"}, {"instruction": "<generated instruction here>"}, ...]\n```'

You must generate exactly a list of {{ len_documents }} json objects, using the contents provided under CONTENTS FOR GENERATION\n

\nCONTENTS FOR GENERATION:\n
{% for doc in documents %}
Content number {{ doc.index }}:
{{ doc.content | e }}
{% endfor %}"""

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        return Prompt(
            template=PromptTemplate.from_template(cls.system_prompt_template),
            input_variables={},
            content=cls.system_prompt_template,
        )

    @classmethod
    def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        grouped_prompts = {}
        grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
        for category, documents in grouped_cleaned_documents.items():
            batched_documents_generator = cls._batch_by_category(category, documents)
            category_prompts = [cls.get_prompt(batch) for batch in batched_documents_generator]
            grouped_prompts[category] = category_prompts

        return grouped_prompts

    @classmethod
    def _batch_by_category(
        cls, category: DataCategory, documents: list[CleanedDocument]
    ) -> Generator[list, None, None]:
        match category:
            case DataCategory.ARTICLES:
                return utils.misc.batch(documents, size=1)
            case DataCategory.POSTS:
                return utils.misc.batch(documents, size=5)
            case DataCategory.REPOSITORIES:
                return utils.misc.batch(documents, size=1)
            case _:
                raise ValueError(f"Unsupported category: {category}")

    @classmethod
    def get_prompt(cls, documents: list[CleanedDocument]) -> GenerateDatasetSamplesPrompt:
        assert len(documents) > 0, "At least one document is required"

        data_category = documents[0].get_category()
        assert all(
            data_category == document.get_category() for document in documents
        ), "All documents must be of the same category"

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )
        input_variables = {
            "data_category": data_category,
            "len_documents": len(documents),
            "documents": [{"index": i, "content": doc.content} for i, doc in enumerate(documents)],
        }
        prompt = prompt_template.format(**input_variables)
        prompt_tokens = cls.tokenizer.encode(prompt)
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)

        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            data_category=data_category,
            documents=documents,
        )

        return prompt

    @classmethod
    def generate(
        cls,
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
        mock: bool = False,
    ) -> dict[DataCategory, domain.dataset.InstructDataset]:
        def _batch_to_langchain_prompt(
            prompt: GenerateDatasetSamplesPrompt,
        ) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

            return messages

        if mock:
            llm = FakeListLLM(
                responses=[
                    '```json\n[{"instruction": "mock instruction"}, {"instruction": "mock instruction"}, {"instruction": "mock instruction"}]\n```'
                ]
            )
        else:
            llm = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0)
        parser = ListPydanticOutputParser(pydantic_object=domain.dataset.InstructDatasetSample)

        chain = llm | parser

        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_batch_to_langchain_prompt(batch) for batch in category_prompts]
            batched_instruct_dataset_samples = chain.batch(langchain_category_prompts)

            flattened_instruct_dataset_samples = []
            for prompt, per_prompt_instruct_dataset_samples in zip(
                category_prompts, batched_instruct_dataset_samples, strict=False
            ):
                prompt_documents_as_response = prompt.documents
                for document_as_response, instruct_dataset_sample in zip(
                    prompt_documents_as_response, per_prompt_instruct_dataset_samples, strict=False
                ):
                    instruct_dataset_sample.response = document_as_response.content

                    flattened_instruct_dataset_samples.append(instruct_dataset_sample)

            dataset = domain.dataset.InstructDataset(category=category, samples=flattened_instruct_dataset_samples)
            datasets[category] = dataset

        return datasets
