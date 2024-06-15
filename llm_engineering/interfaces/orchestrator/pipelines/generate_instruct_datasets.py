from zenml import pipeline

from llm_engineering.interfaces.orchestrator.steps import (
    generate_instruct_datasets as cd_steps,
)


@pipeline
def generate_instruct_datasets() -> None:
    cleaned_documents = cd_steps.query_feature_store()
    prompts = cd_steps.create_prompts(documents=cleaned_documents)
    cd_steps.generate(prompts=prompts)
