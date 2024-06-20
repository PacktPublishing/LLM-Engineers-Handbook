from zenml import pipeline

from llm_engineering.interfaces.orchestrator.steps import feature_engineering as fe_steps


@pipeline
def feature_engineering(user_full_names: list[str]) -> None:
    raw_documents = fe_steps.query_data_warehouse(user_full_names)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    fe_steps.load_to_vector_db(embedded_documents)
