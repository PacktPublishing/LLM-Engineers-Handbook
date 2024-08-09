from zenml import pipeline

from steps import feature_engineering as fe_steps


@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    return [last_step_1.invocation_id, last_step_2.invocation_id]
