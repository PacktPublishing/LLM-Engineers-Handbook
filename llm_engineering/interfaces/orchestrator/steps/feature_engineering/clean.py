from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.preprocessing import CleaningDispatcher
from llm_engineering.domain.cleaned_documents import CleanedDocument


@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    cleaned_documents = []
    for document in documents:
        cleaned_document = CleaningDispatcher.dispatch(document)
        cleaned_documents.append(cleaned_document)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))

    return cleaned_documents


def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    metadata = {"num_documents": len(cleaned_documents)}
    for document in cleaned_documents:
        category = document.get_category()
        if category not in metadata:
            metadata[category] = {}
        if "authors" not in metadata[category]:
            metadata[category]["authors"] = list()

        metadata[category]["num_documents"] = metadata[category].get("num_documents", 0) + 1
        metadata[category]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
