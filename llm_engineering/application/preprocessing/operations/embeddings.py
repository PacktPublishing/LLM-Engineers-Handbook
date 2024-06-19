from llm_engineering.application.networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


def embedd_text(text: str | list[str]) -> list[float] | list[list[float]]:
    return embedding_model(text, to_list=True)
