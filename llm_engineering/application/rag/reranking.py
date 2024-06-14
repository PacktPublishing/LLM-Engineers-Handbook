from sentence_transformers import CrossEncoder

from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.settings import settings


class Reranker:
    def __init__(self, mock: bool = False) -> None:
        self._model = CrossEncoder(
            settings.RERANKING_EMBEDDING_MODEL_ID,
            max_length=settings.TEXT_EMBEDDING_MODEL_MAX_INPUT_LENGTH,
        )
        self._mock = mock

    def generate(
        self, query: str, chunks: list[EmbeddedChunk], keep_top_k: int
    ) -> list[EmbeddedChunk]:
        if self._mock:
            return chunks

        query_doc_tuples = [(query, chunk.content) for chunk in chunks]
        scores = self._model.predict(query_doc_tuples)
        scores = scores.tolist()

        scored_query_doc_tuples = list(zip(scores, chunks))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents
