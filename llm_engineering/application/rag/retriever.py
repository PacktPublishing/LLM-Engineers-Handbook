import concurrent.futures

from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers.SentenceTransformer import SentenceTransformer

from llm_engineering.application import utils
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from llm_engineering.settings import settings

from .query_expanison import QueryExpansion
from .reranking import Reranker
from .self_query import SelfQuery


class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._embedder = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_ID)
        
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)

    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3,
        apply_rerank: bool = True,
    ) -> list:
        n_generated_queries = self._query_expander.generate(
            query, expand_to_n=expand_to_n_queries
        )
        logger.info(
            "Successfully generated queries for search.",
            num_queries=len(n_generated_queries),
        )

        author_id = self._metadata_extractor.generate(query)
        logger.info(
            "Successfully extracted the author_id from the query.",
            author_id=author_id,
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search_single_query, query, author_id, k)
                for query in n_generated_queries
            ]

            n_k_documents = [
                task.result() for task in concurrent.futures.as_completed(search_tasks)
            ]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(
            "All documents retrieved successfully.", num_documents=len(n_k_documents)
        )

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, n_k_documents, k)
        else:
            k_documents = []

        return k_documents

    def _search_single_query(
        self, generated_query: str, author_id: str | None = None, k: int = 3
    ) -> list[EmbeddedChunk]:
        assert k >= 3, "k should be >= 3"

        def _search_data_category(
            data_category_odm: type[EmbeddedChunk], query_vector: list
        ) -> list[EmbeddedChunk]:
            if author_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="author_id",
                            match=MatchValue(
                                value=author_id,
                            ),
                        )
                    ]
                )
            else:
                query_filter = None

            return data_category_odm.search(
                query_vector=query_vector,
                limit=k // 3,
                query_filter=query_filter,
            )

        query_vector = self._embedder.encode(generated_query).tolist()

        post_chunks = _search_data_category(EmbeddedPostChunk, query_vector)
        articles_chunks = _search_data_category(EmbeddedArticleChunk, query_vector)
        repositories_chunks = _search_data_category(
            EmbeddedRepositoryChunk, query_vector
        )

        retrieved_chunks = post_chunks + articles_chunks + repositories_chunks

        return retrieved_chunks

    def rerank(
        self, query: str, documents: list[EmbeddedChunk], keep_top_k: int
    ) -> list[EmbeddedChunk]:
        passages = [chunk.content for chunk in documents]
        reranked_documents = self._reranker.generate(
            query=query, passages=passages, keep_top_k=keep_top_k
        )

        logger.info(
            "Documents reranked successfully.", num_documents=len(reranked_documents)
        )

        return reranked_documents
