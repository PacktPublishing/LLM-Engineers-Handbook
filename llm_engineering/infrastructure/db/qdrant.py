from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Batch, Distance, VectorParams
from qdrant_client.models import CollectionInfo

from llm_engineering.settings import settings


class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __init__(self) -> None:
        if self._instance is None:
            try:
                if settings.USE_QDRANT_CLOUD:
                    self._instance = QdrantClient(
                        url=settings.QDRANT_CLOUD_URL,
                        api_key=settings.QDRANT_APIKEY,
                    )
                    
                    uri = settings.QDRANT_CLOUD_URL
                else:
                    self._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT,
                    )
                    
                    uri = f"{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"
                
                logger.info(
                    f"Connection to Qdrant DB with URI successful: {uri}"
                )
            except UnexpectedResponse:
                logger.exception(
                    "Couldn't connect to Qdrant.",
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                    url=settings.QDRANT_CLOUD_URL,
                )

                raise

    def get_or_create_collection(
        self, collection_name: str, use_vector_index: bool = True
    ) -> CollectionInfo:
        try:
            return self.get_collection(collection_name=collection_name)
        except Exception:
            collection_created = self.create_collection(
                collection_name=collection_name, use_vector_index=use_vector_index
            )
            if collection_created is False:
                raise RuntimeError(f"Couldn't create collection {collection_name}")

            return self.get_collection(collection_name=collection_name)

    def get_collection(self, collection_name: str) -> CollectionInfo:
        return self._instance.get_collection(collection_name=collection_name)

    def create_collection(self, collection_name: str, use_vector_index: bool = True):
        if use_vector_index is True:
            vectors_config = VectorParams(
                size=settings.EMBEDDING_SIZE, distance=Distance.COSINE
            )
        else:
            vectors_config = {}

        return self._instance.create_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )

    def upsert(self, collection_name: str, points: Batch):
        try:
            return self._instance.upsert(
                collection_name=collection_name, wait=True, points=points
            )
        except Exception:
            logger.exception("An error occurred while inserting data.")

            raise

    def search(
        self,
        collection_name: str,
        query_vector: list,
        query_filter: models.Filter,
        limit: int,
    ) -> list:
        return self._instance.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def scroll(self, collection_name: str, limit: int):
        return self._instance.scroll(collection_name=collection_name, limit=limit)

    def close(self) -> None:
        if self._instance:
            self._instance.close()


connection = QdrantDatabaseConnector()
