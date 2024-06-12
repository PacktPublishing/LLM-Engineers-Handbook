from loguru import logger

from llm_engineering.domain.base import DataModel
from llm_engineering.domain.documents import BaseDocument
from llm_engineering.domain.types import DataType

from .chunking_data_handlers import (
    ArticleChunkingHandler,
    ChunkingDataHandler,
    PostChunkingHandler,
    RepositoryChunkingHandler,
)
from .cleaning_data_handlers import (
    ArticleCleaningHandler,
    CleaningDataHandler,
    PostCleaningHandler,
    RepositoryCleaningHandler,
)
from .embedding_data_handlers import (
    ArticleEmbeddingHandler,
    EmbeddingDataHandler,
    PostEmbeddingHandler,
    RepositoryEmbeddingHandler,
)


class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> CleaningDataHandler:
        if data_type == DataType.POSTS:
            return PostCleaningHandler()
        elif data_type == DataType.ARTICLES:
            return ArticleCleaningHandler()
        elif data_type == DataType.REPOSITORIES:
            return RepositoryCleaningHandler()
        else:
            raise ValueError("Unsupported data type")


class CleaningDispatcher:
    cleaning_factory = CleaningHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: BaseDocument) -> DataModel:
        data_type = data_model.get_collection_name()
        handler = cls.cleaning_factory.create_handler(data_type)
        clean_model = handler.clean(data_model)

        logger.info(
            "Data cleaned successfully.",
            data_type=data_type,
            cleaned_content_len=len(clean_model.cleaned_content),
        )

        return clean_model


class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> ChunkingDataHandler:
        if data_type == DataType.POSTS:
            return PostChunkingHandler()
        elif data_type == DataType.ARTICLES:
            return ArticleChunkingHandler()
        elif data_type == DataType.REPOSITORIES:
            return RepositoryChunkingHandler()
        else:
            raise ValueError("Unsupported data type")


class ChunkingDispatcher:
    cleaning_factory = ChunkingHandlerFactory

    @classmethod
    def dispatch(cls, data_model: DataModel) -> list[DataModel]:
        data_type = data_model.type
        handler = cls.cleaning_factory.create_handler(data_type)
        chunk_models = handler.chunk(data_model)

        logger.info(
            "Cleaned content chunked successfully.",
            num=len(chunk_models),
            data_type=data_type,
        )

        return chunk_models


class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> EmbeddingDataHandler:
        if data_type == DataType.POSTS:
            return PostEmbeddingHandler()
        elif data_type == DataType.ARTICLES:
            return ArticleEmbeddingHandler()
        elif data_type == DataType.REPOSITORIES:
            return RepositoryEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type")


class EmbeddingDispatcher:
    cleaning_factory = EmbeddingHandlerFactory

    @classmethod
    def dispatch(cls, data_model: DataModel) -> DataModel:
        data_type = data_model.type
        handler = cls.cleaning_factory.create_handler(data_type)
        embedded_chunk_model = handler.embedd(data_model)

        logger.info(
            "Chunk embedded successfully.",
            data_type=data_type,
            embedding_len=len(embedded_chunk_model.embedded_content),
        )

        return embedded_chunk_model
