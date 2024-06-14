from pydantic import BaseSettings


class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # Selenium Drivers
    # TODO: How to handle binary path? Should we use a docker image for easily handling the chrome drivers?
    SELENIUM_BROWSER_BINARY_PATH: str | None = None
    # TODO: Generalize driver path. Still use docker image for the ETL pipeline?
    SELENIUM_BROWSER_DRIVER_PATH: str = "/Users/pauliusztin/.local/bin/chromedriver"

    # MongoDB NoSQL Database
    DATABASE_HOST: str = "mongodb://decodingml:decodingml@llm_engineering_mongo:27017"
    DATABASE_NAME: str = "twin"

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    # QdrantDB Vector DB
    USE_QDRANT_CLOUD: bool = False

    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_DATABASE_URL: str = "http://localhost:6333"

    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # OpenAI API
    OPENAI_MODEL_ID: str = "gpt-3.5-turbo"
    OPENAI_API_KEY: str | None = None

    # CometML config
    COMET_API_KEY: str | None = None
    COMET_WORKSPACE: str | None = None
    COMET_PROJECT: str | None = None


settings = Settings()
