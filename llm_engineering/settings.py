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

    # MongoDB
    DATABASE_HOST: str = "mongodb://decodingml:decodingml@llm_engineering_mongo:27017"
    DATABASE_NAME: str = "twin"

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None


settings = Settings()
