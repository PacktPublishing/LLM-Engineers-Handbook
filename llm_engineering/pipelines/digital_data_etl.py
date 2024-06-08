from zenml import pipeline

from llm_engineering.steps.digital_data_etl import crawl_links, get_or_create_user


@pipeline(enable_cache=False)
def digital_data_etl() -> None:
    user = get_or_create_user()
    crawl_links(user=user)


if __name__ == "__main__":
    digital_data_etl()
