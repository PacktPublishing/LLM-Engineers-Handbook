from zenml import pipeline

from llm_engineering.steps.digital_data_etl import crawl_links, get_or_create_user


@pipeline(enable_cache=False)
def digital_data_etl(user_id: str, links: list[str]) -> None:
    user = get_or_create_user(user_id)
    crawl_links(user=user, links=links)


if __name__ == "__main__":
    digital_data_etl.with_options(
        config_path="configs/digital_data_etl_paul_iusztin.yaml"
    )()
