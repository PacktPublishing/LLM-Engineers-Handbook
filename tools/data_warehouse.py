import json
from pathlib import Path

import click
from loguru import logger

from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.documents import ArticleDocument, PostDocument, RepositoryDocument, UserDocument


@click.command()
@click.option(
    "--export-raw-data",
    is_flag=True,
    default=False,
    help="Whether to export your data warehouse to a JSON file.",
)
@click.option(
    "--import-raw-data",
    is_flag=True,
    default=False,
    help="Whether to import a JSON file into your data warehouse.",
)
@click.option(
    "--data-dir",
    default=Path("data/data_warehouse_raw_data"),
    type=Path,
    help="Path to the directory containing data warehouse raw data JSON files.",
)
def main(
    export_raw_data,
    import_raw_data,
    data_dir: Path,
) -> None:
    assert export_raw_data or import_raw_data, "Specify at least one operation."

    if export_raw_data:
        __export(data_dir)

    if import_raw_data:
        __import(data_dir)


def __export(data_dir: Path) -> None:
    logger.info(f"Exporting data warehouse to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    __export_data_category(data_dir, ArticleDocument)
    __export_data_category(data_dir, PostDocument)
    __export_data_category(data_dir, RepositoryDocument)
    __export_data_category(data_dir, UserDocument)


def __export_data_category(data_dir: Path, category_class: type[NoSQLBaseDocument]) -> None:
    data = category_class.bulk_find()
    serialized_data = [d.to_mongo() for d in data]
    export_file = data_dir / f"{category_class.__name__}.json"

    logger.info(f"Exporting {len(serialized_data)} items of {category_class.__name__} to {export_file}...")
    with export_file.open("w") as f:
        json.dump(serialized_data, f)


def __import(data_dir: Path) -> None:
    logger.info(f"Importing data warehouse from {data_dir}...")
    assert data_dir.is_dir(), f"{data_dir} is not a directory or it doesn't exists."

    data_category_classes = {
        "ArticleDocument": ArticleDocument,
        "PostDocument": PostDocument,
        "RepositoryDocument": RepositoryDocument,
        "UserDocument": UserDocument,
    }

    for file in data_dir.iterdir():
        if not file.is_file():
            continue

        category_class_name = file.stem
        category_class = data_category_classes.get(category_class_name)
        if not category_class:
            logger.warning(f"Skipping {file} as it does not match any data category.")
            continue

        __import_data_category(file, category_class)


def __import_data_category(file: Path, category_class: type[NoSQLBaseDocument]) -> None:
    with file.open("r") as f:
        data = json.load(f)

    logger.info(f"Importing {len(data)} items of {category_class.__name__} from {file}...")
    if len(data) > 0:
        deserialized_data = [category_class.from_mongo(d) for d in data]
        category_class.bulk_insert(deserialized_data)


if __name__ == "__main__":
    main()
