from pathlib import Path

from typing_extensions import Annotated
from zenml import step

from llm_engineering.infrastructure.files_io import JsonFileManager


@step
def to_json(
    data: dict,
    file: Annotated[str, "file"],
) -> Annotated[Path, "output_file_path"]:
    absolute_file_path = JsonFileManager.write(
        filename=file,
        data=data,
    )

    return absolute_file_path
