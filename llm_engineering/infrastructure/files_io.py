import json
from pathlib import Path


class JsonFileManager:
    @classmethod
    def read(cls, filename: str | Path) -> list:
        file_path: Path = Path(filename)

        try:
            with file_path.open("r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path=}' does not exist.") from None
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                msg=f"File '{file_path=}' is not properly formatted as JSON.",
                doc=e.doc,
                pos=e.pos,
            ) from None

    @classmethod
    def write(cls, filename: str | Path, data: list | dict) -> Path:
        file_path: Path = Path(filename)
        file_path = file_path.resolve().absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

        return file_path
