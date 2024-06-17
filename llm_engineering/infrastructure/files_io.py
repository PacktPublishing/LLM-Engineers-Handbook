import json
from pathlib import Path


class JsonFileManager:
    @classmethod
    def read(cls, filename: str) -> list:
        filename: Path = Path(filename)
        
        try:
            with filename.open("r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename=}' does not exist.") from None
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                msg=f"File '{filename=}' is not properly formatted as JSON.",
                doc=e.doc,
                pos=e.pos,
            ) from None

    @classmethod
    def write(cls, filename: str, data: list) -> None:
        filename: Path = Path(filename)
        
        with filename.open("r") as file:
            json.dump(data, file, indent=4)
