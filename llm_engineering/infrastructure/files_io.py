import json


class JsonFileManager:
    @classmethod
    def read(cls, filename: str) -> list:
        try:
            with open(filename, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename=}' does not exist.")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                msg=f"File '{filename=}' is not properly formatted as JSON.",
                doc=e.doc,
                pos=e.pos,
            )

    @classmethod
    def write(cls, filename: str, data: list):
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
