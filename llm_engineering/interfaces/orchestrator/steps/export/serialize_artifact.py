from typing import Any

from pydantic import BaseModel
from typing_extensions import Annotated
from zenml import step


@step
def serialize_artifact(
    artifact: Any,
) -> Annotated[dict, "serialized_artifact"]:
    serialized_artifact = _serialize_artifact(artifact)

    if serialize_artifact is None:
        raise ValueError("Artifact is None")
    elif not isinstance(serialized_artifact, dict):
        serialized_artifact = {"artifact": serialized_artifact}

    return serialized_artifact


def _serialize_artifact(arfifact: list | dict | BaseModel | str | int | float | bool | None):
    if isinstance(arfifact, list):
        return [_serialize_artifact(item) for item in arfifact]
    elif isinstance(arfifact, dict):
        return {key: _serialize_artifact(value) for key, value in arfifact.items()}
    if isinstance(arfifact, BaseModel):
        return arfifact.dict()
    else:
        return arfifact
