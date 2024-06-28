from pathlib import Path

from zenml import pipeline
from zenml.client import Client

from llm_engineering.interfaces.orchestrator.steps import export as export_steps


@pipeline
def export_artifact_to_json(artifact_name: str, output_dir: Path = Path("output")) -> None:
    artifact = Client().get_artifact_version(name_id_or_prefix=artifact_name)

    data = export_steps.serialize_artifact(artifact=artifact)

    export_steps.to_json(data=data, to_file=output_dir / f"{artifact_name}.json")
