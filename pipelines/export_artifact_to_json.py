from pathlib import Path

from zenml import pipeline
from zenml.client import Client

from steps import export as export_steps


@pipeline
def export_artifact_to_json(artifact_names: list[str], output_dir: Path = Path("output")) -> None:
    for artifact_name in artifact_names:
        artifact = Client().get_artifact_version(name_id_or_prefix=artifact_name)

        data = export_steps.serialize_artifact(artifact=artifact, artifact_name=artifact_name)

        export_steps.to_json(data=data, to_file=output_dir / f"{artifact_name}.json")
