from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger

try:
    from sagemaker.huggingface import HuggingFace
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering import settings

evaluation_dir = Path(__file__).resolve().parent
evaluation_requirements_path = evaluation_dir / "requirements.txt"

# Hard cap on job runtime so a hung job can't keep billing (Training jobs default to 24h).
MAX_RUN_SECONDS = 4 * 60 * 60


def run_evaluation_on_sagemaker(is_dummy: bool = True) -> None:
    """Run evaluate.py on SageMaker.

    It is launched as a SageMaker *Training* job (not a Processing job) because many
    accounts have a 0-instance quota for 'ml.g5.2xlarge for processing job usage' but
    can already use the same instance for training jobs. The script itself does not
    train anything: it generates answers, has an LLM judge score them, and pushes the
    results to the Hugging Face Hub.
    """

    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.OPENAI_API_KEY, "OpenAI API key is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    if not evaluation_dir.exists():
        raise FileNotFoundError(f"The directory {evaluation_dir} does not exist.")
    if not evaluation_requirements_path.exists():
        raise FileNotFoundError(f"The file {evaluation_requirements_path} does not exist.")

    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    env = {
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "OPENAI_BASE_URL": settings.OPENAI_BASE_URL,
        "OPENAI_MODEL_ID": settings.OPENAI_MODEL_ID,
        "DATASET_HUGGINGFACE_WORKSPACE": huggingface_user,
        "MODEL_HUGGINGFACE_WORKSPACE": huggingface_user,
    }
    if is_dummy:
        env["IS_DUMMY"] = "True"
    # SageMaker only accepts string values; drop unset settings (evaluate.py has defaults).
    env = {key: str(value) for key, value in env.items() if value}

    huggingface_estimator = HuggingFace(
        entry_point="evaluate.py",
        source_dir=str(evaluation_dir),
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        role=settings.AWS_ARN_ROLE,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        base_job_name="evaluate-llm-twin",
        max_run=MAX_RUN_SECONDS,
        environment=env,
    )

    # Starts the job and streams its logs (including the final scores) to this terminal.
    huggingface_estimator.fit()


if __name__ == "__main__":
    run_evaluation_on_sagemaker()
