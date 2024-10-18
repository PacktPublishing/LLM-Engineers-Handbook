from pathlib import Path

from sagemaker.huggingface import HuggingFace

from llm_engineering.settings import settings

# Set up paths
finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"

# Verify that the necessary files exist
if not finetuning_requirements_path.exists():
    raise FileNotFoundError(f"The file {finetuning_requirements_path} does not exist.")

# Create the HuggingFace estimator
huggingface_estimator = HuggingFace(
    entry_point="finetune.py",
    source_dir=str(finetuning_dir),
    instance_type="ml.g5.12xlarge",
    instance_count=1,
    role=settings.AWS_ARN_ROLE,
    transformers_version="4.36",
    pytorch_version="2.1",
    py_version="py310",
    hyperparameters={
        "epochs": 1,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "learning_rate": 3e-4,
        "model_id": "meta-llama/Meta-Llama-3.1-8B",
        "use_qlora": True,
        "max_seq_length": 512,
    },
    requirements_file=finetuning_requirements_path,
    environment={
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "COMET_API_KEY": settings.COMET_API_KEY,
        "COMET_WORKSPACE": settings.COMET_WORKSPACE,
        "COMET_PROJECT": settings.COMET_PROJECT,
    },
)

# Start the training job
huggingface_estimator.fit()
