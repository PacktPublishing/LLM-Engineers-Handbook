#!/usr/bin/env python3
"""
Fine-tune Llama models on Apple Silicon (M1/M2/M3/M4) using MPS
This script uses standard PyTorch, Transformers, and PEFT libraries.

Compatible with: M1, M2, M3, M4 Macs with macOS 12.3+
GPU Backend: Metal Performance Shaders (MPS)

Note: This is slower than Unsloth on NVIDIA GPUs but works on Mac.
For faster training, use the Colab notebook with CUDA GPUs.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


class Config:
    # Model settings
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fits on M4 Mac! Switch to "meta-llama/Meta-Llama-3.1-8B" if you have 32GB+ RAM
    max_seq_length = 2048

    # LoRA settings
    lora_r = 32
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Training settings
    num_train_epochs = 3
    per_device_train_batch_size = 1  # Reduced for Mac memory constraints
    gradient_accumulation_steps = 16  # Effective batch size = 16
    learning_rate = 3e-4
    warmup_steps = 10
    weight_decay = 0.01
    logging_steps = 10
    save_steps = 500
    eval_steps = 100

    # Dataset settings
    dataset1_name = "mlabonne/llmtwin"
    dataset2_name = "mlabonne/FineTome-Alpaca-100k"
    dataset2_split = "train[:10000]"
    test_size = 0.05

    # Output
    output_dir = "./output_mac"
    hub_model_id = None  # Set to "username/model-name" to push to HF Hub

    # Experiment tracking
    use_tensorboard = True  # Enable TensorBoard (local, free)
    use_comet_ml = False  # Set to True if you have Comet ML API key

    # Environment
    hf_token = os.getenv("HF_TOKEN")
    comet_api_key = os.getenv("COMET_API_KEY")  # Optional: for Comet ML tracking


# ============================================================================
# Helper Functions
# ============================================================================


def get_report_to(config: Config):
    """Determine which experiment tracking to use."""
    report_to = []

    if config.use_tensorboard:
        report_to.append("tensorboard")
        logger.info("✓ TensorBoard logging enabled")
        logger.info(f"  View with: tensorboard --logdir {config.output_dir}/runs")

    if config.use_comet_ml and config.comet_api_key:
        try:
            import comet_ml

            report_to.append("comet_ml")
            logger.info("✓ Comet ML logging enabled")
        except ImportError:
            logger.warning("⚠️  comet_ml not installed. Install with: pip install comet-ml")

    if not report_to:
        logger.info("ℹ️  No experiment tracking enabled")
        return "none"

    return report_to


def check_mps_availability():
    """Check if MPS (Metal Performance Shaders) is available."""
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.warning("MPS not available because PyTorch was not built with MPS enabled.")
        else:
            logger.warning("MPS not available because macOS version < 12.3 or device doesn't support it.")
        return False
    return True


def get_device():
    """Get the best available device (MPS > CPU)."""
    if torch.backends.mps.is_available():
        logger.info("✓ Using MPS (Metal Performance Shaders) - Apple Silicon GPU")
        return torch.device("mps")
    else:
        logger.warning("⚠️  MPS not available, using CPU (this will be slow)")
        return torch.device("cpu")


def load_model_and_tokenizer(config: Config):
    """Load model and tokenizer optimized for Mac."""
    logger.info(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in fp16 for better performance on MPS
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        token=config.hf_token,
        torch_dtype=torch.float16,  # Use fp16 for MPS
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False  # Required for gradient checkpointing
    model.config.pretraining_tp = 1

    logger.info(f"✓ Model loaded: {config.model_name}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def setup_lora(model, config: Config):
    """Configure LoRA adapters."""
    logger.info("Configuring LoRA adapters...")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"✓ LoRA configured:")
    logger.info(f"  Rank: {config.lora_r}")
    logger.info(f"  Alpha: {config.lora_alpha}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")

    return model


def load_and_prepare_dataset(tokenizer, config: Config):
    """Load and format datasets."""
    logger.info("Loading datasets...")

    # Load datasets
    dataset1 = load_dataset(config.dataset1_name, split="train")
    dataset2 = load_dataset(config.dataset2_name, split=config.dataset2_split)

    # Concatenate
    dataset = concatenate_datasets([dataset1, dataset2])

    logger.info(f"✓ Datasets loaded:")
    logger.info(f"  {config.dataset1_name}: {len(dataset1):,} samples")
    logger.info(f"  {config.dataset2_name}: {len(dataset2):,} samples")
    logger.info(f"  Total: {len(dataset):,} samples")

    # Format with Alpaca template
    alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

    eos_token = tokenizer.eos_token

    def format_samples(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = alpaca_template.format(instruction, output) + eos_token
            texts.append(text)
        return {"text": texts}

    # Apply formatting
    dataset = dataset.map(
        format_samples,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
    )

    # Split into train/test
    dataset = dataset.train_test_split(test_size=config.test_size, seed=42)

    logger.info(f"✓ Dataset formatted:")
    logger.info(f"  Train: {len(dataset['train']):,} samples")
    logger.info(f"  Test: {len(dataset['test']):,} samples")

    return dataset


# ============================================================================
# Main Training Function
# ============================================================================


def main():
    logger.info("=" * 70)
    logger.info("🍎 Fine-tuning on Apple Silicon (M4 Mac)")
    logger.info("=" * 70)

    config = Config()

    # Check MPS availability
    if not check_mps_availability():
        logger.error("MPS is required for efficient training on Mac")
        return

    device = get_device()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup LoRA
    model = setup_lora(model, config)

    # Load dataset
    dataset = load_and_prepare_dataset(tokenizer, config)

    # Training arguments optimized for Mac
    training_args = TrainingArguments(
        # Output
        output_dir=config.output_dir,
        # Training schedule
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Optimization
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        optim="adamw_torch",  # Use PyTorch's AdamW (MPS compatible)
        # Mixed precision for MPS
        fp16=False,  # MPS doesn't support fp16 training
        bf16=False,  # MPS doesn't support bf16 training
        # Logging and saving
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        save_total_limit=2,
        # Other
        gradient_checkpointing=True,
        report_to=get_report_to(config),
        seed=42,
        dataloader_num_workers=0,  # MPS works better with 0 workers
        remove_unused_columns=True,
        load_best_model_at_end=True,  # Load best checkpoint at end
        # Hub
        push_to_hub=config.hub_model_id is not None,
        hub_model_id=config.hub_model_id,
        hub_token=config.hf_token,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )

    logger.info("=" * 70)
    logger.info("🚀 Starting training...")
    logger.info("=" * 70)
    logger.info(f"  Device: {device}")
    logger.info(f"  Total epochs: {config.num_train_epochs}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Total train samples: {len(dataset['train']):,}")
    logger.info("=" * 70)

    # Show tracking info
    if config.use_tensorboard:
        logger.info("")
        logger.info("📊 Monitoring:")
        logger.info("   TensorBoard: Run 'tensorboard --logdir output_mac/runs'")
        logger.info("   Then open: http://localhost:6006")
    if config.use_comet_ml and config.comet_api_key:
        logger.info("   Comet ML: Check your dashboard at comet.ml")

    logger.info("")
    logger.info("⚠️  Note: Training on Mac will be slower than on CUDA GPUs")
    logger.info("   For faster training, use Google Colab or cloud GPU providers")
    logger.info("")

    # Train
    trainer.train()

    logger.info("=" * 70)
    logger.info("✅ Training complete!")
    logger.info("=" * 70)

    # Save final model
    logger.info("Saving model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"✓ Model saved to: {config.output_dir}")

    # Push to Hub if configured
    if config.hub_model_id:
        logger.info(f"Pushing to Hub: {config.hub_model_id}")
        trainer.push_to_hub()
        logger.info(f"✓ Model pushed to: https://huggingface.co/{config.hub_model_id}")

    # Test inference
    logger.info("=" * 70)
    logger.info("🧪 Testing inference...")
    logger.info("=" * 70)

    model.eval()
    test_prompt = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
What is supervised fine-tuning?

### Response:
"""

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated response:\n{response}")

    logger.info("=" * 70)
    logger.info("🎉 All done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
