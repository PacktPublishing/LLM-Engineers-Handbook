"""
Standalone Supervised Fine-Tuning Script using Unsloth

This script demonstrates how to fine-tune a large language model (LLM) using the Unsloth library,
which provides optimized training with reduced memory usage and faster training times.

WHAT WE'RE TRYING TO ACHIEVE:
=============================
The goal is to create a personalized AI assistant (LLM Twin) that mimics a specific writing style
and knowledge base. We do this through Supervised Fine-Tuning (SFT), where we:

1. Take a pre-trained base model (Meta-Llama-3.1-8B)
2. Fine-tune it on domain-specific instruction-response pairs
3. Use LoRA (Low-Rank Adaptation) to make training efficient
4. Create a model that can respond in a specific style and with specific knowledge

WHY UNSLOTH?
============
- Up to 2x faster training compared to standard methods
- Reduced memory footprint (can train larger models on smaller GPUs)
- Native support for LoRA/QLoRA
- Seamless integration with Hugging Face ecosystem
- Works on various GPUs: A40, A100, L4, etc.

TRAINING APPROACH:
==================
We use a two-dataset strategy:
1. llmtwin dataset (~3,000 samples): Domain-specific knowledge
2. FineTome-Alpaca-100k (10,000 samples): General instruction-following capability

This prevents overfitting and ensures the model learns both specific knowledge and
general instruction-following patterns.

PREREQUISITES:
==============
- CUDA-capable GPU (A40, A100, L4, or similar)
- Hugging Face account with access token (for gated models)
- Comet ML account for experiment tracking (optional)
- Required packages (see requirements.txt)

USAGE:
======
1. Set environment variables:
   HF_TOKEN=your_huggingface_token
   COMET_API_KEY=your_comet_api_key

2. Run the script:
   python unsloth_sft_standalone.py

3. Monitor training in Comet ML dashboard

4. Test the fine-tuned model

5. Push to Hugging Face Hub (optional)
"""

import os
import torch
from typing import Optional
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"  # Base model to fine-tune
MAX_SEQ_LENGTH = 2048  # Maximum sequence length for training
LOAD_IN_4BIT = False  # Use QLoRA (4-bit quantization) or LoRA (full precision)

# LoRA configuration
# LoRA adds small trainable matrices to the model instead of training all parameters
# This dramatically reduces memory usage and training time
LORA_RANK = 32  # Rank of LoRA matrices (higher = more capacity, more memory)
LORA_ALPHA = 32  # Scaling factor for LoRA updates
LORA_DROPOUT = 0  # Dropout for LoRA layers (0 = no dropout, faster training)
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",  # Attention projections
    "up_proj",
    "down_proj",  # Feed-forward network
    "o_proj",
    "gate_proj",  # Output and gating projections
]  # Apply LoRA to all linear layers for maximum quality

# Training hyperparameters
LEARNING_RATE = 3e-4  # Learning rate (3e-4 is a good starting point for LLM fine-tuning)
NUM_TRAIN_EPOCHS = 3  # Number of complete passes through the dataset
PER_DEVICE_BATCH_SIZE = 2  # Batch size per GPU
GRADIENT_ACCUMULATION_STEPS = 8  # Accumulate gradients over N steps (effective batch size = 2*8=16)
WEIGHT_DECAY = 0.01  # L2 regularization to prevent overfitting
WARMUP_STEPS = 10  # Gradual learning rate warmup
LR_SCHEDULER_TYPE = "linear"  # Learning rate schedule

# Dataset configuration
DATASET_WORKSPACE = "mlabonne"  # Hugging Face workspace for datasets
LLMTWIN_DATASET = f"{DATASET_WORKSPACE}/llmtwin"  # Domain-specific dataset
FINETOME_DATASET = "mlabonne/FineTome-Alpaca-100k"  # General instruction dataset
FINETOME_SAMPLES = 10000  # Number of samples to use from FineTome

# Output configuration
OUTPUT_DIR = "output"  # Directory for training outputs
MODEL_OUTPUT_DIR = "model_sft"  # Directory for final model
HF_REPO_ID = f"{DATASET_WORKSPACE}/TwinLlama-3.1-8B"  # Hugging Face repo for uploading

# Alpaca prompt template
# This template structures our instruction-response pairs in a consistent format
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


# ============================================================================
# STEP 1: AUTHENTICATION
# ============================================================================
def setup_authentication():
    """
    Set up authentication for Hugging Face and Comet ML.

    This is necessary to:
    - Access gated models (like Llama 3.1)
    - Upload fine-tuned models to Hugging Face Hub
    - Track experiments in Comet ML
    """
    print("=" * 80)
    print("STEP 1: Setting up authentication")
    print("=" * 80)

    # Check for Hugging Face token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("✓ Hugging Face token found")
    else:
        print("⚠ Warning: HF_TOKEN not set. You may not be able to access gated models.")
        print("  Set it with: export HF_TOKEN=your_token")

    # Check for Comet ML token
    comet_token = os.environ.get("COMET_API_KEY")
    if comet_token:
        print("✓ Comet ML API key found")
    else:
        print("⚠ Warning: COMET_API_KEY not set. Experiment tracking will be disabled.")
        print("  Set it with: export COMET_API_KEY=your_api_key")

    print()


# ============================================================================
# STEP 2: LOAD MODEL AND TOKENIZER
# ============================================================================
def load_model_and_tokenizer():
    """
    Load the base model and apply LoRA configuration.

    This step:
    1. Downloads the base model (Meta-Llama-3.1-8B)
    2. Configures it for efficient training with LoRA
    3. Only trains a small percentage of parameters (LoRA matrices)

    Returns:
        tuple: (model, tokenizer)
    """
    print("=" * 80)
    print("STEP 2: Loading model and configuring LoRA")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"Using 4-bit quantization: {LOAD_IN_4BIT}")
    print()

    # Load base model with Unsloth's optimized loader
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print("✓ Model loaded successfully")
    print()

    # Apply LoRA configuration
    print("Configuring LoRA...")
    print(f"  Rank: {LORA_RANK}")
    print(f"  Alpha: {LORA_ALPHA}")
    print(f"  Dropout: {LORA_DROPOUT}")
    print(f"  Target modules: {', '.join(TARGET_MODULES)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,  # Rank determines the dimension of LoRA matrices
        lora_alpha=LORA_ALPHA,  # Scaling factor for LoRA updates
        lora_dropout=LORA_DROPOUT,  # Dropout for regularization (0 = no dropout)
        target_modules=TARGET_MODULES,  # Which layers to apply LoRA to
    )

    print("✓ LoRA configured successfully")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    print()

    return model, tokenizer


# ============================================================================
# STEP 3: PREPARE DATASETS
# ============================================================================
def prepare_datasets(tokenizer):
    """
    Load and prepare training datasets.

    Strategy:
    1. Load llmtwin dataset (domain-specific, ~3,000 samples)
    2. Load FineTome dataset (general instructions, 10,000 samples)
    3. Concatenate them to prevent overfitting and maintain general capabilities
    4. Format using Alpaca template
    5. Split into train/test sets (95%/5%)

    Args:
        tokenizer: The model's tokenizer

    Returns:
        dict: Dataset with 'train' and 'test' splits
    """
    print("=" * 80)
    print("STEP 3: Preparing datasets")
    print("=" * 80)

    # Get EOS token for proper sequence termination
    EOS_TOKEN = tokenizer.eos_token
    print(f"EOS token: {EOS_TOKEN}")
    print()

    # Load domain-specific dataset
    print(f"Loading llmtwin dataset from {LLMTWIN_DATASET}...")
    dataset1 = load_dataset(LLMTWIN_DATASET, split="train")
    print(f"✓ Loaded {len(dataset1)} samples from llmtwin")

    # Load general instruction dataset
    print(f"Loading FineTome dataset (first {FINETOME_SAMPLES} samples)...")
    dataset2 = load_dataset(FINETOME_DATASET, split=f"train[:{FINETOME_SAMPLES}]")
    print(f"✓ Loaded {len(dataset2)} samples from FineTome")
    print()

    # Concatenate datasets
    print("Combining datasets...")
    dataset = concatenate_datasets([dataset1, dataset2])
    total_samples = len(dataset)
    print(f"✓ Total dataset size: {total_samples} samples")
    print(f"  - llmtwin: {len(dataset1)} ({100 * len(dataset1) / total_samples:.1f}%)")
    print(f"  - FineTome: {len(dataset2)} ({100 * len(dataset2) / total_samples:.1f}%)")
    print()

    # Format samples using Alpaca template
    print("Formatting samples with Alpaca template...")

    def format_samples(examples):
        """
        Format instruction-response pairs using the Alpaca template.

        Each sample is formatted as:
        ### Instruction: {instruction}
        ### Response: {output}<EOS>

        The EOS token is critical - it teaches the model when to stop generating.
        """
        text = []
        for instruction, output in zip(examples["instruction"], examples["output"], strict=False):
            message = ALPACA_TEMPLATE.format(instruction, output) + EOS_TOKEN
            text.append(message)
        return {"text": text}

    dataset = dataset.map(
        format_samples,
        batched=True,
        remove_columns=dataset.column_names,  # Remove original columns, keep only 'text'
    )
    print("✓ Samples formatted")
    print()

    # Split into train/test sets
    print("Splitting into train/test sets (95%/5%)...")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"✓ Training samples: {len(dataset['train'])}")
    print(f"✓ Test samples: {len(dataset['test'])}")
    print()

    # Show example
    print("Example training sample:")
    print("-" * 80)
    example_text = dataset["train"][0]["text"]
    # Truncate if too long
    if len(example_text) > 500:
        print(example_text[:500] + "...[truncated]")
    else:
        print(example_text)
    print("-" * 80)
    print()

    return dataset


# ============================================================================
# STEP 4: TRAIN THE MODEL
# ============================================================================
def train_model(model, tokenizer, dataset):
    """
    Train the model using Supervised Fine-Tuning (SFT).

    Training configuration:
    - Optimizer: AdamW 8-bit (memory efficient)
    - Learning rate: 3e-4 with linear decay
    - Batch size: 2 per device × 8 gradient accumulation = effective batch size of 16
    - Precision: BF16 if supported, otherwise FP16
    - Packing: Enabled (combines multiple samples to fill max_seq_length)

    Args:
        model: The LoRA-configured model
        tokenizer: The tokenizer
        dataset: Prepared dataset with train/test splits

    Returns:
        SFTTrainer: The trained trainer object
    """
    print("=" * 80)
    print("STEP 4: Training the model")
    print("=" * 80)

    # Determine precision
    use_bf16 = is_bfloat16_supported()
    precision = "BF16" if use_bf16 else "FP16"
    print(f"Using {precision} precision")
    print()

    # Display training configuration
    print("Training configuration:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"  Batch size per device: {PER_DEVICE_BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  LR scheduler: {LR_SCHEDULER_TYPE}")
    print()

    # Create trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",  # Field containing the formatted text
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,  # Number of processes for data loading
        packing=True,  # Pack multiple samples to fill max_seq_length efficiently
        args=TrainingArguments(
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            # Precision settings
            fp16=not use_bf16,  # Use FP16 if BF16 not supported
            bf16=use_bf16,  # Use BF16 if supported (better for training)
            # Optimizer settings
            optim="adamw_8bit",  # 8-bit AdamW (memory efficient)
            weight_decay=WEIGHT_DECAY,
            # Learning rate schedule
            warmup_steps=WARMUP_STEPS,
            # Logging and saving
            logging_steps=1,  # Log every step
            output_dir=OUTPUT_DIR,
            report_to="comet_ml",  # Track experiments in Comet ML
            # Reproducibility
            seed=42,
        ),
    )
    print("✓ Trainer initialized")
    print()

    # Start training
    print("Starting training...")
    print("This may take a while depending on your GPU.")
    print("Expected time on A100: ~50 minutes for 3 epochs")
    print()
    print("You can monitor training in Comet ML dashboard")
    print("-" * 80)

    trainer.train()

    print()
    print("-" * 80)
    print("✓ Training complete!")
    print()

    return trainer


# ============================================================================
# STEP 5: TEST THE MODEL
# ============================================================================
def test_model(model, tokenizer):
    """
    Test the fine-tuned model with a sample prompt.

    This is a quick sanity check to ensure:
    1. The model can generate text
    2. It follows the Alpaca format
    3. It generates coherent responses
    4. It properly uses the EOS token to stop

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
    """
    print("=" * 80)
    print("STEP 5: Testing the fine-tuned model")
    print("=" * 80)

    # Enable fast inference mode
    print("Switching to inference mode...")
    FastLanguageModel.for_inference(model)
    print("✓ Model ready for inference")
    print()

    # Prepare test prompt
    test_instruction = "Write a paragraph to introduce supervised fine-tuning."
    print(f"Test prompt: {test_instruction}")
    print()

    # Format with Alpaca template (empty response to prompt the model)
    message = ALPACA_TEMPLATE.format(test_instruction, "")
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    # Generate with streaming
    print("Generated response:")
    print("-" * 80)
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )
    print("-" * 80)
    print()


# ============================================================================
# STEP 6: SAVE AND UPLOAD MODEL
# ============================================================================
def save_and_upload_model(model, tokenizer, push_to_hub: bool = False):
    """
    Save the fine-tuned model locally and optionally upload to Hugging Face Hub.

    The model is saved in 16-bit precision (merged with LoRA weights) for inference.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        push_to_hub: Whether to upload to Hugging Face Hub
    """
    print("=" * 80)
    print("STEP 6: Saving model")
    print("=" * 80)

    # Save locally
    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained_merged(
        MODEL_OUTPUT_DIR,
        tokenizer,
        save_method="merged_16bit",  # Merge LoRA weights and save in 16-bit
    )
    print(f"✓ Model saved to {MODEL_OUTPUT_DIR}/")
    print()

    # Upload to Hugging Face Hub
    if push_to_hub:
        print(f"Uploading model to Hugging Face Hub: {HF_REPO_ID}")
        print("This may take several minutes...")

        try:
            model.push_to_hub_merged(HF_REPO_ID, tokenizer, save_method="merged_16bit")
            print(f"✓ Model uploaded successfully!")
            print(f"  View at: https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"✗ Failed to upload: {e}")
            print("  Make sure HF_TOKEN is set and you have write access to the repo")
    else:
        print("Skipping upload to Hugging Face Hub")
        print(f"To upload later, use: model.push_to_hub_merged('{HF_REPO_ID}', tokenizer)")

    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main execution function that orchestrates the entire fine-tuning pipeline.

    Pipeline:
    1. Authentication setup
    2. Load model and configure LoRA
    3. Prepare datasets
    4. Train the model
    5. Test the model
    6. Save and optionally upload
    """
    print("\n")
    print("=" * 80)
    print(" " * 15 + "SUPERVISED FINE-TUNING WITH UNSLOTH")
    print("=" * 80)
    print()
    print("This script will fine-tune Meta-Llama-3.1-8B using LoRA")
    print("to create a personalized AI assistant (LLM Twin)")
    print()

    try:
        # Step 1: Setup authentication
        setup_authentication()

        # Step 2: Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()

        # Step 3: Prepare datasets
        dataset = prepare_datasets(tokenizer)

        # Step 4: Train the model
        trainer = train_model(model, tokenizer, dataset)

        # Step 5: Test the model
        test_model(model, tokenizer)

        # Step 6: Save and upload
        save_and_upload_model(
            model,
            tokenizer,
            push_to_hub=False,  # Set to True to upload to Hugging Face Hub
        )

        print("=" * 80)
        print(" " * 20 + "FINE-TUNING COMPLETE!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Review training metrics in Comet ML")
        print("2. Test the model with more prompts")
        print("3. Upload to Hugging Face Hub if satisfied")
        print("4. Deploy for inference")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback

        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
