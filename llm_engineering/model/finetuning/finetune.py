import argparse
import os
import warnings

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def model_fn(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def setup_torch_config():
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        try:
            import flash_attn
            from packaging import version

            if version.parse(flash_attn.__version__) >= version.parse("2.1.0"):
                attn_implementation = "flash_attention_2"
                print("Using FlashAttention 2")  # noqa
            else:
                attn_implementation = "eager"
                print(f"FlashAttention version {flash_attn.__version__} is not compatible. Using eager implementation.")  # noqa
        except ImportError:
            warnings.warn("FlashAttention not installed. Defaulting to eager attention.")  # noqa
            attn_implementation = "eager"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"
    return torch_dtype, attn_implementation


def load_model_and_tokenizer(args, torch_dtype, attn_implementation):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    return model, tokenizer, peft_config


def train(args):
    torch_dtype, attn_implementation = setup_torch_config()
    model, tokenizer, peft_config = load_model_and_tokenizer(args, torch_dtype, attn_implementation)

    # Load dataset
    dataset = load_from_disk(args.data_dir)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_dir=f"{args.output_data_dir}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--use_qlora", type=bool, default=False)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Data, model, and output directories
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    args = parser.parse_args()

    train(args)
