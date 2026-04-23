# Quick Start Guide: Fine-Tuning with Unsloth

## Overview

This guide will help you quickly get started with fine-tuning Meta-Llama-3.1-8B using Unsloth to create your own LLM Twin.

**Time required**: ~1-2 hours (including setup)  
**GPU required**: A100, A40, L4, or similar (24GB+ VRAM)  
**Cost estimate**: $2-5 on cloud GPU services

---

## Step 1: Environment Setup (10 minutes)

### Option A: Local GPU

```bash
# Create conda environment
conda create -n unsloth python=3.10 -y
conda activate unsloth

# Install dependencies
pip install torch transformers datasets trl peft accelerate

# Install Unsloth (for CUDA 12.1+)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install optional dependencies
pip install comet-ml python-dotenv bitsandbytes
```

### Option B: Google Colab

```python
# Run in Colab notebook
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install transformers datasets trl comet-ml
```

### Option C: Lambda Labs / RunPod

```bash
# Usually comes with PyTorch pre-installed
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets trl comet-ml
```

---

## Step 2: Get API Keys (5 minutes)

### Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "Create new token"
3. Give it a name and select "Read" permissions
4. Copy the token

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

### Accept Llama 3.1 License

1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
2. Click "Agree and access repository"
3. Wait for approval (usually instant)

### Comet ML Key (Optional)

1. Go to https://www.comet.com/signup
2. Create account (free tier available)
3. Go to Settings → API Keys
4. Copy your API key

```bash
export COMET_API_KEY="your_comet_api_key"
```

---

## Step 3: Prepare Your Data (15 minutes)

### Format Requirements

Your dataset should have two columns:
- `instruction`: The input prompt/question
- `output`: The expected response

Example:
```json
{
  "instruction": "Explain what is machine learning.",
  "output": "Machine learning is a subset of artificial intelligence..."
}
```

### Option A: Use Existing Datasets

The script uses:
1. `mlabonne/llmtwin` - Domain-specific (3K samples)
2. `mlabonne/FineTome-Alpaca-100k` - General (10K samples)

**No changes needed!** Just run the script.

### Option B: Use Your Own Dataset

1. **Prepare your data**:
```python
# Create a JSONL file with your data
import json

data = [
    {"instruction": "Question 1", "output": "Answer 1"},
    {"instruction": "Question 2", "output": "Answer 2"},
    # ... more examples
]

with open("my_data.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
```

2. **Upload to Hugging Face** (or load locally):
```python
from datasets import load_dataset

# Load local file
dataset = load_dataset("json", data_files="my_data.jsonl")

# Or upload to HF and load
# dataset = load_dataset("your-username/your-dataset")
```

3. **Modify the script**:
```python
# In unsloth_sft_standalone.py, replace:
dataset1 = load_dataset("mlabonne/llmtwin", split="train")

# With:
dataset1 = load_dataset("your-username/your-dataset", split="train")
# Or:
dataset1 = load_dataset("json", data_files="my_data.jsonl", split="train")
```

---

## Step 4: Run Training (50 minutes on A100)

### Basic Usage

```bash
cd /path/to/LLM-Engineers-Handbook/llm_engineering/model/finetuning
python unsloth_sft_standalone.py
```

### What You'll See

```
================================================================================
                   SUPERVISED FINE-TUNING WITH UNSLOTH
================================================================================

STEP 1: Setting up authentication
✓ Hugging Face token found
✓ Comet ML API key found

STEP 2: Loading model and configuring LoRA
Loading base model...
✓ Model loaded successfully
✓ LoRA configured successfully
Trainable parameters: 167,772,160 (1.63%)

STEP 3: Preparing datasets
✓ Loaded 3,000 samples from llmtwin
✓ Loaded 10,000 samples from FineTome
✓ Total dataset size: 13,000 samples
✓ Training samples: 12,350
✓ Test samples: 650

STEP 4: Training the model
Using BF16 precision
Starting training...
Expected time on A100: ~50 minutes

[Training progress bars and metrics...]

✓ Training complete!

STEP 5: Testing the fine-tuned model
Generated response:
--------------------------------------------------------------------------------
Supervised fine-tuning is a method used to enhance a language model
by providing it with a curated dataset of instructions and their
corresponding answers. This process is designed to align the model's
responses with human expectations...
--------------------------------------------------------------------------------

STEP 6: Saving model
✓ Model saved to model_sft/

================================================================================
                        FINE-TUNING COMPLETE!
================================================================================
```

---

## Step 5: Monitor Training (During Training)

### Comet ML Dashboard

1. Go to https://www.comet.com/
2. Navigate to your project
3. View metrics:
   - **Training Loss**: Should decrease steadily
   - **Validation Loss**: Should decrease then plateau
   - **Learning Rate**: Should decay linearly
   - **GPU Memory**: Should be consistently high

### Expected Loss Curves

```
Training Loss:    Validation Loss:
2.5 ┐              2.4 ┐
    │\                 │\
2.0 │ \                │ \
    │  \               │  \
1.5 │   \_             │   \_
    │     \_           │     \_
1.0 │       \_         │       \_
    │         \_       │         ──
0.5 │           ─      │
    └───────────────   └───────────────
    Epoch 1  2  3      Epoch 1  2  3
```

### Troubleshooting During Training

| Issue | Solution |
|-------|----------|
| Loss is NaN | Stop training, reduce learning rate to 1e-4 |
| OOM Error | Reduce batch size to 1, increase grad accumulation |
| Very slow | Check GPU utilization (should be >90%) |
| Loss not decreasing | Verify data format, check learning rate |

---

## Step 6: Test Your Model (10 minutes)

### Quick Test (Already in Script)

The script automatically tests with:
```python
prompt = "Write a paragraph to introduce supervised fine-tuning."
```

### Manual Testing

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load your fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="model_sft",  # Your saved model
    max_seq_length=2048,
    load_in_4bit=False,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Prepare prompt
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

message = alpaca_template.format("Your question here", "")
inputs = tokenizer([message], return_tensors="pt").to("cuda")

# Generate
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
outputs = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
```

### Test Cases to Try

1. **Style consistency**:
   ```
   "Write a blog post introduction about AI."
   ```

2. **Domain knowledge**:
   ```
   "Explain the difference between LoRA and full fine-tuning."
   ```

3. **Instruction following**:
   ```
   "List 3 benefits of using Unsloth for fine-tuning."
   ```

4. **Edge cases**:
   ```
   "What is quantum computing?"  # Out of domain
   ```

---

## Step 7: Upload to Hugging Face (5 minutes)

### Modify Script

In `unsloth_sft_standalone.py`, change:

```python
save_and_upload_model(
    model,
    tokenizer,
    push_to_hub=True  # Change from False to True
)
```

### Or Upload Manually

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="model_sft",
    max_seq_length=2048,
)

# Upload
model.push_to_hub_merged(
    "your-username/TwinLlama-3.1-8B",
    tokenizer,
    save_method="merged_16bit"
)
```

### Verify Upload

1. Go to https://huggingface.co/your-username/TwinLlama-3.1-8B
2. Check that model files are present:
   - `config.json`
   - `model-*.safetensors`
   - `tokenizer.json`
   - `tokenizer_config.json`

---

## Quick Reference

### File Structure
```
llm_engineering/model/finetuning/
├── unsloth_sft_standalone.py    # Main training script
├── FINE_TUNING_GUIDE.md         # Comprehensive guide
├── QUICK_START.md               # This file
├── requirements-unsloth.txt     # Dependencies
├── finetune.py                  # Existing codebase integration
└── sagemaker.py                 # SageMaker deployment
```

### Key Commands

```bash
# Setup
conda activate unsloth
export HF_TOKEN="your_token"
export COMET_API_KEY="your_key"

# Train
python unsloth_sft_standalone.py

# Monitor
# Visit comet.com dashboard

# Test
python -c "from unsloth import FastLanguageModel; ..."
```

### Configuration Quick Edit

To change key parameters, edit these lines in `unsloth_sft_standalone.py`:

```python
# Line 62: Model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

# Line 68: LoRA rank (higher = more capacity)
LORA_RANK = 32  # Try 64 or 128 for better quality

# Line 75: Learning rate
LEARNING_RATE = 3e-4  # Lower if unstable

# Line 76: Epochs
NUM_TRAIN_EPOCHS = 3  # Increase if underfitting

# Line 87: Your datasets
LLMTWIN_DATASET = f"{DATASET_WORKSPACE}/llmtwin"
```

---

## Common Issues and Solutions

### 1. "CUDA out of memory"

```python
# Reduce batch size
PER_DEVICE_BATCH_SIZE = 1  # Line 77

# Increase gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 16  # Line 78

# Or use QLoRA
LOAD_IN_4BIT = True  # Line 64
```

### 2. "Model not found" or "Authentication failed"

```bash
# Re-login
huggingface-cli login

# Or set token explicitly
export HF_TOKEN="hf_your_token"
```

### 3. Training too slow

```python
# Reduce sequence length
MAX_SEQ_LENGTH = 1024  # Line 63

# Use fewer samples for testing
FINETOME_SAMPLES = 5000  # Line 91
```

### 4. Model generates nonsense

```python
# Reduce learning rate
LEARNING_RATE = 1e-4  # Line 75

# Add dropout
LORA_DROPOUT = 0.1  # Line 70

# Check data formatting
# Ensure EOS token is added
```

---

## Next Steps

1. **Evaluate thoroughly**: Test on diverse prompts
2. **Iterate**: Collect failure cases, add to training data
3. **Optimize**: Quantize for smaller size (GPTQ, AWQ)
4. **Deploy**: Use vLLM or HF TGI for production
5. **DPO**: Further refine with preference data (optional)

---

## Cost Estimates

| GPU | Provider | Cost/Hour | Training Time | Total Cost |
|-----|----------|-----------|---------------|------------|
| A100 (40GB) | Lambda Labs | $1.10 | 50 min | ~$0.92 |
| A100 (80GB) | RunPod | $1.89 | 50 min | ~$1.57 |
| A40 | Lambda Labs | $0.75 | 80 min | ~$1.00 |
| L4 | Google Cloud | $0.70 | 120 min | ~$1.40 |

**Note**: Prices and times are approximate and may vary.

---

## Getting Help

- **Issues with code**: Check [GitHub Issues](https://github.com/PacktPublishing/LLM-Engineering/issues)
- **Unsloth problems**: [Unsloth GitHub](https://github.com/unslothai/unsloth)
- **General questions**: [Hugging Face Forums](https://discuss.huggingface.co/)
- **Community**: r/LocalLLaMA on Reddit

---

## Success Checklist

- [ ] Environment set up correctly
- [ ] API keys configured
- [ ] Model access granted
- [ ] Data prepared (or using defaults)
- [ ] Training completed without errors
- [ ] Training loss decreased steadily
- [ ] Validation loss reasonable (<1.5)
- [ ] Model generates coherent text
- [ ] Model follows instructions
- [ ] Model saved successfully
- [ ] (Optional) Model uploaded to HF

---

**Congratulations! You've successfully fine-tuned your first LLM!** 🎉

For more details, see [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md)
