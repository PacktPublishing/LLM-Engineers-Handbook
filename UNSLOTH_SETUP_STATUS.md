# Unsloth Setup Status

## ✅ Installed Dependencies (macOS Compatible)

All core dependencies for Unsloth fine-tuning have been successfully installed:

| Package | Version | Status |
|---------|---------|--------|
| transformers | 4.57.3 | ✅ Installed |
| datasets | 4.4.2 | ✅ Installed |
| trl | 0.26.2 | ✅ Installed |
| peft | 0.18.0 | ✅ Installed |
| accelerate | 1.12.0 | ✅ Installed |
| torch | 2.5.1 | ✅ Installed |
| python-dotenv | 1.2.1 | ✅ Installed |
| scipy | 1.16.3 | ✅ Installed |
| tqdm | 4.67.1 | ✅ Installed |

## ⚠️ Unsloth Installation Status

**Unsloth is NOT installed** on your local macOS system.

### Why?

Unsloth requires:
- **NVIDIA GPU** with CUDA support
- CUDA 11.8+ or CUDA 12.1+
- Linux or Windows with proper CUDA drivers

Your system:
- **Platform**: macOS (darwin) - M4 Apple Silicon
- **GPU Backend**: MPS (Metal Performance Shaders) ✅
- **CUDA Available**: ❌ No (not compatible with Apple Silicon)

### This Means:

1. ✅ **You can develop and test locally** with all other libraries
2. ✅ **You can prepare datasets and configurations**
3. ✅ **You can read/edit training scripts**
4. ❌ **You CANNOT run Unsloth training** (requires NVIDIA GPU)
5. ✅ **You CAN run standard PyTorch/PEFT training** using MPS

## 🍎 Apple Silicon (M4) Alternative

### Option: Use Standard PyTorch with MPS

**Script**: `tools/finetune_mac.py`

This script uses standard libraries (PyTorch, Transformers, PEFT) that **DO support M4 Mac**:
- ✅ Uses Metal Performance Shaders (MPS) for GPU acceleration
- ✅ Compatible with M1, M2, M3, M4 Macs
- ✅ LoRA fine-tuning with all features
- ⚠️ Slower than Unsloth on NVIDIA GPUs (but much faster than CPU)

**Run locally on M4 Mac:**
```bash
export HF_TOKEN="your_token_here"
python tools/finetune_mac.py
```

**Performance:**
- M4 Mac: ~4-6 hours for full training (vs ~50 min on A100)
- Still practical for development and small-scale fine-tuning
- Uses much less power than cloud GPU solutions

## 🚀 Training Options

### Option 1: Run Locally on M4 Mac (Slowest but Convenient)

**Use this if**: You want to train locally without cloud costs

**Script**: `tools/finetune_mac.py`
**Cost**: FREE (uses your Mac's M4 GPU)
**Training Time**: ~4-6 hours for full training
**Memory**: Requires ~32GB RAM recommended

```bash
# Set your HuggingFace token
export HF_TOKEN="your_token_here"

# Run training
python tools/finetune_mac.py
```

**Pros:**
- ✅ No cloud costs
- ✅ Privacy - data stays on your machine
- ✅ Good for development and testing

**Cons:**
- ⚠️ 5-10x slower than cloud GPUs
- ⚠️ Higher power consumption
- ⚠️ Not using Unsloth optimizations

### Option 2: Google Colab (Fastest Free Option)

**Ready-to-use notebook**: `notebooks/unsloth_finetuning.ipynb`

**Steps:**
1. Upload the notebook to Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU (free tier)
3. Run all cells

**Cost**: FREE (with T4 GPU)
**Training Time**: ~2-3 hours for full training

### Option 2: Cloud GPU Providers (Fast & Flexible)

**Providers:**
- **RunPod**: https://runpod.io (~$0.39/hr for A100)
- **Lambda Labs**: https://lambdalabs.com (~$1.10/hr for A100)
- **Vast.ai**: https://vast.ai (cheapest, ~$0.20/hr for A100)

**Steps:**
1. Rent a GPU instance with CUDA
2. Clone this repository
3. Install dependencies:
   ```bash
   pip install -r llm_engineering/model/finetuning/requirements-unsloth.txt
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```
4. Run the training script

**Cost**: $0.20-$1.10/hour depending on GPU
**Training Time**: ~50 minutes on A100

### Option 3: AWS SageMaker (Production)

The repo already has a ZenML pipeline configured for AWS SageMaker.

**Configuration**: `configs/training.yaml`

**Steps:**
1. Set up AWS credentials and ZenML
2. Configure SageMaker orchestrator
3. Run the pipeline:
   ```bash
   python pipelines/training.py
   ```

**Cost**: AWS SageMaker pricing (varies)
**Training Time**: ~50 minutes on ml.g5.xlarge

## 📋 Available Files

### Training Scripts
- ✅ `notebooks/unsloth_finetuning.ipynb` - Google Colab notebook (RECOMMENDED)
- ✅ `llm_engineering/model/finetuning/requirements-unsloth.txt` - Dependencies list
- ✅ `configs/training.yaml` - ZenML pipeline config

### Documentation Created
- ✅ Standalone fine-tuning script with detailed comments
- ✅ Step-by-step implementation guide
- ✅ Conceptual overview of SFT and LoRA
- ✅ Quick start guide
- ✅ Comparison between approaches

## 🎯 Next Steps

### For Immediate Training:

**Use Google Colab** (fastest path):
```bash
# 1. Open the notebook
open notebooks/unsloth_finetuning.ipynb

# 2. Upload to Google Colab
# 3. Enable GPU runtime
# 4. Run all cells
```

### For Local Development:

You can continue developing locally:
- ✅ Prepare datasets
- ✅ Modify configurations
- ✅ Test data preprocessing
- ✅ Create custom prompts/templates

When ready to train, upload to a GPU environment.

## 🔧 Verification Commands

Check installed dependencies:
```bash
python -c "import transformers, datasets, trl, peft, accelerate; print('All dependencies OK')"
```

Check GPU availability:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

## 📚 Reference Links

- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **LLM Engineering Handbook**: https://github.com/PacktPublishing/LLM-Engineering
- **HuggingFace Hub**: https://huggingface.co/
- **Google Colab**: https://colab.research.google.com/

## ⚡ Quick Start Command

To start training on Google Colab right now:
1. Go to https://colab.research.google.com/
2. Upload `notebooks/unsloth_finetuning.ipynb`
3. Runtime → Change runtime type → GPU (T4)
4. Runtime → Run all

Done! Your model will be training in minutes. 🚀
