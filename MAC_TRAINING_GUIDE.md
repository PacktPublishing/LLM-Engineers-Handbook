# Fine-Tuning on M4 Mac - Quick Start Guide

## ✅ Your M4 Mac Setup Status

**GPU Detection**: ✓ MPS (Metal Performance Shaders) available
**PyTorch Version**: 2.5.1
**GPU Acceleration**: Enabled and working

## 🎯 Two Approaches Available

### Approach 1: Local Training on M4 Mac (Recommended for You)

**Best for**: Development, testing, small-scale fine-tuning, privacy

```bash
# 1. Set your HuggingFace token (if needed for gated models)
export HF_TOKEN="your_token_here"

# 2. Run the M4-optimized training script
python tools/finetune_mac.py
```

**What happens**:
- Uses your M4 GPU via Metal Performance Shaders
- Fine-tunes Llama 3.1 8B with LoRA
- Trains on ~13,000 samples (llmtwin + FineTome)
- Saves model to `./output_mac/`

**Expected Performance**:
- Training time: ~4-6 hours (full 3 epochs)
- Memory usage: ~20-30GB RAM
- GPU utilization: High (your Mac will be warm!)

**Customization**:
Edit `tools/finetune_mac.py` and modify the `Config` class:
```python
class Config:
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    num_train_epochs = 3  # Reduce to 1 for faster testing
    per_device_train_batch_size = 1  # Keep at 1 for M4
    lora_r = 32  # LoRA rank (8-64 typical range)
```

---

### Approach 2: Cloud GPU with Unsloth (Faster)

**Best for**: Production training, faster iterations, using Unsloth optimizations

Use the **Google Colab notebook**: `notebooks/unsloth_finetuning.ipynb`

**Advantages**:
- 5-10x faster (50 min on A100 vs 4-6 hours on M4)
- Uses Unsloth optimizations
- Lower power consumption
- No wear on your Mac

**Trade-offs**:
- Requires internet connection
- Data uploaded to cloud
- May have costs (free tier available)

---

## 📊 Detailed Comparison

| Feature | M4 Mac Training | Cloud GPU (Unsloth) |
|---------|----------------|---------------------|
| **Speed** | ~4-6 hours | ~50 min (A100) |
| **Cost** | Free | $0-2 per run |
| **Privacy** | ✅ Local | ❌ Cloud |
| **Setup** | Instant | 5 min |
| **Power** | ~20W | Data center |
| **Interruptions** | Can pause | Must complete |
| **Development** | ✅ Ideal | ⚠️ Network dependent |

---

## 🚀 Running on M4 Mac - Step by Step

### 1. Verify Setup

```bash
python verify_setup.py
```

You should see:
- ✅ All dependencies installed
- ✅ MPS Support: Available
- ✅ Ready for M4 Mac training

### 2. (Optional) Test with Small Dataset

Edit `tools/finetune_mac.py` and change:
```python
num_train_epochs = 1  # Quick test
dataset2_split = "train[:1000]"  # Smaller dataset
```

### 3. Run Training

```bash
# Set HuggingFace token (for Llama access)
export HF_TOKEN="hf_your_token_here"

# Start training
python tools/finetune_mac.py
```

### 4. Monitor Progress

Training output will show:
```
🍎 Fine-tuning on Apple Silicon (M4 Mac)
======================================================================
✓ Using MPS (Metal Performance Shaders) - Apple Silicon GPU
✓ Model loaded: meta-llama/Meta-Llama-3.1-8B
  Total parameters: 8,030,261,248
✓ LoRA configured:
  Rank: 32
  Trainable params: 41,943,040 (0.5222%)
✓ Dataset formatted:
  Train: 12,350 samples
  Test: 650 samples
======================================================================
🚀 Starting training...
```

### 5. While Training

**Activity Monitor**:
- Open Activity Monitor
- Check "GPU History" - should show high usage
- Memory pressure should be yellow/green

**Expected behavior**:
- First epoch: ~1.5-2 hours
- Subsequent epochs: Similar duration
- Mac will get warm (this is normal)
- Fan may spin up (also normal)

### 6. After Training

Model saved to: `./output_mac/`

**Test the model**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="mps"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./output_mac")
tokenizer = AutoTokenizer.from_pretrained("./output_mac")

# Test
prompt = "What is supervised fine-tuning?"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

---

## 🔧 Troubleshooting

### Out of Memory Errors

**Reduce batch size** in `tools/finetune_mac.py`:
```python
per_device_train_batch_size = 1  # Already at minimum
gradient_accumulation_steps = 8  # Reduce from 16
```

**Or reduce model context**:
```python
max_seq_length = 1024  # Reduce from 2048
```

### Slow Training

**Normal for M4 Mac** - 4-6 hours is expected. To speed up:
- Reduce epochs: `num_train_epochs = 1`
- Use smaller dataset: `dataset2_split = "train[:5000]"`
- Consider cloud GPU for production

### MPS Errors

If you see MPS-related errors, fall back to CPU:
```python
# In finetune_mac.py, modify get_device():
def get_device():
    return torch.device("cpu")  # Force CPU
```

### Mac Goes to Sleep

**Prevent sleep during training**:
```bash
# Run training with caffeinate (prevents sleep)
caffeinate -i python tools/finetune_mac.py
```

---

## 💡 Pro Tips for M4 Mac Training

1. **Close other apps** - Free up RAM and GPU
2. **Use caffeinate** - Prevent sleep during long training
3. **Monitor temperature** - Use iStat Menus or similar
4. **Test first** - Run 1 epoch with small data before full training
5. **Save checkpoints** - Script saves every 500 steps automatically
6. **Use SSD** - Ensure adequate disk space (model + checkpoints ~20GB)

---

## 📈 What to Expect - Timeline

```
[0:00] Starting training...
[0:05] First batch processed
[0:30] 100 steps completed
[1:30] First epoch 50% complete
[2:00] First epoch complete
[4:00] Second epoch complete
[6:00] Training complete! ✅
```

---

## 🎉 Success Criteria

After training completes, you should see:
- ✅ Model saved to `./output_mac/`
- ✅ Final loss < 1.0 (typically 0.5-0.8)
- ✅ Evaluation loss decreasing
- ✅ Generated text is coherent and on-topic

---

## 📚 Next Steps

1. **Evaluate the model** - Test on your specific use cases
2. **Merge LoRA adapters** - For deployment
3. **Quantize** - For smaller size and faster inference
4. **Deploy** - Use Ollama, LM Studio, or HuggingFace Inference

---

## 🆘 Need Help?

- Check logs in `./output_mac/`
- Review error messages carefully
- Consider cloud GPU for faster iterations
- Join HuggingFace or Reddit communities for support

---

**Ready to start?** Just run:
```bash
export HF_TOKEN="your_token"
python tools/finetune_mac.py
```

🍎 Happy fine-tuning on your M4 Mac!
