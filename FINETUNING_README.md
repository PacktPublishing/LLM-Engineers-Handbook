# README - Fine-Tuning Setup

## ✅ Your M4 Mac is Ready!

All dependencies have been installed and your M4 GPU (Metal Performance Shaders) is working perfectly.

## 🚀 Start Training Now

### Option 1: M4 Mac (Recommended for Development)

```bash
export HF_TOKEN="your_huggingface_token"
python tools/finetune_mac.py
```

- **Time**: 4-6 hours
- **Cost**: FREE
- **Privacy**: Data stays local
- **Guide**: [MAC_TRAINING_GUIDE.md](MAC_TRAINING_GUIDE.md)

### Option 2: Google Colab (Recommended for Speed)

1. Open `notebooks/unsloth_finetuning.ipynb`
2. Upload to https://colab.research.google.com
3. Runtime → GPU
4. Run all cells

- **Time**: 50 min - 2 hours
- **Cost**: FREE (T4) or $10/mo (A100)
- **Speed**: 5-10x faster than M4 Mac

## 📚 Documentation

- **[M4_MAC_SETUP_COMPLETE.md](M4_MAC_SETUP_COMPLETE.md)** - Complete overview
- **[MAC_TRAINING_GUIDE.md](MAC_TRAINING_GUIDE.md)** - Step-by-step guide
- **[UNSLOTH_SETUP_STATUS.md](UNSLOTH_SETUP_STATUS.md)** - Technical details

## 🔧 Verify Anytime

```bash
python verify_setup.py
```

## 🎯 What's Configured

- ✅ PyTorch 2.5.1 with MPS
- ✅ Transformers, Datasets, PEFT, TRL
- ✅ M4 GPU working
- ✅ Ready to fine-tune Llama models

## 💡 Key Differences

**Unsloth** (CUDA GPUs only):
- Uses custom CUDA kernels
- 2x faster, 60% less memory
- Requires NVIDIA GPU
- Not compatible with Mac

**Standard PyTorch** (Your M4 Mac):
- Uses Metal Performance Shaders
- Works on Apple Silicon
- Slightly slower but fully functional
- Everything else identical (LoRA, datasets, etc.)

Both produce the same quality models!

## 🆘 Need Help?

Check [MAC_TRAINING_GUIDE.md](MAC_TRAINING_GUIDE.md) for:
- Detailed instructions
- Troubleshooting
- Performance tips
- Example outputs

---

**Ready?** Start training:
```bash
export HF_TOKEN="your_token"
python tools/finetune_mac.py
```

🍎 Happy fine-tuning on M4 Mac!
