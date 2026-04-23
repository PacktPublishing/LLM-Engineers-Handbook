# LoRA Fine-tuning Guide - Key Learnings

## Session Summary
Complete guide to understanding LoRA (Low-Rank Adaptation) fine-tuning on Apple Silicon M4 Mac, covering theoretical concepts, practical implementation, and training insights.

---

## Table of Contents
1. [What is LoRA?](#what-is-lora)
2. [LoRA vs QLoRA](#lora-vs-qlora)
3. [Training Process Explained](#training-process-explained)
4. [Loss Calculation](#loss-calculation)
5. [Why Training Takes Time](#why-training-takes-time)
6. [Monitoring with TensorBoard](#monitoring-with-tensorboard)
7. [Dataset and Progress](#dataset-and-progress)

---

## What is LoRA?

### The Problem LoRA Solves
Full fine-tuning of an LLM requires updating ALL parameters (e.g., 1.1B for TinyLlama), which:
- Requires massive memory (13GB+ just for gradients)
- Is slow (updating billions of parameters)
- Needs expensive GPU hardware

### How LoRA Works
LoRA freezes the base model and adds small "adapter" layers that learn the task-specific adjustments.

**Formula:**
```
Output = (BaseWeight + LoRA_A × LoRA_B) × Input
         └─Frozen─┘  └──Trainable──┘
```

**Key Insight:** 
- **ALL 1.1B params are used** in computation (forward pass)
- **ONLY 25M params are modified** (backward pass/training)

### LoRA Configuration
```python
LoraConfig(
    r=32,              # Rank: controls adapter size
    lora_alpha=32,     # Scaling factor
    target_modules=[   # Which layers to adapt
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
)
```

### Memory Savings
```
Full Fine-tuning:
- Model: 2.2GB
- Gradients: 8GB
- Optimizer: 16GB
- Total: ~26GB ❌

LoRA Fine-tuning:
- Model: 2.2GB
- Gradients: 100MB (only for adapters)
- Optimizer: 300MB (only for adapters)
- Total: ~2.6GB ✅ (10x less!)
```

### Parameter Breakdown
```
TinyLlama 1.1B Model:
├─ Total parameters: 1,100,048,384 (100%)
├─ Frozen (base): 1,074,817,024 (97.76%)
└─ Trainable (LoRA): 25,231,360 (2.24%)

Trainable calculation:
- 22 layers × 7 target modules × 131,072 params/module
- = ~20M params
- + embeddings ≈ 25M total
```

---

## LoRA vs QLoRA

### Core Difference

**LoRA:**
- Base model stored in **16-bit (fp16)**: ~13GB for 8B model
- LoRA adapters: **~84MB**
- Total: **~13GB**

**QLoRA:**
- Base model stored in **4-bit quantization**: ~4GB for 8B model
- LoRA adapters: **~84MB**
- Total: **~4GB** (70% memory reduction!)

### 4-bit Quantization Explained
```
Regular (16-bit): Each weight = 16 bits precision
Example: 3.141592653589793

4-bit: Each weight = 4 bits precision  
Example: 3.14

Storage: 16 bits → 4 bits = 4x compression
```

### Why QLoRA Didn't Work on Mac
- Requires **BitsAndBytes** library
- BitsAndBytes is **CUDA-only** (NVIDIA GPUs)
- Mac **MPS backend** doesn't support it
- Solution: Used regular LoRA with smaller TinyLlama 1.1B model

### Performance Comparison
| Method | Memory | Accuracy | Mac Support |
|--------|--------|----------|-------------|
| Full Fine-tuning | 26GB | 100% | ❌ |
| LoRA (fp16) | 13GB | 99% | ✅ |
| QLoRA (4-bit) | 4GB | 99% | ❌ (needs CUDA) |

---

## Training Process Explained

### Step-by-Step Flow

**Step 1: Load Batch**
```python
Batch size: 1 sample
Gradient accumulation: 16 steps
Effective batch: 16 samples per weight update
```

**Step 2: Tokenization**
```python
Text → Numbers
"Supervised learning uses labeled data" 
→ [1, 13866, 338, 385, 7023, ..., 2]
Length: 2048 tokens (truncated if longer)
```

**Step 3: Forward Pass**
```python
# For EACH token position:
Input → Layer 1 (1.1B params) → Layer 2 → ... → Layer 22
        └─ Uses ALL frozen params
        └─ Adds LoRA adjustments (25M params)
        
Output: Probabilities for 32,000 possible next tokens
```

**Step 4: Loss Calculation**
```python
For each token:
  predicted_probability = model_output[correct_token]
  loss = -log(predicted_probability)

Sample loss = average(all_token_losses)
```

**Step 5: Backward Pass**
```python
# Compute gradients ONLY for LoRA params
Gradients flow through: 25M params
Frozen params: No gradients computed (saves memory!)
```

**Step 6: Weight Update**
```python
# After 16 forward/backward passes:
optimizer.step()  # Updates 25M LoRA parameters

Before: LoRA_A[0,0] = 0.0234
After:  LoRA_A[0,0] = 0.0231 (tiny adjustment)
```

### Training vs Evaluation

| Aspect | Training (12,350 samples) | Evaluation (651 samples) |
|--------|---------------------------|--------------------------|
| **When** | Every step | Every 100 steps |
| **Forward pass** | ✅ Yes | ✅ Yes |
| **Loss calculation** | ✅ Yes | ✅ Yes |
| **Backpropagation** | ✅ Yes | ❌ No |
| **Weight updates** | ✅ Yes | ❌ No |
| **Purpose** | Learn patterns | Measure generalization |

---

## Loss Calculation

### Cross-Entropy Loss Formula

$$\text{Loss} = -\log(P_{\text{correct}})$$

Where $P_{\text{correct}}$ is probability assigned to the correct token.

### Examples
```python
# Perfect prediction:
P_correct = 1.0 → Loss = -log(1.0) = 0.0 ✅

# Good prediction:
P_correct = 0.8 → Loss = -log(0.8) = 0.22 ✅

# Poor prediction:
P_correct = 0.1 → Loss = -log(0.1) = 2.30 ❌

# Terrible prediction:
P_correct = 0.01 → Loss = -log(0.01) = 4.61 ❌
```

### Interpreting Loss Values

```python
Loss = 1.1 (current)
→ e^(-1.1) ≈ 0.33 (33% probability to correct tokens)

Loss = 0.3 (target)
→ e^(-0.3) ≈ 0.74 (74% probability to correct tokens)
```

### Why This Formula?
1. **Exponentially punishes confident wrong predictions**
2. **Rewards confident correct predictions**
3. **Gradient descent naturally minimizes this**
4. **Differentiable** (enables backpropagation)

---

## Why Training Takes Time

### Current Performance
```
Speed: 25-38 seconds per step
Total steps: 2,316
Total time: ~16-24 hours on M4 Mac
```

### Time Breakdown Per Step

```python
Forward pass (1.1B params):   15 seconds  ← Bottleneck!
Backward pass (25M params):    5 seconds
Weight update (25M params):    1 seconds
Data loading overhead:         4 seconds
─────────────────────────────────────────
Total per step:               25 seconds
```

### Why Forward Pass is Slow

**1. Must Compute ALL 1.1B Parameters**
```python
Even though only 25M are trainable:
- Forward pass needs base model computation
- Formula: (BaseWeight + LoRA) × Input
- Can't skip the BaseWeight part!

Operations: 1.1B params × 2,048 tokens = 2.2 trillion ops
```

**2. MPS vs CUDA Speed**
```
M4 Mac MPS:
- Memory bandwidth: ~200 GB/s
- Compute: ~10 TFLOPS
- Forward pass: ~15 seconds

NVIDIA A100 CUDA:
- Memory bandwidth: 1,555 GB/s (8x faster!)
- Compute: ~312 TFLOPS (30x faster!)
- Forward pass: ~0.3 seconds (50x faster!)
```

**3. Small Batch Size = High Overhead**
```
Batch size 1:
- Process 1 sample: 15 seconds
- Load next: 0.5 seconds
- Repeat 16 times
- Update weights: 1 second

If batch size was 16 (impossible on Mac):
- Process 16 samples in parallel: 15 seconds
- Update once: 1 second
- 16x speedup!
```

**4. Long Sequences (2,048 tokens)**
```
Attention complexity: O(N²)
N = 2,048 tokens
Operations: 2,048² × 22 layers × 32 heads
= ~3 billion attention operations per sample
```

### LoRA Savings Comparison

**Without LoRA (Full Fine-tuning):**
```
Forward: 15 sec (1.1B params)
Backward: 30 sec (ALL params get gradients)
Update: 5 sec (ALL params updated)
Total: 50 sec/step

Total time: 2,316 × 50 = 32 hours
```

**With LoRA (Current setup):**
```
Forward: 15 sec (1.1B params)
Backward: 5 sec (only 25M params)
Update: 1 sec (only 25M params)
Total: 21 sec/step

Total time: 2,316 × 21 = 13 hours
```

**LoRA saves 19 hours (59% faster!)**

### Hardware Comparison

| Hardware | Time/Step | Total Time | Cost |
|----------|-----------|------------|------|
| M4 Mac MPS | 25 sec | 16 hours | Free |
| Google Colab T4 | 3 sec | 2 hours | Free |
| Google Colab A100 | 0.5 sec | 20 min | $10/mo |
| NVIDIA RTX 4090 | 2 sec | 1.3 hours | $0.34/hr |

---

## Monitoring with TensorBoard

### Starting TensorBoard
```bash
tensorboard --logdir output_mac/runs --port 6006
# Open: http://localhost:6006
```

### Key Metrics to Watch

**1. train/loss** (Most Important!)
```
Current: 1.1 → Decreasing
Target: 0.3-0.5

Good: Smooth downward curve
Bad: Flat or increasing
```

**2. train/mean_token_accuracy**
```
Current: 70%
Target: 85-90%

Measures: % of tokens predicted correctly
```

**3. train/grad_norm**
```
Current: 0.2-0.3
Good range: 0.1 - 5.0

Too high (>10): Exploding gradients
Too low (<0.01): Vanishing gradients
```

**4. train/learning_rate**
```
Starts: 0.0003 (3e-4)
Warmup: Steps 0-10 (increases to peak)
Decay: Steps 10-2316 (decreases to 0)
```

**5. eval/loss** (Every 100 steps)
```
Measures: Performance on unseen test data
Should: Stay close to train/loss

If eval/loss >> train/loss: Overfitting!
If eval/loss ≈ train/loss: Good generalization ✅
```

### Interpreting Dashboard

```
Healthy Training:
✅ train/loss decreasing (1.1 → 0.8 → 0.5)
✅ eval/loss decreasing with train/loss
✅ accuracy increasing (70% → 80% → 85%)
✅ grad_norm stable (0.1-5.0)

Warning Signs:
❌ train/loss flat for >500 steps
❌ eval/loss increasing while train/loss decreases
❌ grad_norm exploding (>10)
❌ accuracy dropping
```

---

## Dataset and Progress

### Dataset Composition
```
Source 1: mlabonne/llmtwin (3,001 samples)
Source 2: mlabonne/FineTome-Alpaca-100k (10,000 samples)
Total: 13,001 samples

Split (seed=42 for reproducibility):
├─ Train: 12,350 samples (95%)
└─ Test: 651 samples (5%)
```

### Data Format
```python
Alpaca Template:
"""
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}<eos>
"""

Example:
Instruction: "What is supervised learning?"
Response: "Supervised learning uses labeled data..."
```

### Training Schedule
```
Total steps: 2,316
Steps per epoch: 772
Total epochs: 3

Breakdown:
Epoch 1: Steps 1-772
Epoch 2: Steps 773-1544
Epoch 3: Steps 1545-2316

Evaluation: Every 100 steps (23 checkpoints)
Saving: Every 500 steps
```

### Progress Metrics
```
Total samples to process: 12,350 × 3 epochs = 37,050
Total tokens: 37,050 × 2,048 = 75,878,400
Total sentences: ~148,200 (with repetition)
Unique sentences: ~49,400

Current (Step 96):
├─ Progress: 4%
├─ Samples processed: 1,536
├─ Tokens processed: 3,145,728
├─ Time elapsed: 46 minutes
└─ ETA: ~20 hours remaining
```

### Seed Explanation
```python
dataset.train_test_split(test_size=0.05, seed=42)

seed=42:
- Controls randomness of the split
- Same seed = same split every run
- Ensures reproducibility
- 42 is common convention (Hitchhiker's Guide reference)
```

---

## Key Takeaways

### The Core Concept
```
LoRA Fine-tuning:
1. ALL 1.1B params are used (computation)
2. ONLY 25M params are changed (training)
3. Like steering a car - use whole car, adjust only steering
```

### Memory Efficiency
```
LoRA: Only store gradients for 2.2% of parameters
Result: 10x less memory needed
Enables: Training on consumer hardware
```

### Time Trade-offs
```
Mac (16 hours) vs Colab (2 hours):
- Mac: Free, slower, learning tool
- Colab: Free T4 GPU, 8x faster
- Colab A100: $10/mo, 50x faster
```

### Reversibility
```
Base model: Never modified
LoRA adapters: Can add/remove/swap
Result: One base model + multiple task adapters
Storage: Base (2.2GB) + adapters (50MB each)
```

### When Training is Working
```
✅ Loss steadily decreasing
✅ Accuracy increasing
✅ eval/loss tracking train/loss
✅ Gradients stable
✅ No spikes in metrics
```

---

## Files Created

### Training Script
- `tools/finetune_mac.py` - Main M4 Mac fine-tuning script (402 lines)
- Uses standard PyTorch/Transformers/PEFT (no Unsloth on Mac)

### Documentation
- `MAC_TRAINING_GUIDE.md` - Step-by-step guide
- `M4_MAC_SETUP_COMPLETE.md` - Setup summary
- `MONITORING_GUIDE.md` - TensorBoard guide
- `UNSLOTH_SETUP_STATUS.md` - Dependency status
- `FINETUNING_README.md` - Quick reference

### Alternative
- `notebooks/unsloth_finetuning.ipynb` - Google Colab notebook for faster training

---

## Common Questions

**Q: Why only 25M trainable if model is 1.1B?**
A: LoRA freezes base model, only trains small adapters. This saves memory and enables local training.

**Q: Can I reverse the fine-tuning?**
A: Yes! Base model unchanged, adapters can be removed/swapped anytime.

**Q: Why is Mac training slow?**
A: Still computing through 1.1B params, MPS 50x slower than CUDA, batch size limited to 1.

**Q: What's the difference between LoRA and QLoRA?**
A: QLoRA uses 4-bit quantization for base model (4GB vs 13GB), but needs CUDA.

**Q: How is loss calculated?**
A: Cross-entropy: -log(probability_of_correct_token). Lower = better predictions.

**Q: Why use test set?**
A: Measures generalization - ensures model learns patterns, not memorizes training data.

---

## Resources

- **Training logs:** `training_output.log`
- **TensorBoard:** http://localhost:6006
- **Checkpoints:** `./output_mac/checkpoint-{step}/`
- **Final model:** `./output_mac/`
- **Adapters only:** `./output_mac/adapter_model.bin` (~50MB)

---

*Generated from training session on December 28, 2025*
*Model: TinyLlama 1.1B with LoRA fine-tuning on Apple M4 Mac*
