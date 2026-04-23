# Supervised Fine-Tuning with Unsloth: Complete Guide

## Table of Contents
1. [What We're Trying to Achieve](#what-were-trying-to-achieve)
2. [Core Concepts](#core-concepts)
3. [The Fine-Tuning Pipeline](#the-fine-tuning-pipeline)
4. [Technical Details](#technical-details)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## What We're Trying to Achieve

### The Goal: Building an LLM Twin

We're creating a **personalized AI assistant** that mimics a specific person's:
- **Writing style**: How they structure sentences, their tone, vocabulary
- **Domain knowledge**: Expertise in specific areas (e.g., machine learning, engineering)
- **Response patterns**: How they typically answer questions

This is called an **LLM Twin** - a digital twin that can communicate like the person it's modeled after.

### Why This Matters

1. **Personal AI Assistant**: Automate responses while maintaining your unique voice
2. **Knowledge Preservation**: Capture and share expertise at scale
3. **Consistency**: Maintain consistent communication across contexts
4. **Scalability**: Handle multiple conversations simultaneously

### The Approach: Supervised Fine-Tuning (SFT)

We take a pre-trained base model (Meta-Llama-3.1-8B) and teach it to:
1. Follow instructions in a specific format (Alpaca template)
2. Generate responses that match the target writing style
3. Incorporate domain-specific knowledge
4. Stop generating at appropriate points

---

## Core Concepts

### 1. Base Model: Meta-Llama-3.1-8B

**What it is**: A foundation language model with 8 billion parameters, pre-trained on vast amounts of text data.

**Why we use it**:
- Strong baseline capabilities (language understanding, reasoning)
- Open-source and accessible
- Gated model (requires authentication, ensuring quality control)
- Good balance between performance and resource requirements

**What it needs**: Fine-tuning to specialize for our specific task and style.

### 2. LoRA (Low-Rank Adaptation)

**The Problem**: Training all 8 billion parameters is:
- Extremely memory-intensive (requires hundreds of GB of VRAM)
- Very slow
- Expensive

**The Solution**: LoRA adds small, trainable matrices to the model instead of updating all weights.

```
Original weight matrix: W (large, frozen)
LoRA update: W' = W + A × B
Where A and B are small matrices (rank r << model dimension)
```

**Key Parameters**:
- **Rank (r=32)**: Dimension of the LoRA matrices
  - Higher rank = more capacity but more memory
  - 32 is a sweet spot for style transfer
  - Can increase to 64 or 128 for more complex tasks

- **Alpha (α=32)**: Scaling factor for LoRA updates
  - Controls the magnitude of updates
  - Usually set equal to rank
  - α/r determines the effective learning rate for LoRA

- **Target Modules**: Which layers get LoRA
  ```python
  ["q_proj", "k_proj", "v_proj",  # Attention: Query, Key, Value
   "o_proj",                       # Attention: Output
   "gate_proj",                    # FFN: Gating
   "up_proj", "down_proj"]         # FFN: Up/Down projections
  ```
  - We target **all linear layers** for maximum quality
  - Targeting only q_proj/v_proj is faster but lower quality

**Benefits**:
- Only train ~1-2% of parameters
- Drastically reduced memory usage
- Faster training
- Easy to swap between different LoRA adapters

### 3. Unsloth Library

**What it is**: An optimized training framework that makes fine-tuning 2x faster with less memory.

**Key Optimizations**:
- Custom CUDA kernels for faster forward/backward passes
- Optimized gradient checkpointing
- Memory-efficient attention mechanisms
- Native LoRA/QLoRA support

**Why we use it**:
- Faster training = lower costs
- Can train on smaller GPUs (L4 instead of A100)
- Seamless integration with Hugging Face
- Well-maintained and actively developed

### 4. Training Data Strategy

#### Dataset 1: llmtwin (~3,000 samples)
**Purpose**: Domain-specific knowledge and style

**Content**:
- Instruction-response pairs from the target person
- Examples of their writing style
- Domain-specific knowledge

**Challenge**: Too small alone - model might not learn the instruction format properly

#### Dataset 2: FineTome-Alpaca-100k (10,000 samples)
**Purpose**: General instruction-following capability

**Content**:
- High-quality general instruction-response pairs
- Filtered from a larger dataset using quality classifiers
- Covers diverse topics and question types

**Why we combine them**:
```
llmtwin only → Overfitting, poor instruction following
FineTome only → Generic responses, loses personal style
Both combined → Best of both worlds
```

**Ratio**: ~23% domain-specific, ~77% general
- Enough domain knowledge to maintain style
- Enough general data to prevent overfitting

### 5. Alpaca Prompt Template

**Format**:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}<EOS>
```

**Why this format**:
- **Clear structure**: Model learns where instruction ends and response begins
- **Consistency**: Same format for all training samples
- **Simple**: No special tokens beyond EOS
- **Proven**: Used successfully in many fine-tuning projects

**Alternative**: ChatML format (more complex but better for multi-turn conversations)

### 6. Training Hyperparameters

#### Learning Rate: 3e-4
**Why**: 
- Base models need smaller LR than training from scratch
- Too high → unstable training, catastrophic forgetting
- Too low → slow convergence, underfitting
- 3e-4 is empirically proven for LLM fine-tuning

#### Batch Size: Effective 16 (2 × 8)
**Calculation**:
```
Per-device batch size: 2
Gradient accumulation steps: 8
Effective batch size: 2 × 8 = 16
```

**Why split it**:
- Large batch sizes don't fit in memory
- Gradient accumulation simulates larger batches
- Same final result, less memory

#### Epochs: 3
**Why**:
- 1 epoch: Underfitting, hasn't learned enough
- 3 epochs: Sweet spot for most tasks
- 5+ epochs: Risk of overfitting

**How to decide**:
- Monitor validation loss
- Stop when validation loss stops decreasing

#### Optimizer: AdamW 8-bit
**Components**:
- **Adam**: Adaptive learning rates per parameter
- **W** (Weight decay): L2 regularization to prevent overfitting
- **8-bit**: Quantized optimizer states (saves memory)

**Benefits**:
- 50% memory reduction vs 32-bit Adam
- Minimal performance impact
- Allows training with limited VRAM

#### Precision: BF16 or FP16
**BF16 (BFloat16)**: Preferred on newer GPUs
- Better numerical stability
- Same range as FP32
- Native support on A100, H100

**FP16 (Half Precision)**: Fallback for older GPUs
- 2x memory reduction vs FP32
- 2x faster computation
- May have numerical stability issues

---

## The Fine-Tuning Pipeline

### Phase 1: Setup and Authentication

```python
# Environment variables needed
HF_TOKEN=your_huggingface_token      # Access gated models
COMET_API_KEY=your_comet_api_key     # Track experiments
```

**What happens**:
1. Verify credentials for Hugging Face
2. Verify credentials for Comet ML
3. Prepare logging infrastructure

### Phase 2: Model Loading

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    load_in_4bit=False,  # Use LoRA, not QLoRA
)
```

**What happens**:
1. Download model from Hugging Face Hub (~16GB)
2. Load into GPU memory
3. Load tokenizer (converts text ↔ tokens)
4. Set maximum sequence length (2048 tokens ≈ 1500 words)

**Memory usage**: ~16GB GPU VRAM for model weights

### Phase 3: LoRA Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                    # Rank
    lora_alpha=32,           # Alpha
    lora_dropout=0,          # No dropout
    target_modules=[...],    # All linear layers
)
```

**What happens**:
1. Freeze all original model parameters
2. Add LoRA matrices to target modules
3. Only LoRA parameters are trainable (~0.5GB)

**Result**: Reduced from 8B trainable params to ~130M (~1.6%)

### Phase 4: Data Preparation

#### Step 4.1: Load Datasets
```python
dataset1 = load_dataset("mlabonne/llmtwin", split="train")
dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
dataset = concatenate_datasets([dataset1, dataset2])
```

**Result**: ~13,000 total samples

#### Step 4.2: Format with Alpaca Template
```python
def format_samples(examples):
    text = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        message = ALPACA_TEMPLATE.format(instruction, output) + EOS_TOKEN
        text.append(message)
    return {"text": text}
```

**Critical**: Adding `EOS_TOKEN` teaches the model when to stop generating

#### Step 4.3: Train/Test Split
```python
dataset = dataset.train_test_split(test_size=0.05)
```

**Result**:
- Training: ~12,350 samples (95%)
- Testing: ~650 samples (5%)

**Purpose**: Monitor overfitting during training

### Phase 5: Training

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    # ... configuration ...
)
trainer.train()
```

**What happens during training**:

1. **Epoch 1** (Pass 1 through data):
   - Model learns basic instruction-response patterns
   - High training loss initially
   - Validation loss decreases rapidly

2. **Epoch 2** (Pass 2):
   - Model refines understanding
   - Learns writing style nuances
   - Training loss continues decreasing

3. **Epoch 3** (Pass 3):
   - Model solidifies learning
   - Fine-tunes edge cases
   - Validation loss plateaus (good sign)

**Duration**: 
- A100 GPU: ~50 minutes
- A40 GPU: ~80 minutes
- L4 GPU: ~120 minutes

**Monitoring**: Track in Comet ML
- Training loss (should decrease)
- Validation loss (should decrease then plateau)
- Learning rate (decreases linearly)
- GPU utilization (should be high)

### Phase 6: Testing

```python
FastLanguageModel.for_inference(model)
message = ALPACA_TEMPLATE.format("Write a paragraph...", "")
outputs = model.generate(**inputs, max_new_tokens=256)
```

**What we check**:
1. **Format compliance**: Follows Alpaca template
2. **Coherence**: Generates sensible text
3. **Style**: Matches target writing style
4. **Termination**: Properly uses EOS token

**Example output**:
```
Supervised fine-tuning is a method used to enhance a language model
by providing it with a curated dataset of instructions and their
corresponding answers. This process is designed to align the model's
responses with human expectations, thereby improving its accuracy
and relevance...
```

### Phase 7: Saving and Deployment

```python
# Save locally
model.save_pretrained_merged("model_sft", tokenizer, save_method="merged_16bit")

# Upload to Hugging Face Hub
model.push_to_hub_merged("username/TwinLlama-3.1-8B", tokenizer)
```

**What happens**:
1. **Merge**: Combine base weights with LoRA weights
2. **Convert**: Save in 16-bit precision
3. **Package**: Include tokenizer and config files
4. **Upload**: Push to Hugging Face Hub (optional)

**Result**: ~16GB model ready for inference

---

## Technical Details

### Memory Requirements

| Component | Memory Usage |
|-----------|--------------|
| Base model (BF16) | ~16 GB |
| LoRA adapters | ~0.5 GB |
| Optimizer states (8-bit) | ~2 GB |
| Gradients | ~1 GB |
| Activations (per batch) | ~4 GB |
| **Total** | **~24 GB** |

**GPU Requirements**:
- Minimum: L4 (24GB) or RTX 4090 (24GB)
- Recommended: A40 (48GB) or A100 (40GB/80GB)
- Can use QLoRA for smaller GPUs (reduces to ~12GB)

### Packing: Efficient Sequence Handling

**Without packing**:
```
Sample 1: [150 tokens] + [1898 padding] = 2048
Sample 2: [200 tokens] + [1848 padding] = 2048
Sample 3: [100 tokens] + [1948 padding] = 2048
Efficiency: ~12%
```

**With packing**:
```
Batch 1: [Sample 1: 150] + [Sample 2: 200] + [Sample 3: 100] + [more...] = 2048
Efficiency: ~95%
```

**Benefits**:
- Much faster training (less wasted computation)
- Better GPU utilization
- Same final model quality

### Loss Function: Cross-Entropy

**What it measures**: How well predicted tokens match actual tokens

```python
For each position in the sequence:
  predicted_probs = softmax(model_output)
  actual_token = target[position]
  loss += -log(predicted_probs[actual_token])
```

**Training goal**: Minimize this loss
- Lower loss = better predictions
- Track both training and validation loss

### Learning Rate Schedule: Linear Decay

```
Learning Rate
^
|  /\
| /  \___
|/      \___
|           \___
+--------------> Training Steps
  Warmup   Decay
```

**Phases**:
1. **Warmup** (10 steps): 0 → 3e-4
   - Gentle start prevents instability
   
2. **Linear decay**: 3e-4 → 0
   - Gradually reduces learning rate
   - Helps fine-tune at the end

### Gradient Accumulation

**Standard training**:
```python
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Update after each batch
```

**With gradient accumulation**:
```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update after N batches
        optimizer.zero_grad()
```

**Effect**: Simulates larger batch size without extra memory

---

## Best Practices

### 1. Dataset Quality Over Quantity

**Do**:
- ✅ Curate high-quality instruction-response pairs
- ✅ Remove duplicates and low-quality samples
- ✅ Balance domain-specific and general data
- ✅ Use consistent formatting

**Don't**:
- ❌ Include noisy or contradictory examples
- ❌ Use too many samples from a single source
- ❌ Mix different prompt formats
- ❌ Ignore data preprocessing

### 2. Hyperparameter Tuning

**Start with defaults**:
```python
learning_rate = 3e-4
num_epochs = 3
batch_size = 16 (effective)
lora_rank = 32
```

**Tune if needed**:
- **Underfitting** (high train/val loss):
  - Increase LoRA rank (32 → 64)
  - Increase epochs (3 → 5)
  - Increase learning rate (3e-4 → 5e-4)

- **Overfitting** (low train loss, high val loss):
  - Decrease epochs (3 → 2)
  - Add more general data
  - Increase weight decay (0.01 → 0.1)
  - Add LoRA dropout (0 → 0.1)

### 3. Monitoring Training

**Watch these metrics**:

1. **Training Loss**:
   - Should decrease steadily
   - Sudden spikes indicate instability

2. **Validation Loss**:
   - Should decrease initially
   - Plateau is normal
   - Increase indicates overfitting

3. **Learning Rate**:
   - Should decay linearly
   - Verify warmup is working

4. **GPU Utilization**:
   - Should be 90%+
   - Low utilization = bottleneck elsewhere

**Red flags**:
- Loss becomes NaN → Learning rate too high
- Loss oscillates wildly → Reduce batch size or LR
- Validation loss increases → Stop training early

### 4. Evaluation

**Quantitative**:
- Validation loss
- Perplexity
- Accuracy on held-out test set

**Qualitative**:
- Generate responses to diverse prompts
- Check for:
  - Coherence
  - Style consistency
  - Factual accuracy
  - Format compliance
  - Appropriate length

### 5. Deployment Considerations

**Model size**: ~16GB
- Consider quantization (GPTQ, AWQ) for smaller size
- 4-bit quantization → ~4GB

**Inference speed**:
- A100: ~50 tokens/sec
- L4: ~20 tokens/sec
- Consider using vLLM for faster inference

**API integration**:
- Use Hugging Face TGI or vLLM
- Implement proper error handling
- Add rate limiting and caching

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce batch size (2 → 1)
2. Increase gradient accumulation (8 → 16)
3. Reduce max sequence length (2048 → 1024)
4. Use QLoRA instead of LoRA (load_in_4bit=True)
5. Enable CPU offloading

### Issue: Training Too Slow

**Solutions**:
1. Reduce max sequence length
2. Enable packing (if not already)
3. Use fewer LoRA target modules
4. Reduce dataset size for testing
5. Use a GPU with more VRAM (less memory swapping)

### Issue: Model Not Learning

**Check**:
1. Learning rate too low → Increase to 5e-4
2. LoRA rank too small → Increase to 64
3. Not enough epochs → Increase to 5
4. Data formatting issues → Verify template
5. Dataset too small → Add more data

### Issue: Model Overfitting

**Solutions**:
1. Add more general data (FineTome)
2. Reduce epochs (3 → 2)
3. Add LoRA dropout (0 → 0.1)
4. Increase weight decay (0.01 → 0.05)
5. Use early stopping

### Issue: Model Generates Gibberish

**Check**:
1. EOS token added to training data?
2. Tokenizer configured correctly?
3. Chat template applied properly?
4. Learning rate too high? (reduce to 1e-4)
5. Training data corrupted?

### Issue: Can't Access Llama Model

**Solutions**:
1. Verify HF_TOKEN is set correctly
2. Accept license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
3. Check token has read permissions
4. Log in: `huggingface-cli login`

---

## What Success Looks Like

### Training Metrics

**Good training run**:
```
Epoch 1:
  Train Loss: 2.5 → 1.2
  Val Loss: 2.4 → 1.3

Epoch 2:
  Train Loss: 1.2 → 0.8
  Val Loss: 1.3 → 1.1

Epoch 3:
  Train Loss: 0.8 → 0.6
  Val Loss: 1.1 → 1.0
```

**Key indicators**:
- Steady decrease in both losses
- Val loss not diverging from train loss
- Final validation loss < 1.5

### Model Behavior

**Good responses**:
- Follows instruction format
- Matches target writing style
- Factually accurate (where applicable)
- Appropriate length
- Stops at EOS token
- Coherent and fluent

**Example**:
```
Instruction: Explain gradient descent in simple terms.

Good response:
Gradient descent is an optimization algorithm that helps machine learning 
models learn by iteratively adjusting parameters to minimize error. Think 
of it like walking down a hill – you take steps in the direction that goes 
downward (reduces error) until you reach the bottom (optimal solution).

Bad response:
Gradient descent Gradient descent is when you gradient descent the model
and then descent gradient until [continues forever or generates nonsense]
```

---

## Next Steps After Fine-Tuning

### 1. Comprehensive Evaluation
- Test on diverse prompts
- Compare with base model
- Get human feedback
- Calculate metrics (BLEU, ROUGE, etc.)

### 2. Iterative Improvement
- Collect failure cases
- Add them to training data
- Fine-tune again with adjusted hyperparameters
- Repeat until satisfied

### 3. DPO (Optional)
- Collect preference data (good vs bad responses)
- Run DPO fine-tuning for alignment
- Further refine model behavior

### 4. Deployment
- Choose inference framework (vLLM, TGI, etc.)
- Set up API endpoint
- Implement monitoring and logging
- Deploy to production

### 5. Monitoring
- Track latency and throughput
- Collect user feedback
- Monitor for drift
- Plan for continuous improvement

---

## Additional Resources

### Documentation
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)

### Papers
- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Llama 3: [Llama 3 Model Card](https://github.com/meta-llama/llama3)
- Alpaca: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

### Community
- Hugging Face Forums
- Unsloth Discord
- r/LocalLLaMA subreddit

---

## Summary

**What we built**: A personalized AI assistant (LLM Twin) that mimics specific writing style and knowledge

**How we did it**: 
1. Took a base model (Llama 3.1 8B)
2. Applied LoRA for efficient training
3. Fine-tuned on domain-specific + general data
4. Used Alpaca template for consistency
5. Trained with optimized hyperparameters

**Key takeaways**:
- LoRA makes large model fine-tuning feasible
- Data quality matters more than quantity
- Combining domain-specific and general data prevents overfitting
- Unsloth accelerates training significantly
- Proper monitoring is essential for success

**Result**: A production-ready model that can generate personalized responses while maintaining quality and coherence.
