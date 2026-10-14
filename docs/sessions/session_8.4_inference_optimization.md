# Session 8.4: Inference Optimization (Book Chapter 8)

## 🎯 Learning Objectives

By the end of this session, you will:
- Explain why decoder-only LLM inference is memory- and latency-bound
- Apply the core optimizations: KV cache, continuous batching, speculative decoding, optimized attention
- Compute a KV-cache size from model dimensions and place it on a 16 GB budget
- Distinguish data, pipeline, and tensor parallelism and when to use each
- Compare quantization methods (PTQ/QAT, INT8, NF4, GGUF/llama.cpp, GPTQ/EXL2, AWQ, QuIP#/HQQ)
- Choose between the TGI, vLLM, and TensorRT-LLM inference engines using the book's feature table
- Place every technique on the RTX 5000 16 GB constraints (Turing sm_75: no bfloat16, no FlashAttention-2)

> This session covers **Book Chapter 8: Inference Optimization** (pp. 318-343). Theory follows the book; runnable snippets are from the book; the engine comparison uses the book's Table 8.1. Hardware notes are specific to this workstation.

---

## 🏗️ Architecture Overview

### The Inference Bottleneck

A decoder-only model generates one token at a time. Prefill (processing the prompt) is highly parallel, but decode (generating the next token) is inherently sequential.

```
Prompt: "I have a dream"
   │
   ├─ 1. Tokenize + embedding + positional encoding      ── parallel, cheap
   ├─ 2. Compute K, V pairs for all input tokens          ── parallel, expensive
   └─ 3. Generate output tokens one at a time             ── SEQUENTIAL, the bottleneck
           "of" → "a" → "..." (each needs all prior tokens)
```

Every optimization in this chapter either (a) removes redundant work in step 3, (b) fills the accelerator during step 3, or (c) shrinks the memory footprint so more work fits.

### The Optimization Map

```
┌─────────────────────────────────────────────────────────────────────┐
│  Speed up decoding          │  Fill the GPU        │  Shrink memory  │
│  ────────────────           │  ────────────        │  ─────────────  │
│  KV cache (static)          │  Continuous batching │  Quantization   │
│  Speculative decoding       │  (in-flight)         │  (INT8, NF4,    │
│  torch.compile + static KV  │                      │   GGUF, GPTQ,   │
│  FlashAttention-2           │                      │   EXL2, AWQ)    │
│  PagedAttention             │                      │                 │
├─────────────────────────────────────────────────────────────────────┤
│  Scale across GPUs: data parallelism, pipeline parallelism,         │
│  tensor parallelism (combinable)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Which bottleneck do you have?

| Symptom | Likely bottleneck | First technique to reach for |
|---------|-------------------|------------------------------|
| Long time to first token | Prefill compute | Faster hardware, FlashAttention-2, bigger batch of prefills |
| Low tokens/sec after first token | Decode under-utilization | Continuous batching, speculative decoding, CUDA graphs |
| OOM at moderate batch/context | KV cache | Quantization, PagedAttention, GQA, shorter context |
| Model will not fit at all | Weights | Quantization (4-bit/2-bit), sharding |
| High latency per request, spare GPU | Sequential decode | Speculative decoding |

---

## 📁 Key Files Explained: The Optimization Techniques

> There is no single repo file for this chapter; the "files" here are the techniques and the code snippets from the book. Each subsection names the method, its mechanism, and the code that exercises it.

### 1. KV Cache

**The problem**: to predict token 101, the model recomputes attention over tokens 1-100, even though it already computed them for token 100.

**The fix**: store the key/value pairs from self-attention and reuse them. When a new token is generated, only its own K/V is computed and appended. The book notes the KV cache is "an immediate optimization that is implemented in every popular tool and library" (page 321).

```
Token 1:  compute K1,V1  → cache [K1,V1]
Token 2:  compute K2,V2  → cache [K1,V1,K2,V2]   (K1,V1 read, not recomputed)
Token 3:  compute K3,V3  → cache [...K3,V3]
                                    ▲
                    only ONE new K/V per step instead of all
```

**Cache size formula** (book, page 321):

```
Cache size = 2 × n_tokens × n_layers × n_heads × dim_head × n_bytes
             ▲
             2 accounts for both K and V
```

**Worked numbers.** Bytes per token scales linearly. Using 2 bytes (FP16):

| Model dims | K/V heads | Per-token cache | @2,048 | @4,096 | @8,192 |
|------------|-----------|-----------------|--------|--------|--------|
| 32 layers × 32 heads × 128 | 32 (MHA) | 512 KiB | 1.0 GiB | 2.1 GiB | 4.3 GiB |
| 32 layers × 32 q / 8 kv × 128 | 8 (GQA) | 128 KiB | 256 MiB | 512 MiB | 1.0 GiB |
| 32 layers, 1 kv head | 1 (MQA) | 16 KiB | 32 MiB | 64 MiB | 128 MiB |

Derivation for the first row: `2 × 2048 × 32 × 32 × 128 × 2 = 1,073,741,824 bytes ≈ 1.0 GiB`. The book's statement that a 7B model "exceeds 2 GB for high sequence lengths (higher than 2,048 tokens)" matches: at 4,096 tokens it is 2.1 GiB.

> **Accuracy note.** The vanished predecessor version of this section printed "~2.1 GB at 2k" and "~8.6 GB at 8k" for a 32-layer/32-head model. Those values are 2x too large (the factor of 2 for K and V appears to have been applied twice). The correct values are in the table above. Always recompute; do not trust a remembered figure.

> **GQA matters.** Llama-3.1-8B uses **grouped-query attention** with 8 K/V heads, not 32. Its KV cache is 4x smaller than the MHA formula suggests. When you budget memory, check the model's `num_key_value_heads`, not just `num_attention_heads`.

**Python helper (usable on this workstation):**

```python
def kv_cache_gb(n_tokens, n_layers, n_kv_heads, dim_head, bytes_per=2):
    return (2 * n_tokens * n_layers * n_kv_heads * dim_head * bytes_per) / 1e9

# 7B MHA (32 kv heads), FP16
print(kv_cache_gb(2048, 32, 32, 128))    # ~1.07 GB
print(kv_cache_gb(4096, 32, 32, 128))    # ~2.15 GB
# Llama-3.1-8B GQA (8 kv heads), FP16
print(kv_cache_gb(8192, 32, 8, 128))     # ~1.07 GB
```

**Static KV cache + `torch.compile`**: the cache grows each step, which blocks `torch.compile` because it needs static shapes. Pre-allocating the cache to a maximum fixes the shape and fuses the forward pass, for up to **4x** speedup (book, page 322).

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

model.generation_config.cache_implementation = "static"
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer("What is 2+2?", return_tensors="pt").to(device)
outputs = compiled_model.generate(**inputs, do_sample=True, temperature=0.7, max_new_tokens=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

The book's expected output is `['What is 2+2?\n\nThe answer is 4. 2+2 = 4.']`. Note the book's step 5 passes `**inputs` to `generate`; whether you call the compiled or the original model, `generate` is the entry point.

**RTX 5000 note**: `torch.compile` works on Turing but is less mature than on Ampere+. The static-cache trick is still worth measuring. The static cache "doesn't work with all architectures" (book, page 323) - check the Transformers documentation for the model you target.

---

### 2. Continuous Batching (In-Flight Batching)

**The problem**: requests have wildly different prompt/output lengths. Traditional batching waits for the longest request to finish, leaving the GPU partly idle ("straggler" waste).

**The fix**: as soon as one request finishes, evict it and feed a new request into the same batch. The batch stays full, maximizing utilization.

```
Static batching (batch of 4):
 t0 ──[req A ██████████████][req B ██████][req C ████][req D ██████████]
                                          ▲ B,C done but GPU held until A,D finish
 t1 ──[new batch only after ALL finish]

Continuous batching:
 t0 ──[A ██████████████][B ██████][C ████][D ██████████]
 t1 ──[A ██████████████][D ██████████][E ███████]   ← E admitted the moment C leaves
```

- Prefill (encoding waiting requests) is periodically interleaved with decode; the balance is tuned with a **waiting-served ratio** hyperparameter (book, page 323).
- Implemented natively in **TGI**, **vLLM**, and **TensorRT-LLM**.
- The payoff grows with request-length variance. If all requests are identical, continuous batching gains little.

---

### 3. Speculative Decoding (Assisted Generation)

**The idea**: use spare compute to predict several tokens with a small draft model, then validate them in one pass of the large model.

```
1. Draft model predicts k tokens (5-10) in parallel:
      draft → [t1 t2 t3 t4 t5]
2. Large model validates which match what it would have generated:
      large([prompt ... t1 t2 t3 t4 t5]) → [t1 ✓ t2 ✓ t3 ✓ t4 ✗ t5 ✗]
3. Keep the longest matching prefix (t1..t3), discard the rest.
   Net: 3 tokens for the cost of one large-model pass.
```

If the draft matches ~90%, you get a **3-4x speedup** (book, page 324). Crucially, **both models must share a tokenizer**: "If this is not the case, the tokens generated by the draft model will not align with those produced by the large model."

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", device_map="auto")

inputs = tokenizer("What is 2+2?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, do_sample=True, assistant_model=draft_model, temperature=0.7, max_new_tokens=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

The book's output is `['What is 2+2? 2 + 2 equals 4!']`. The book also notes "the speedup in this small example is not significant, but it is clearly noticeable with bigger models" - the draft must be much smaller than the main model to win.

**Prompt lookup decoding**: a variant for input-grounded tasks (summarization) where output overlaps the prompt. Uses shared n-grams as candidates:

```python
outputs = model.generate(**inputs, prompt_lookup_num_tokens=4)
```

**Medusa**: jointly trains speculation heads on the main model; Medusa-1 freezes the large model, Medusa-2 fine-tunes both. A 70M head can approximate a 7B model's next tokens. Speculative decoding is natively supported by TGI (book, page 326).

---

### 4. Optimized Attention

**PagedAttention** (Kwon, Li, et al., 2023): partitions the KV cache into fixed-size blocks, like OS virtual memory. No contiguous allocation needed; blocks are fetched on demand.

- Cuts memory overhead up to **55%**, improves throughput up to **2.2x** (book, page 326).
- Enables memory sharing across sequences from the same prompt (parallel sampling, beam search).
- First implemented in **vLLM**; now in TGI and TensorRT-LLM.

```
Before: each sequence needs one contiguous KV buffer
 [seq A ░░░░░░░░░░░░ reserved but unused]
 [seq B ░░░░░░                      ]
After: KV cache in shared fixed-size blocks
 [blk0][blk1][blk2][blk3]...[blkN]  ← sequences point at blocks; no reservation waste
```

**FlashAttention-2** (Tri Dao, 2023): tiles the attention matrices so blocks fit in on-chip SRAM, and uses **online softmax** to avoid materializing the full score matrix. It computes the softmax block-by-block, keeping a running max and running sum, so large intermediate matrices are never stored.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    attn_implementation="flash_attention_2",
)
```

- Reduces memory from quadratic to linear in sequence length (especially impactful for training, where recomputation replaces storing intermediates).
- Install with `pip install flash-attn --no-build-isolation` (book, page 327).
- **RTX 5000 note**: FlashAttention-2 requires **Ampere (sm_80)+**. The RTX 5000 is **Turing (sm_75)**, so FlashAttention-2 is **not available** on this workstation. Use PyTorch's `scaled_dot_product_attention` (SDPA), which has a memory-efficient backend that works on Turing, or the standard attention path.

---

### 5. Model Parallelism

Three orthogonal ways to spread a model across GPUs:

| Technique | What is split | Best for | Downside |
|-----------|---------------|----------|----------|
| **Data parallelism (DP)** | replicas, split the batch | throughput, small models that fit on one GPU | replicates weights; needs a copy per GPU |
| **Pipeline parallelism (PP)** | layers across GPUs | fitting huge models in memory | "pipeline bubbles" (idle GPUs) |
| **Tensor parallelism (TP)** | weight matrices within a layer | low latency, huge models | needs high-speed interconnect |

**Data parallelism** (book, page 328): copy the model to each GPU, split the batch. Great for training; during inference it handles concurrent requests. Its limit is that each replica holds the full weights, so it only works when the model fits on one GPU.

**Pipeline parallelism** (GPipe, Huang et al., 2019): partition layers across GPUs (first 25% on GPU 1, next 25% on GPU 2, ...). Reduces per-GPU memory, but creates **pipeline bubbles** where GPUs idle waiting for activations. **Micro-batching** mitigates bubbles by splitting the batch into sub-batches that flow through stages, so a stage can start the next micro-batch before the previous finishes. The number of GPUs is the **degree of parallelism**. At the time of writing, "only certain inference frameworks like TensorRT-LLM support pipeline parallelism" (book, page 330).

**Tensor parallelism** (Megatron-LM, Shoeybi, Patwary, Puri et al., 2019): split the weight matrices within a layer across GPUs. Inputs are broadcast, each GPU computes its slice, and partial results are combined with an **all-reduce**. Attention heads parallelize naturally. Layers with whole-input dependencies (`LayerNorm`, `Dropout`) cannot be partitioned and are replicated, or split along the sequence dimension (**sequence parallelism**). TP needs high-bandwidth interconnect, so it is impractical across nodes with weak links.

- **They combine**: a common layout is a few pipeline stages, each using tensor parallelism within. PP gives the greatest memory reduction but sacrifices efficiency to bubbles; TP gives low latency at a larger memory footprint.

**Practical note**: the LLM Twin uses a single GPU for inference (`SM_NUM_GPUS=1`), so parallelism is out of scope locally. It matters when serving 30B+ models.

```
Data parallel:      [GPU0: full model | batch A]
                    [GPU1: full model | batch B]
Pipeline parallel:  [GPU0: layers 0-7][GPU1: 8-15][GPU2: 16-23][GPU3: 24-31]
Tensor parallel:    [GPU0: cols 0:k][GPU1: cols k:2k]  (each layer, all-reduce after)
Combined:           stage0 = {TP over 2 GPUs} → stage1 = {TP over 2 GPUs}
```

---

### 6. Quantization

Represent weights (and sometimes activations) in lower precision.

**Two approaches** (book, page 333):
- **PTQ (Post-Training Quantization)**: convert a trained model directly; simple, slight quality loss.
- **QAT (Quantization-Aware Training)**: quantize during training; better quality, more expensive and needs representative data.

**Precision formats**: FP32 (32-bit), FP16/BF16 (16-bit). The format is `(-1)^sign × base^exponent × significand`. Neural nets prefer **range over precision**, so **BF16** is preferred when supported. NVIDIA Ampere (A100, A30) supports BF16; **Turing (T4, and the RTX 5000 here) does not**, so use FP16 locally.

**Naïve quantization** (book pages 335-336):
- **Absmax**: `X_quant = round(127 · X / max|X|)`.
- **Zero-point**: asymmetric, maps to `[-128, 127]` with an offset.

```python
import torch

def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))
    # Quantize
    X_quant = (scale * X).round()
    return X_quant.to(torch.int8)


def zeropoint_quantize(X):
    # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range
    # Calculate scale
    scale = 255 / x_range
    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()
    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)
    return X_quant.to(torch.int8)
```

**Worked example from the book.** Range `[-3.0, 3.2]`, weight `0.1`:
- Absmax: `round(127 · 0.1 / 3.2) = round(3.97) = 4`. Dequantize: `3.2 · 4 / 127 ≈ 0.1008`, a rounding error of `0.0008`.
- Zero-point: `scale = 255 / 6.2 ≈ 41.13`, `zeropoint = -round(41.13 · -3.0) - 128 = -5`, so `round(41.13 · 0.1 - 5) = -1`. Different integer, same approximate value.

**The outlier problem**: ~0.1% of weight values are extreme and wreck naïve quantization. **LLM.int8()** (Dettmers et al., 2022) processes outliers in FP16 and the rest in INT8, with <1% degradation but ~20% slower inference. **NF4** (Dettmers et al., 2023) is the 4-bit format behind QLoRA (Chapter 5).

```python
from transformers import AutoModelForCausalLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# 8-bit (LLM.int8())
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
# 4-bit NF4 (needs bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```

**Approximate weight memory** (weights only; KV/activations on top):

| Precision | 7B | 8B | 13B | 70B |
|-----------|----|----|-----|-----|
| FP16/BF16 (2 bytes) | ~14 GB | ~16 GB | ~26 GB | ~140 GB |
| INT8 (1 byte) | ~7 GB | ~8 GB | ~13 GB | ~70 GB |
| 4-bit (~0.5 byte) | ~3.5 GB | ~4 GB | ~6.5 GB | ~35 GB |
| 2-bit (~0.25 byte) | ~1.75 GB | ~2 GB | ~3.3 GB | ~17.5 GB |

On a 16 GB card: a 7-8B model at 4-bit (~4 GB) fits with room for a modest KV cache and activations; at FP16 it does not fit alongside runtime overhead.

---

### 7. GGUF and llama.cpp

**llama.cpp** is a C++ inference library that runs on CPU and offloads layers to GPU. It is the most portable option (CPUs, Android) and its **GGUF** format is ubiquitous on the Hub. "It is the most popular quantization technique, with many quantized models available on the Hugging Face Hub" (book, page 338).

**GGUF naming** (bits → quality; book pages 338-339):

| Format | Bits | Quality |
|--------|------|---------|
| `IQ1_S`/`IQ1_M` | 1 | very low |
| `IQ2_XXS/XS/S/M`/`Q2_K` | 2 | low (IQ2 usable for large models) |
| `IQ3_XXS/XS/S/M`/`Q3_K_S/M/L` | 3 | low but usable for large models |
| `IQ4_XS/NL`/`Q4_K_S/M`/`Q4_0/1` | 4 | good, most models |
| `Q5_K_S/M`/`Q5_0/1` | 5 | high |
| `Q6_K` | 6 | very high |
| `Q8_0` | 8 | highest |

Quantize with llama.cpp (book, page 339):

```bash
# 1. Build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && git pull && make clean && LLAMA_CUBLAS=1 make
pip install -r llama.cpp/requirements.txt
# 2. Convert to FP16 (intermediate artifact for every GGUF type)
python llama.cpp/convert.py MODEL_NAME --outtype f16 --outfile MODEL_NAME.fp16.bin
# 3. Quantize (Q4_K_M)
./llama.cpp/quantize MODEL_NAME.fp16.bin MODEL_NAME.Q4_K_M.gguf q4_k_m
```

**Block quant variants**: `Q4_0` scales and quantizes 32 values per block by the largest weight (`w = q × block_scale`); `Q4_1` adds the block minimum (`w = q × block_scale + block_minimum`); `Q4_K` uses super-blocks of 8 blocks × 32 values with 6-bit scales/minimums; i-quants (`IQ4_XS`) use QuIP#-style E8-lattice encoding with an even number of signs per group of eight.

Compatible backends: `llama-cpp-python`, LangChain, llama.cpp's server, LM Studio, Text Generation Web UI.

---

### 8. GPTQ and EXL2

GPU-dedicated formats, faster than GGUF because they avoid llama.cpp's CPU orientation.

- **GPTQ** (Frantar et al., 2023): refines Optimal Brain Quantization, uses a Cholesky decomposition of the Hessian inverse, updates columns in batches. Fixed at **4-bit**. Supported by TGI and TensorRT-LLM (book, page 342).
- **EXL2**: lets you pick any bitrate (2.0-8.0, e.g. 2.3, 3.5, 6.0) and mix per-layer precision, selecting combinations that minimize error at a target bitrate. Run by **ExLlamaV2**; offers the **highest throughput** of the three families. A 70B model can fit a single 24 GB GPU at ~2.55-bit. Supported by TGI (book, page 342).

```bash
# EXL2 quantization (book, page 342)
git clone https://github.com/turboderp/exllamav2
pip install -e exllamav2
python exllamav2/convert.py -i MODEL_NAME -o quant -c wikitext-test.parquet -b 4.5
```

GPTQ and EXL2 are less widely supported than GGUF (LM Studio does not integrate them), but they are integrated into Transformers, TGI, and (GPTQ) TensorRT-LLM.

---

### 9. Other Techniques

- **AWQ (Activation-aware Weight Quantization)**: protects salient weights chosen by **activation** magnitude (not weight magnitude), applies optimal per-channel scaling, no backprop, so the model cannot overfit the calibration set. Slightly slower than GPTQ/EXL2 but supported by TGI, vLLM, and TensorRT-LLM (book, page 342).
- **QuIP#** and **HQQ**: target 1-2 bit precision with better quality preservation; large models (>30B) at 2-bit can beat 7B/13B models at higher precision (book, page 342). The book expects this trend to continue.

---

### 10. Inference Engines

**Table 8.1, reproduced from the book (page 343), with the correct column alignment:**

| Technique | TGI | vLLM | TensorRT-LLM |
|-----------|-----|------|--------------|
| Continuous batching | ✓ | ✓ | ✓ |
| Speculative decoding | ✓ | | |
| FlashAttention-2 | ✓ | ✓ | ✓ |
| PagedAttention | ✓ | ✓ | ✓ |
| Pipeline parallelism | | | ✓ |
| Tensor parallelism | ✓ | ✓ | ✓ |
| GPTQ | ✓ | | ✓ |
| EXL2 | ✓ | | |
| AWQ | ✓ | ✓ | ✓ |

Reconciliation notes (why the alignment above is correct):
- **Speculative decoding is TGI-only** in the book. The surrounding text says "Speculative decoding is natively supported by TGI" (page 326). Modern vLLM and TensorRT-LLM have since added it, but the book's table marks one engine.
- **GPTQ is TGI + TensorRT-LLM**. The text says GPTQ "is also directly integrated into the transformers library and supported by TGI. GPTQ models are also supported in TensorRT-LLM" (page 342) - no mention of vLLM for GPTQ.
- **Pipeline parallelism is TensorRT-LLM-only**. "At the time of writing, only certain inference frameworks like TensorRT-LLM support pipeline parallelism" (page 330).
- **EXL2 is TGI-only**.
- A prior version of this document listed GPTQ under vLLM and speculative decoding under TensorRT-LLM; that was incorrect against the book. The table above is the book's.

Engine notes:
- **TGI** powers the Hugging Face SageMaker DLC used in the LLM Twin (Session 5.3).
- **vLLM** is used by the project's evaluation script (Session 7.3) for fast batch generation.
- **TensorRT-LLM** is the most feature-complete but the most NVIDIA-specific; it is the only one offering pipeline parallelism in the book's table.

---

## 🔧 How the LLM Twin Uses These Techniques

The project already applies several Chapter 8 ideas, even though the curriculum did not name them:

| Technique | Where in the repo |
|-----------|-------------------|
| Production quantization | `HF_MODEL_QUANTIZE="bitsandbytes"` in `deploy/huggingface/config.py` |
| Continuous batching + tensor parallelism | The Hugging Face **TGI** DLC behind the SageMaker endpoint |
| PagedAttention / fast batch generation | **vLLM** in `model/evaluation/evaluate.py` |
| 4-bit / 8-bit loading | `load_in_4bit` in `finetune.py`, 8-bit endpoint serving |

What is **not** wired in but is on the optimization table: static KV cache + `torch.compile`, speculative decoding, and GGUF/EXL2 export for local serving.

---

## 🛠️ Hands-On: Measure a KV Cache and Quantize a Model

### Step 1: Estimate the KV cache against your VRAM

```python
def kv_cache_gb(n_tokens, n_layers, n_kv_heads, dim_head, bytes_per=2):
    return (2 * n_tokens * n_layers * n_kv_heads * dim_head * bytes_per) / 1e9

# 7B MHA-style (32 kv heads), FP16
print(kv_cache_gb(2048, 32, 32, 128))   # ~1.07 GB at 2k tokens
print(kv_cache_gb(4096, 32, 32, 128))   # ~2.15 GB at 4k tokens
print(kv_cache_gb(8192, 32, 32, 128))   # ~4.29 GB at 8k tokens

# Llama-3.1-8B GQA (8 kv heads), FP16 - 4x smaller
print(kv_cache_gb(8192, 32, 8, 128))    # ~1.07 GB at 8k tokens
```

On a 16 GB card, an 8B FP16 model (~16 GB weights) does not fit at all; even 4-bit weights (~5 GB) plus a long KV cache plus activations is tight. This is why the book serves on `ml.g5.2xlarge` (24 GB).

### Step 2: Confirm BF16 is unavailable on this GPU

```python
import torch
print("bf16 supported:", torch.cuda.is_bf16_supported())  # False on Turing RTX 5000
```

Consequence: use FP16 for training and inference; do not enable `torch.autocast(dtype=torch.bfloat16)` or `bf16=True` on this card.

### Step 3: Confirm FlashAttention-2 is unavailable

```python
import torch
print(torch.cuda.get_device_capability())   # (7, 5) on the RTX 5000
print(torch.cuda.get_device_name(0))        # NVIDIA Quadro RTX 5000
```

`sm_75` < `sm_80`, so `attn_implementation="flash_attention_2"` will not work. Use SDPA instead:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    attn_implementation="sdpa",   # works on Turing; FA2 does not
)
```

### Step 4: Try a llama.cpp GGUF path (CPU-friendly)

Download a pre-quantized GGUF (for example a `Q4_K_M` community build) and run it with `llama-cpp-python`:

```python
from llama_cpp import Llama
llm = Llama(model_path="model.Q4_K_M.gguf", n_ctx=2048)
print(llm("Below is an instruction...\n### Instruction:\nWhat is an LLM Twin?\n### Response:\n", max_tokens=128)["choices"][0]["text"])
```

This is the most reliable local option on a 16 GB Turing card, because layers can spill to CPU when VRAM is short.

### Step 5: Compare engines conceptually

Run the same prompt through vLLM and (if you have an NVIDIA GPU with Ampere+) TensorRT-LLM, and record tokens/sec. On Turing, note that FlashAttention-2 is unavailable and that pipeline parallelism is TensorRT-LLM-only in the book's table.

---

## 📝 Exercise: Speculative Decoding Speedup

### Task

Measure the effect of a draft model.

1. Load a main model and a same-family draft model that share a tokenizer.
2. Generate 128 tokens with `assistant_model=draft_model` and without.
3. Record wall-clock time and output identity (speculative decoding is exact, so outputs must match for greedy decoding).
4. Explain why the speedup depends on the draft model's fidelity.

**Goal**: Internalize that speculative decoding trades spare compute for latency, and is lossless for greedy decoding.

---

## 📝 Exercise 2: Quantize and Measure the Quality/Memory Tradeoff

### Task

Quantize a small model and quantify what you gained and lost.

1. Load a 4-bit version of a small instruct model with `load_in_4bit=True` (bitsandbytes) and record `torch.cuda.max_memory_allocated()`.
2. Load the same model in FP16 and record the same metric.
3. On a fixed prompt set (10-20 prompts), compare the two outputs and score them by hand or with a rubric (coherence, factuality, instruction-following).
4. Compute the memory saved and state whether the quality cost is acceptable for your use case.

**Goal**: Turn "quantization loses some quality" into a measured number, and learn that the right bit width depends on the task, not on a default.

---

## 🐛 Common Pitfalls

- **Mismatched tokenizers** in speculative decoding: the draft tokens will not align; results are invalid.
- **Dynamic KV cache + `torch.compile`**: `torch.compile` needs static shapes; use `cache_implementation="static"`.
- **Static cache unsupported architecture**: the static cache does not work for all models; the book warns to check the Transformers docs.
- **FlashAttention-2 on Turing**: not supported (needs Ampere+). Do not force `attn_implementation="flash_attention_2"` on the RTX 5000; use SDPA.
- **Assuming bfloat16 works**: Turing has no BF16. `torch.cuda.is_bf16_supported()` returns False; code that hard-codes BF16 silently falls back or errors.
- **Ignoring GQA in KV budgets**: using `num_attention_heads` instead of `num_key_value_heads` overestimates the KV cache by 4x on modern models and leads to overly conservative batch sizes.
- **Double-counting the factor of 2**: the KV formula already includes `2` for K and V; applying it again doubles the estimate.
- **Quantization quality cliff**: 2-bit and below can degrade badly unless using QuIP#/HQQ on large models. Always evaluate after quantizing.
- **GPTQ is 4-bit only**: do not ask GPTQ for 3-bit or 5-bit; EXL2 is the flexible option.
- **Confusing throughput and latency under batching**: more batching raises throughput but also per-request latency; pick the right objective.
- **LLM.int8() overhead**: it saves memory but is ~20% slower for large models; do not assume INT8 is always faster.
- **Parallelism on one GPU**: DP/PP/TP require multiple devices; `SM_NUM_GPUS=1` means none apply locally, and TP additionally needs fast interconnect.
- **Benchmarking on an emulated arch**: FlashAttention-2 and BF16 results from an Ampere+ machine do not transfer to the Turing workstation.

---

## 🎓 Knowledge Check

1. **Why is token generation the bottleneck?**
   - Answer: It is inherently sequential; each token depends on all previous ones, so it under-uses the parallel hardware.

2. **What does the static KV cache unlock?**
   - Answer: Compatible shapes for `torch.compile`, up to ~4x faster forward passes.

3. **Write the KV cache size formula.**
   - Answer: `2 × n_tokens × n_layers × n_heads × dim_head × n_bytes`.

4. **How big is the KV cache for a 32-layer/32-head/128-dim model in FP16 at 4,096 tokens?**
   - Answer: `2 × 4096 × 32 × 32 × 128 × 2 ≈ 2.15 GB`.

5. **How does continuous batching improve utilization?**
   - Answer: It evicts finished requests and admits new ones immediately, keeping the batch full.

6. **Why must draft and main models share a tokenizer in speculative decoding?**
   - Answer: Draft tokens are validated by the main model, so they must refer to the same vocabulary.

7. **What is prompt lookup decoding?**
   - Answer: A speculative-decoding variant for input-grounded tasks that uses shared prompt n-grams as candidate tokens (`prompt_lookup_num_tokens`).

8. **Difference between pipeline and tensor parallelism?**
   - Answer: PP splits layers across GPUs (bubbles); TP splits weight matrices within a layer (needs fast interconnect).

9. **What does PagedAttention borrow from operating systems?**
   - Answer: Virtual memory and paging - KV cache in fixed-size blocks fetched on demand, not contiguous allocation.

10. **Which attention optimization requires Ampere or newer?**
    - Answer: FlashAttention-2 (the RTX 5000, being Turing, cannot use it; use SDPA).

11. **What is the difference between PTQ and QAT?**
    - Answer: PTQ converts a trained model directly (simple, slight loss); QAT quantizes during training (better quality, more expensive).

12. **Why is BF16 preferred over FP16 when available, and does the RTX 5000 have it?**
    - Answer: Neural nets prefer range over precision and BF16 has more exponent range; Turing has no BF16, so use FP16 locally.

13. **What does LLM.int8() do about outliers, and what is its cost?**
    - Answer: It processes outlier features in FP16 and the rest in INT8, with <1% degradation but ~20% slower inference on large models.

14. **Which format gives the highest GPU throughput, and which is fixed at 4-bit?**
    - Answer: EXL2 (via ExLlamaV2) gives the highest throughput; GPTQ is fixed at 4-bit.

15. **Per the book's Table 8.1, which engines support pipeline parallelism, speculative decoding, and GPTQ?**
    - Answer: Pipeline parallelism: TensorRT-LLM only. Speculative decoding: TGI only. GPTQ: TGI and TensorRT-LLM.

---

## 📖 Glossary

| Term | Meaning |
|------|---------|
| **Prefill** | Processing the prompt; parallel and compute-heavy. |
| **Decode** | Generating tokens one at a time; sequential and memory-bound. |
| **KV cache** | Stored key/value tensors reused across decode steps. |
| **GQA / MQA** | Grouped-query / multi-query attention; fewer K/V heads, smaller cache. |
| **Continuous batching** | Admitting new requests as others finish; a.k.a. in-flight batching. |
| **Speculative decoding** | Draft model proposes tokens; large model validates; a.k.a. assisted generation. |
| **Medusa** | Speculation heads trained on the main model. |
| **PagedAttention** | Block-based KV cache management inspired by OS paging. |
| **FlashAttention-2** | Tiled attention with online softmax; Ampere+ only. |
| **Data / pipeline / tensor parallelism** | Split batch / layers / weight matrices across GPUs. |
| **PTQ / QAT** | Quantize after training / during training. |
| **Absmax / zero-point** | Naïve symmetric / asymmetric INT8 quantization. |
| **NF4** | 4-bit format behind QLoRA. |
| **GGUF** | llama.cpp's quantized file format. |
| **GPTQ / EXL2** | GPU 4-bit / variable-bitrate (ExLlamaV2) formats. |
| **AWQ** | Activation-aware weight quantization. |
| **SDPA** | PyTorch `scaled_dot_product_attention`; the Turing-safe alternative to FA2. |

---

## 📚 References

- [Efficient Memory Management with PagedAttention (Kwon, Li, et al., 2023)](https://arxiv.org/abs/2309.06180)
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [Fast Inference from Transformers via Speculative Decoding (Leviathan, Kalman, Matias, 2023)](https://arxiv.org/abs/2211.17192)
- [Medusa (Cai et al., 2024)](https://arxiv.org/abs/2401.10774)
- [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323)
- [AWQ (Lin et al., 2023)](https://arxiv.org/abs/2306.00978)
- [LLM.int8() (Dettmers et al., 2022)](https://arxiv.org/abs/2208.07339)
- [QLoRA / NF4 (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ExLlamaV2](https://github.com/turboderp/exllamav2)
- [vLLM](https://docs.vllm.ai/) and [Text Generation Inference](https://huggingface.co/docs/text-generation-inference)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- Book: *LLM Engineer's Handbook*, Chapter 8 (pages 318-343).

---

## 🔗 Next Session

**Session 9.1 (book Chapter 9)**: RAG Inference Pipeline - see [Session 4.1 Advanced RAG](session_4.1_advanced_rag.md) and [Session 6.2 RAG Inference Flow](session_6.2_rag_inference_flow.md).

---

## 📚 Additional Resources

- [Hugging Face Transformers generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- Cross-links: [Session 5.1 SFT](session_5.1_sft.md), [Session 5.3 SageMaker Deployment](session_5.3_sagemaker_deployment.md), [Session 7.3 Model Evaluation](session_7.3_model_evaluation.md), [Session 6.1 FastAPI API](session_6.1_fastapi_api.md)

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Sessions 5.1, 5.3, 6.1

**Outcome**: You can explain and apply the full inference-optimization toolbox, compute KV-cache budgets accurately, reconcile the engine feature matrix with the book, and know which techniques your Turing GPU can and cannot use.
