# Fine-Tuning Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SUPERVISED FINE-TUNING PIPELINE                          │
│                         with Unsloth & LoRA                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: MODEL LOADING                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────────────┐                                             │
│   │  Hugging Face Hub        │                                             │
│   │  Meta-Llama-3.1-8B      │                                             │
│   │  (16GB, 8B parameters)   │                                             │
│   └────────────┬─────────────┘                                             │
│                │                                                             │
│                ▼                                                             │
│   ┌──────────────────────────┐                                             │
│   │  FastLanguageModel       │                                             │
│   │  .from_pretrained()      │                                             │
│   │  - Load model weights    │                                             │
│   │  - Load tokenizer        │                                             │
│   │  - Set max_seq=2048     │                                             │
│   └────────────┬─────────────┘                                             │
│                │                                                             │
│                ▼                                                             │
│   ┌──────────────────────────┐                                             │
│   │  Apply LoRA Config       │                                             │
│   │  - Rank: 32              │                                             │
│   │  - Alpha: 32             │                                             │
│   │  - Target: all layers    │                                             │
│   │  - Freeze base weights   │                                             │
│   │  - Add trainable LoRA    │                                             │
│   └────────────┬─────────────┘                                             │
│                │                                                             │
│                ▼                                                             │
│   ┌──────────────────────────┐                                             │
│   │  Model Ready for Training│                                             │
│   │  8B params (frozen)      │                                             │
│   │  + 130M LoRA (trainable) │                                             │
│   └──────────────────────────┘                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: DATA PREPARATION                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────┐         ┌──────────────────┐                       │
│   │  Dataset 1       │         │  Dataset 2       │                       │
│   │  llmtwin         │         │  FineTome        │                       │
│   │  3,000 samples   │         │  10,000 samples  │                       │
│   │  (domain-specific)│        │  (general)       │                       │
│   └────────┬─────────┘         └────────┬─────────┘                       │
│            │                            │                                   │
│            └────────────┬───────────────┘                                   │
│                         │                                                    │
│                         ▼                                                    │
│            ┌─────────────────────────┐                                     │
│            │  Concatenate Datasets   │                                     │
│            │  Total: 13,000 samples  │                                     │
│            └────────────┬────────────┘                                     │
│                         │                                                    │
│                         ▼                                                    │
│            ┌─────────────────────────┐                                     │
│            │  Format with Alpaca     │                                     │
│            │  Template               │                                     │
│            │                         │                                     │
│            │  ### Instruction:       │                                     │
│            │  {instruction}          │                                     │
│            │                         │                                     │
│            │  ### Response:          │                                     │
│            │  {output}<EOS>          │                                     │
│            └────────────┬────────────┘                                     │
│                         │                                                    │
│                         ▼                                                    │
│            ┌─────────────────────────┐                                     │
│            │  Train/Test Split       │                                     │
│            │  Train: 12,350 (95%)    │                                     │
│            │  Test:     650 (5%)     │                                     │
│            └─────────────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: TRAINING LOOP                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────┐              │
│   │  For each epoch (1, 2, 3):                             │              │
│   │                                                         │              │
│   │    For each batch (size=2, grad_accum=8):             │              │
│   │                                                         │              │
│   │      ┌──────────────────────────────────┐             │              │
│   │      │  1. Forward Pass                 │             │              │
│   │      │     Input: Tokenized text        │             │              │
│   │      │     Output: Next token predictions│            │              │
│   │      └───────────┬──────────────────────┘             │              │
│   │                  │                                     │              │
│   │                  ▼                                     │              │
│   │      ┌──────────────────────────────────┐             │              │
│   │      │  2. Calculate Loss               │             │              │
│   │      │     Cross-entropy between:       │             │              │
│   │      │     - Predicted tokens           │             │              │
│   │      │     - Actual tokens              │             │              │
│   │      └───────────┬──────────────────────┘             │              │
│   │                  │                                     │              │
│   │                  ▼                                     │              │
│   │      ┌──────────────────────────────────┐             │              │
│   │      │  3. Backward Pass                │             │              │
│   │      │     Compute gradients for:       │             │              │
│   │      │     - LoRA matrices (only!)      │             │              │
│   │      │     - Base model frozen          │             │              │
│   │      └───────────┬──────────────────────┘             │              │
│   │                  │                                     │              │
│   │                  ▼                                     │              │
│   │      ┌──────────────────────────────────┐             │              │
│   │      │  4. Accumulate Gradients         │             │              │
│   │      │     Every 8 steps:               │             │              │
│   │      │     - Update LoRA weights        │             │              │
│   │      │     - Apply AdamW 8-bit          │             │              │
│   │      │     - Clear gradients            │             │              │
│   │      └───────────┬──────────────────────┘             │              │
│   │                  │                                     │              │
│   │                  ▼                                     │              │
│   │      ┌──────────────────────────────────┐             │              │
│   │      │  5. Learning Rate Schedule       │             │              │
│   │      │     - Warmup: 0 → 3e-4           │             │              │
│   │      │     - Decay: 3e-4 → 0            │             │              │
│   │      └───────────┬──────────────────────┘             │              │
│   │                  │                                     │              │
│   │                  ▼                                     │              │
│   │      ┌──────────────────────────────────┐             │              │
│   │      │  6. Log Metrics                  │             │              │
│   │      │     - Training loss              │             │              │
│   │      │     - Validation loss            │             │              │
│   │      │     - Learning rate              │             │              │
│   │      │     - GPU usage                  │             │              │
│   │      │     → Send to Comet ML           │             │              │
│   │      └──────────────────────────────────┘             │              │
│   │                                                         │              │
│   └────────────────────────────────────────────────────────┘              │
│                                                                              │
│   Expected Duration: ~50 minutes on A100                                   │
│   Memory Usage: ~24GB GPU VRAM                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: MODEL OUTPUT & DEPLOYMENT                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────────────┐                                             │
│   │  Fine-Tuned Model        │                                             │
│   │  Base + LoRA merged      │                                             │
│   └────────────┬─────────────┘                                             │
│                │                                                             │
│                ├─────────────┐                                              │
│                │             │                                              │
│                ▼             ▼                                              │
│   ┌──────────────┐  ┌──────────────┐                                      │
│   │  Test        │  │  Save        │                                      │
│   │  Inference   │  │  Locally     │                                      │
│   │              │  │  model_sft/  │                                      │
│   └──────────────┘  └──────┬───────┘                                      │
│                            │                                                │
│                            ▼                                                │
│                   ┌──────────────────┐                                     │
│                   │  Upload to HF    │                                     │
│                   │  (optional)      │                                     │
│                   └────────┬─────────┘                                     │
│                            │                                                │
│                            ▼                                                │
│                   ┌──────────────────┐                                     │
│                   │  Deploy          │                                     │
│                   │  - API endpoint  │                                     │
│                   │  - vLLM/TGI      │                                     │
│                   │  - Production    │                                     │
│                   └──────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ LoRA ARCHITECTURE DETAIL                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Traditional Fine-Tuning:                                                  │
│   ┌────────────────┐                                                       │
│   │ Full Model     │  ← All 8B parameters trained                         │
│   │ 8B params      │  ← High memory usage                                 │
│   │ (all trainable)│  ← Slow training                                     │
│   └────────────────┘                                                       │
│                                                                              │
│   LoRA Fine-Tuning:                                                         │
│   ┌────────────────────────────────────────┐                              │
│   │ Base Model (Frozen)                    │                              │
│   │ 8B parameters                           │                              │
│   │                                         │                              │
│   │  Layer 1:                              │                              │
│   │  ┌──────────┐    ┌─────────┐          │                              │
│   │  │ W        │  + │ A × B   │          │  ← LoRA matrices            │
│   │  │ (frozen) │    │ (train) │          │     (rank 32)                │
│   │  └──────────┘    └─────────┘          │                              │
│   │                                         │                              │
│   │  Layer 2:                              │                              │
│   │  ┌──────────┐    ┌─────────┐          │                              │
│   │  │ W        │  + │ A × B   │          │                              │
│   │  │ (frozen) │    │ (train) │          │                              │
│   │  └──────────┘    └─────────┘          │                              │
│   │                                         │                              │
│   │  ... (all attention & FFN layers)      │                              │
│   │                                         │                              │
│   └────────────────────────────────────────┘                              │
│                                                                              │
│   Result:                                                                   │
│   - Only ~130M parameters trained (1.6%)                                   │
│   - 10x less memory                                                         │
│   - 2-3x faster training                                                    │
│   - Similar quality to full fine-tuning                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ MEMORY LAYOUT (24GB GPU)                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ████████████████████████████████████████████████ 16GB  Model Weights      │
│  ███ 2GB                                           Optimizer States         │
│  ██ 1GB                                            Gradients                │
│  █████ 4GB                                         Activations              │
│  █ 0.5GB                                           LoRA Adapters            │
│  █ 0.5GB                                           Other                    │
│  ─────────────────────────────────────────────────────────────────────    │
│  Total: ~24GB                                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING METRICS VISUALIZATION                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Training Loss:              Validation Loss:                              │
│                                                                              │
│  2.5 ┤                        2.4 ┤                                        │
│      │╲                           │╲                                       │
│  2.0 │ ╲                          │ ╲                                      │
│      │  ╲                         │  ╲                                     │
│  1.5 │   ╲                        │   ╲                                    │
│      │    ╲_                      │    ╲_                                  │
│  1.0 │      ╲_                    │      ╲_                                │
│      │        ╲_                  │        ──                              │
│  0.5 │          ─                 │                                        │
│      └─────────────────           └─────────────────                      │
│       Epoch 1   2   3              Epoch 1   2   3                        │
│                                                                              │
│  Good signs:                                                                │
│  ✓ Steady decrease                                                          │
│  ✓ Val loss doesn't diverge from train loss                               │
│  ✓ No sudden spikes                                                         │
│  ✓ Final val loss < 1.5                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA FLOW THROUGH TRAINING                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw Text                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │ Instruction: "Explain machine learning"                     │          │
│  │ Output: "Machine learning is a subset of AI that..."        │          │
│  └───────────────────────────┬─────────────────────────────────┘          │
│                               │                                             │
│                               ▼                                             │
│  Formatted (Alpaca Template)                                               │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │ ### Instruction:                                            │          │
│  │ Explain machine learning                                    │          │
│  │                                                             │          │
│  │ ### Response:                                               │          │
│  │ Machine learning is a subset of AI that...<EOS>            │          │
│  └───────────────────────────┬─────────────────────────────────┘          │
│                               │                                             │
│                               ▼                                             │
│  Tokenized                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │ [2534, 45, 8721, 12, 9845, ..., 3847, 2]                   │          │
│  │  ↑                              ↑     ↑                     │          │
│  │  "Explain"                     "AI"  <EOS>                 │          │
│  └───────────────────────────┬─────────────────────────────────┘          │
│                               │                                             │
│                               ▼                                             │
│  Model Processing                                                           │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │ For each token position:                                    │          │
│  │   Predict next token probability distribution               │          │
│  │   Compare with actual next token                            │          │
│  │   Calculate loss                                            │          │
│  │   Backpropagate gradients to LoRA matrices                 │          │
│  └───────────────────────────┬─────────────────────────────────┘          │
│                               │                                             │
│                               ▼                                             │
│  Updated Model                                                              │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │ LoRA weights adjusted to better predict:                    │          │
│  │ - Writing style                                             │          │
│  │ - Domain knowledge                                          │          │
│  │ - Response format                                           │          │
│  └─────────────────────────────────────────────────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ FILE STRUCTURE                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  llm_engineering/model/finetuning/                                          │
│  │                                                                           │
│  ├── unsloth_sft_standalone.py     ← Main training script                  │
│  │   └── 600+ lines, fully documented                                      │
│  │       - Phase 1: Authentication                                          │
│  │       - Phase 2: Model loading + LoRA                                   │
│  │       - Phase 3: Data preparation                                       │
│  │       - Phase 4: Training                                               │
│  │       - Phase 5: Testing                                                │
│  │       - Phase 6: Saving/uploading                                       │
│  │                                                                           │
│  ├── IMPLEMENTATION_SUMMARY.md     ← This file (high-level overview)       │
│  ├── FINE_TUNING_GUIDE.md          ← Comprehensive guide (8000+ words)     │
│  ├── QUICK_START.md                ← Quick tutorial (3000+ words)          │
│  ├── ARCHITECTURE_DIAGRAM.md       ← Visual diagrams (this file)           │
│  ├── requirements-unsloth.txt      ← Dependencies                          │
│  │                                                                           │
│  ├── finetune.py                   ← Existing: Core logic                  │
│  ├── sagemaker.py                  ← Existing: SageMaker integration       │
│  └── requirements.txt               ← Existing: Dependencies                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION TREE: When to Adjust What                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         Start Training                                      │
│                               │                                             │
│                               ▼                                             │
│                    ┌──────────────────┐                                    │
│                    │ Monitor Metrics  │                                    │
│                    └────────┬─────────┘                                    │
│                             │                                               │
│            ┌────────────────┼────────────────┐                             │
│            │                │                │                             │
│            ▼                ▼                ▼                             │
│    ┌──────────────┐ ┌─────────────┐ ┌──────────────┐                     │
│    │ Underfitting │ │   Good!     │ │ Overfitting  │                     │
│    │ High train & │ │ Both losses │ │ Low train,   │                     │
│    │   val loss   │ │  decrease   │ │ high val loss│                     │
│    └──────┬───────┘ └─────────────┘ └──────┬───────┘                     │
│           │                                  │                             │
│           ▼                                  ▼                             │
│    ┌──────────────┐                  ┌──────────────┐                     │
│    │ Increase:    │                  │ Decrease:    │                     │
│    │ • LoRA rank  │                  │ • Epochs     │                     │
│    │ • Epochs     │                  │ Add:         │                     │
│    │ • Learn rate │                  │ • Dropout    │                     │
│    └──────────────┘                  │ • Weight decay│                    │
│                                       │ • More data  │                     │
│                                       └──────────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Diagram Legend

**Symbols Used:**
- `┌─┐ └─┘`: Boxes for components
- `│ ─`: Connections and borders
- `▼ ▲`: Data flow direction
- `█`: Visual bars for memory/metrics
- `╲ ╱ ─`: Line graphs

**Color Coding (if viewing with syntax highlighting):**
- Headers: Section titles
- Components: Individual modules/steps
- Metrics: Numbers and measurements
- Notes: Additional information

## How to Read These Diagrams

1. **Top to Bottom**: Follow the flow from Phase 1 → Phase 4
2. **Left to Right**: Within each phase, read components left to right
3. **Arrows (▼)**: Show data/control flow between components
4. **Boxes**: Represent distinct components or stages
5. **Nested Boxes**: Show detailed views of larger components

## Quick Navigation

- **Overall Pipeline**: See "SUPERVISED FINE-TUNING PIPELINE" at top
- **LoRA Details**: See "LoRA ARCHITECTURE DETAIL" section
- **Memory Usage**: See "MEMORY LAYOUT" section
- **Training Progress**: See "TRAINING METRICS VISUALIZATION"
- **Data Processing**: See "DATA FLOW THROUGH TRAINING"
- **File Organization**: See "FILE STRUCTURE" section
- **Troubleshooting**: See "DECISION TREE" at bottom
