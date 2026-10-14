# Session 7.4: RAG Evaluation & Benchmarking (Book Chapter 7)

## 🎯 Learning Objectives

By the end of this session, you will:
- Distinguish ML evaluation from LLM evaluation
- Navigate general-purpose, domain-specific, and task-specific benchmarks
- Understand the three dimensions of RAG evaluation
- Use Ragas metrics (faithfulness, answer relevancy, context precision/recall)
- Explain the ARES three-stage pipeline and when to prefer it over Ragas
- Relate the project's own judge (accuracy/style) to this broader landscape
- Choose the right evaluation method for a given failure mode
- Recognize benchmark contamination, gaming, and judge bias

> This session completes **Book Chapter 7: Evaluating LLMs** (pp. 290-316). `session_7.3_model_evaluation.md` covers the project's hands-on judge; this session covers the evaluation theory and RAG frameworks the earlier curriculum omitted.

---

## 🏗️ Architecture Overview

```
LLM evaluation
├── Model evaluation        (a single model, no RAG, no prompt engineering)
│   ├── General-purpose     MMLU, HellaSwag, IFEval, MT-Bench, GAIA
│   ├── Domain-specific     Medical-LLM, BigCodeBench, Hallucinations, language LBs
│   └── Task-specific       ROUGE, accuracy/P/R/F1, multiple-choice, LLM-as-a-judge
└── RAG evaluation          (the whole system: retriever + generator)
    ├── Retrieval accuracy        retrieval precision/recall
    ├── Integration quality       how well context is used
    └── Factuality & relevance    grounded and on-topic
         ├── Ragas   (LLM-assisted metrics + synthetic data)
         └── ARES    (fine-tuned classifiers + synthetic data)
```

### The central distinction

**Model evaluation** isolates the model. There is no retriever, no prompt engineering, no post-processing. You are measuring what the weights can do.

**RAG evaluation** measures a *system*. The retriever can fetch the wrong documents, the prompt can bury the right facts, and the generator can ignore context. A model that aces MMLU can still fail a RAG task because the failure was upstream.

```
Model-only view:     input ──► [ LLM ] ──► output        measure output
RAG-system view:     query ──► [ Retriever ] ──► context ──► [ LLM ] ──► answer
                                ▲                              ▲
                                └── retrieval accuracy         └── faithfulness
                                    context precision/recall        answer relevancy
```

This is why a RAG system needs metrics that a model benchmark cannot provide.

### A decision tree for choosing an evaluation

```
What am I evaluating?
│
├── A base model, to pick a fine-tuning anchor?
│      → general-purpose suite (MMLU + reasoning)
│
├── A fine-tuned instruct model?
│      → IFEval + MT-Bench + AlpacaEval + LLM-as-a-judge
│
├── A domain model (medical/code/Arabic)?
│      → the matching domain leaderboard + translated general benchmarks
│
├── One specific downstream task?
│      → task-specific metrics (ROUGE, F1) or custom LLM-judge
│
└── A full RAG system?
       → Ragas (fast, LLM-assisted) and/or ARES (trained classifiers)
```

---

## 📁 Model Evaluation

### 1. ML vs LLM Evaluation

| Aspect | ML models | LLMs |
|--------|-----------|------|
| Metrics | Accuracy, precision, recall, MSE | Multiple tasks, rarely a single metric |
| Feature engineering | Manual, part of evaluation | Raw text directly, little feature engineering |
| Interpretability | Direct | Requires requesting explanations during generation |
| Task scope | Narrow, well-defined | Open-ended, many tasks |
| Data type | Numeric / categorical | Free-form text |

The root cause: ML models solve narrowly defined problems on structured data, so a single number (accuracy, MSE) summarizes performance. LLMs interpret and generate language, which adds subjectivity and demands qualitative assessment alongside numbers.

- **Numerical metrics** are less decisive for LLMs because one model does many tasks; there is rarely one metric.
- **Feature engineering** disappears for LLMs because the model consumes raw text.
- **Interpretability** is harder; you ask the model to explain itself rather than inspecting weights.

### 2. General-Purpose Evaluations

**During pre-training** (low-level, cheap):
- **Training loss**, **validation loss**: cross-entropy.
- **Perplexity**: `exp(loss)`; how surprised the model is (lower is better).
- **Gradient norm**: detect instability or vanishing/exploding gradients.

**After pre-training** (base model):
- **MMLU** (knowledge): 57-subject multiple choice.
- **HellaSwag** (reasoning): plausible story endings.
- **ARC-C**, **Winogrande**, **PIQA**: reasoning variants.

**After fine-tuning** (instruct/alignment):
- **IFEval** (instruction following with constraints).
- **Chatbot Arena** (human head-to-head votes).
- **AlpacaEval** (automatic, correlates with Arena).
- **MT-Bench** (multi-turn conversation).
- **GAIA** (agentic, tool use).

**Benchmark taxonomy table**:

| Benchmark | Phase | Capability | Format | Scoring |
|-----------|-------|------------|--------|---------|
| Training/val loss | Pre-training | learning progress | next-token | cross-entropy |
| Perplexity | Pre-training | surprise | next-token | exp(loss) |
| MMLU | Post pre-training | knowledge | 57-subj MCQ | accuracy |
| HellaSwag | Post pre-training | common-sense reasoning | completion MCQ | accuracy |
| ARC-C | Post pre-training | causal reasoning | science MCQ | accuracy |
| Winogrande | Post pre-training | common-sense reasoning | pronoun resolution | accuracy |
| PIQA | Post pre-training | physical common sense | MCQ | accuracy |
| IFEval | Post fine-tuning | instruction following | constrained gen | rule-based |
| Chatbot Arena | Post fine-tuning | conversation | human votes | Elo |
| AlpacaEval | Post fine-tuning | instruction following | auto judge | win rate |
| MT-Bench | Post fine-tuning | multi-turn | LLM judge | 1-10 |
| GAIA | Post fine-tuning | agentic / tool use | multi-step tasks | pass rate |

**Watch for contamination**: a MMLU jump of ~10 points after fine-tuning is implausible and signals test leakage. Benchmarks are **signals, not truth**, and can be gamed; human evaluation itself favours long, confidently formatted answers. Private test sets avoid gaming but are less scrutinized and have their own biases.

**A note on phase-appropriate choice**: to fine-tune, pick the best base model by knowledge and reasoning at a given size (MMLU + reasoning suite). If you only need information extraction, instruction-following (IFEval) matters more than conversation (MT-Bench). Match the benchmark to the downstream job, not to leaderboard prestige.

### 3. Domain-Specific Evaluations

- **Open Medical-LLM Leaderboard**: 9 metrics, 1,273 MedQA (US license exam), 500 PubMedQA, 4,183 MedMCQA (Indian entrance), 1,089 from 6 medical MMLU sub-categories.
- **BigCodeBench**: `Complete` (structured docstrings) and `Instruct` (natural language); Pass@1, greedy decoding, plus an Elo for the Complete variant.
- **Hallucinations Leaderboard**: 16 tasks across QA (NQ Open, TruthfulQA, SQuADv2), Reading Comprehension (TriviaQA, RACE), Summarization (HaluEval Summ, XSum, CNN/DM), Dialogue (HaluEval Dial, FaithDial), and Fact Checking (MemoTrap, SelfCheckGPT, FEVER, TrueFalse), plus IFEval.
- **Enterprise Scenarios**: FinanceBench (100 financial questions with context), Legal Confidentiality (100 LegalBench prompts), Writing Prompts, Customer Support Dialogue, Toxic Prompts, Enterprise PII. Some test sets are closed-source to prevent gaming.
- **Language leaderboards**:
  - **OpenKo-LLM** (Korean): 9 metrics mixing translated general benchmarks (GPQA, Winogrande, GSM8K, EQ-Bench, IFEval) with custom native ones (Knowledge, Social Value, Harmlessness, Helpfulness).
  - **Open Portuguese LLM**: educational (ENEM 1,430, BLUEX 724), professional (OAB 2,000+), language understanding (ASSIN2 RTE/STS, FAQUAD NLI), social media (HateBR 7,000 Instagram comments, PT Hate Speech 5,668 tweets, tweetSentBR).
  - **Open Arabic LLM Leaderboard**: native Arabic tasks (AlGhafa, Arabic-Culture-Value-Alignment) plus 12 translated benchmarks (MMLU, ARC-Challenge, HellaSwag, PIQA).

Three design principles: **complex** (discriminate good from bad), **diverse** (cover many scenarios), **practical** (easy to run via `lm-evaluation-harness` or Hugging Face `lighteval`).

**Domain suites differ in shape**: BigCodeBench uses only two metrics because two capture the domain. The Hallucinations Leaderboard regroups 16. The lesson is that a suite can reuse general benchmarks alongside custom ones - language leaderboards routinely translate general benchmarks and add native tasks. Prefer **human-translated** benchmarks over machine-translated ones.

### 4. Task-Specific Evaluations

- **Structured tasks reuse ML metrics**: ROUGE for summarization, accuracy/precision/recall/F1 for classification/NER.
- **Multiple-choice question answering** (e.g. MMLU): two modes - **text generation** (model emits A/B/C/D) or **log-likelihood** (compare option probabilities). The book recommends **text generation** because it mimics human test-taking and is more discriminative.
- **Open-ended tasks**: LLM-as-a-judge with a structured prompt and scale, optionally with ground-truth context.

**Worked MCQ example** (from the book, MMLU abstract algebra):

```
Instruction
Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.
A. 0
B. 4
C. 2
D. 6

Output
B
```

- **Text generation mode**: the model emits a letter, checked against the key. Discriminative; tests real-world format.
- **Log-likelihood mode**: compare the model's probability for each full answer string. More nuanced confidence, but low-quality models overperform because a short option string can have high probability without real reasoning.

**The book's recommendation**: use text generation. It is easier to implement, mimics human test-taking, and penalizes weak models that skate by on likelihood.

**Judge biases**: assertive/verbose answers rated higher; weak domain expertise; inconsistent scoring; style preferences. Mitigate with multiple judges, ground truth, and careful prompts.

**A structured judge prompt template** (the book's Table 7.2 pattern):

```
You are an evaluator who assesses the quality of an answer to an instruction.
Your goal is to provide a score that represents how well the answer addresses the instruction.
You will use a scale of 1 to 4, where each number represents the following:
1. The answer is not relevant to the instruction.
2. The answer is relevant but not helpful.
3. The answer is relevant and helpful but could be more detailed.
4. The answer is relevant, helpful, and detailed.
Please provide your evaluation as follows:
##Evaluation##
Explanation: (analyze the relevance, helpfulness, and complexity of the answer)
Total rating: (final score as a number between 1 and 4)
Instruction:
{instruction}
Answer:
{answer}
```

Note the trailing `##Evaluation##` and `Explanation:` markers: they prime the model to continue in the exact format, and the `Explanation` first forces reasoning before the score.

### 5. Building a custom benchmark end to end

When no existing benchmark fits, compose one. The repeatable recipe:

```
1. Choose the failure mode you must catch (factuality? format? tone? safety?).
2. Pick a format: MCQ (cheap, objective), reference text (ROUGE/EM/F1), or open-ended (LLM judge).
3. Assemble items: reuse public datasets, then add domain-specific ones.
4. Hold out a private slice the model has never seen.
5. Fix the scoring script and the prompt format - they must never change mid-comparison.
6. Record baseline scores for 2-3 known models to sanity-check the scale.
7. Re-run whenever the model, data, or prompt changes.
```

**Anti-pattern**: changing the rubric between model A and model B. A benchmark is only comparable if its items, prompt, and scorer are frozen. Version the benchmark alongside the model.

### Worked example: an Arabic/Islamic RAG evaluation suite

For a domain like Athar (Islamic-knowledge RAG), a defensible suite layers three levels:

| Level | Example benchmark | Catches |
|-------|-------------------|---------|
| Native language | Open Arabic LLM Leaderboard (AlGhafa, Arabic-Culture-Value-Alignment) | Arabic fluency and cultural grounding |
| Translated general | translated MMLU / ARC / HellaSwag | baseline reasoning in Arabic |
| Task-specific | Ragas faithfulness + context recall on Quran/Hadith chunks | retrieval and citation grounding |

The first two answer *"is the model competent in Arabic?"*; the third answers *"does the RAG system cite the right verses and not hallucinate?"*. No single level is sufficient. This mirrors the book's own layering of general, domain, and task-specific evaluations.

---

## 📁 RAG Evaluation

RAG evaluation must cover the whole system, not just the LLM:

1. **Retrieval accuracy**: did the system fetch relevant documents?
2. **Integration quality**: is the retrieved information used effectively?
3. **Factuality and relevance**: is the final answer grounded and on-topic?

Key metrics: retrieval precision/recall, integration quality, output factuality/coherence.

### The customer-support worked example

The book's running example: a user asks *"What's your return policy for laptops purchased during the holiday sale?"* The retriever finds two documents - one on the electronics return policy, one on holiday sale terms - appends them to the question, and the model answers:

```
For laptops purchased during our holiday sale, you have an extended return period of 60 days
from the date of purchase. This is longer than our standard 30-day return policy for electronics.
Please ensure the laptop is in its original packaging with all accessories to be eligible for a
full refund.
```

Three evaluations fall out of this single example:

| Dimension | Question | How to check |
|-----------|----------|--------------|
| Retrieval accuracy | Did the two documents match what was expected? | precision/recall against a labelled set |
| Integration quality | Does the answer change when context is added? | compare with/without context |
| Factuality/relevance | Is the answer grounded in the documents? | faithfulness + relevance metrics |

### Ragas

**RAG Assessment (Ragas)** is an open-source toolkit built on **metrics-driven development (MDD)**: continuously monitor metrics to guide product decisions.

- **Synthetic test generation**: uses an **Evol-Instruct**-style evolutionary approach to create complex questions (reasoning, conditional, multi-context) and conversational samples.
- **Metrics**:
  - **Faithfulness**: decomposes the answer into claims and verifies each is inferable from the context. Score = verifiable claims / total claims.
  - **Answer relevancy**: generates questions *from the answer* and measures mean cosine similarity to the original question; catches factually-correct-but-off-topic answers.
  - **Context precision**: are relevant items ranked at the top?
  - **Context recall**: can each ground-truth claim be attributed to the retrieved context?
- **Production monitoring**: building blocks to monitor RAG quality continuously.

```
Ragas faithfulness, step by step
  answer: "Laptops have a 60-day holiday return window."
     │ decompose
     ▼
  claims: ["Laptops have a holiday return window.", "The window is 60 days."]
     │ verify each against retrieved context
     ▼
  claim 1 → supported   claim 2 → supported
     │
     ▼
  faithfulness = supported / total = 2/2 = 1.0
```

```
Ragas answer relevancy, step by step
  original question: "What is the holiday return window for laptops?"
     │ generate candidate questions FROM the answer
     ▼
  "How long can I return a laptop bought on sale?"
  "What is the laptop holiday return period?"
     │ embed and average cosine similarity to the original question
     ▼
  relevancy = mean(sim)  (low if the answer drifts off-topic)
```

**Why faithfulness and relevance are different**: an answer can be perfectly faithful (every claim in the context) yet off-topic, or on-topic yet hallucinated. You need both axes.

### ARES

**ARES** (Automated evaluation framework for RAG systems) combines synthetic data + fine-tuned classifiers, in three configurable stages:

1. **Synthetic data generation**: mimic real scenarios (document paths, few-shot prompts). Default LM: `google/flan-t5-xxl`.
2. **Classifier training**: high-precision classifiers for relevance/faithfulness. Default model: `microsoft/deberta-v3-large`; configurable epochs, patience, learning rate.
3. **RAG evaluation**: run classifiers on synthetic data; produces confidence intervals. Supports cloud or local (**vLLM**) execution and multiple artifact types (code, HTML, images).

```
ARES pipeline

  documents ──► (1) synthetic data generation ──► plausible queries + labelled answers
                       LM: flan-t5-xxl                      │
                                                            ▼
                        (2) classifier training ──► relevance + faithfulness classifiers
                               deberta-v3-large              │
                                                            ▼
                        (3) RAG evaluation ──► scores + confidence intervals
                               on synthetic + gold data      (cloud or vLLM local)
```

**Ragas vs ARES**:

| Dimension | Ragas | ARES |
|-----------|-------|------|
| Core mechanism | LLM-assisted metrics | fine-tuned classifiers |
| Setup cost | low (call an LLM) | higher (train classifiers) |
| Consistency | varies with the judge LLM | high once trained |
| Speed at scale | per-call LLM latency+ cost | fast after training |
| Configurability | prompt-level | epochs, LR, patience, model |
| Strength | iteration + production monitoring | deep, customized checkpoints |
| Default models | judge LLM (e.g. gpt-4o-mini) | flan-t5-xxl, deberta-v3-large |

**Combine them**: Ragas for quick loops, ARES for deep, customized evaluation at key checkpoints. This is the book's explicit recommendation.

---

## 🔬 Deep Dive: Metric Math and Worked Examples

### Context precision, computed

Context precision rewards relevant retrieved items that appear near the top. A common formulation is the mean of precision@k evaluated at each rank where a relevant item appears:

```
retrieved = [relevant, not, relevant, not]   (relevant = 1,0,1,0)

precision@1 = 1/1 = 1.0        (rank 1 is relevant)
precision@3 = 2/3 = 0.667      (ranks 1..3 contain 2 relevant)
context_precision = mean(1.0, 0.667) = 0.833
```

If the same two relevant items were pushed to ranks 2 and 4, the score falls - the retriever put junk first. This is why precision is a *ranking* metric, not a set metric.

### Context recall, computed

Context recall asks whether each ground-truth claim is attributable to the retrieved context:

```
ground_truth claims: [c1, c2, c3]
attributable to context: [yes, yes, no]
context_recall = 2/3 = 0.667
```

Low recall means the retriever missed material the answer needed. It requires an annotated (ground-truth) answer; without one you cannot compute it.

### Faithfulness vs correctness, side by side

| Case | Context | Answer | Faithful? | Correct? |
|------|---------|--------|-----------|----------|
| Good | "60-day window" | "60-day window" | yes | yes |
| Hallucinated | "60-day window" | "90-day window" | no | no |
| Faithful-but-wrong-context | "90-day window" (wrong doc) | "90-day window" | yes | no |
| Off-topic | "60-day window" | "Our stores open at 9am" | yes | irrelevant |

The third row is the trap: faithfulness measures grounding in *whatever context was retrieved*, not truth. If retrieval is bad, a faithful answer is still wrong. This is why RAG evaluation needs both retrieval metrics and faithfulness.

### Benchmark contamination, how to detect it

```
suspicious signal: MMLU jumps +10 points after fine-tuning on 13k samples

checks:
  1. Inspect the fine-tuning data for benchmark items (exact + fuzzy match).
  2. Evaluate on a private held-out set the model never saw.
  3. Compare n-gram overlap between train and benchmark questions.
  4. Re-run the benchmark with paraphrased items; a real gain survives.
  5. Compare against a public leaderboard submission; inflated scores diverge.
```

A genuine capability gain generalizes; contamination does not survive paraphrasing or a private set.

### Judge juries

A single judge carries style and self-preference bias. A jury of 3 judges with majority voting cuts variance at 3x cost:

```
answer ──► judge A ──► score 2
       ──► judge B ──► score 3
       ──► judge C ──► score 3
       majority = 3   (and report disagreement rate)
```

Report the inter-judge agreement. Low agreement signals an ambiguous rubric, not a good model.

---

## 🛠️ Tooling: lm-evaluation-harness and lighteval

The book recommends these to run benchmarks instead of reimplementing them.

**lm-evaluation-harness** (EleutherAI) - the standard batch runner:

```bash
# after installing, run MMLU with few-shot examples on a local model
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B \
        --tasks mmlu,hellaswag,arc_challenge,winogrande,piqa \
        --batch_size 8 \
        --output_path ./eval_results
```

**lighteval** (Hugging Face) - a lighter, HF-integrated runner:

```bash
lighteval accelerate \
  "pretrained=meta-llama/Llama-3.1-8B" \
  "mmlu|5,leaderboard|0" \
  --output-dir ./lighteval_results
```

| Tool | Strength | Best for |
|------|----------|----------|
| lm-evaluation-harness | widest task coverage, the de-facto standard | reproducing leaderboard numbers |
| lighteval | HF ecosystem, faster to wire | custom and translated benchmarks |

**Why not hand-roll a benchmark**: a benchmark is only meaningful if everyone computes it identically. Using the reference harness guarantees the prompt format, few-shot count, and scoring match the published numbers.

---

## 🧪 Edge Cases and Failure Modes

| Failure | Symptom | Root cause | Fix |
|---------|---------|-----------|-----|
| Grounding mistaken for truth | Faithfulness 1.0 but wrong answer | Retrieved context is itself wrong | Add retrieval precision/recall; audit the corpus |
| Recall uncomputable | Metric errors or is skipped | No ground-truth answer in the row | Supply `ground_truth` or drop context_recall |
| Ragas cost blowup | Surprise LLM bill | Each metric issues judge calls per row | Cache embeddings; batch; sample the eval set |
| ARES classifier overfits | Great scores on synthetic, poor on real | Synthetic data distribution is narrow | Mix in a small real gold set |
| Leaderboard gaming | A model tops a public bench, fails in prod | Test items leaked to training | Prefer private/paraphrased sets |
| Machine-translation noise | Arabic/other model ranks oddly | MT artifacts in the benchmark | Use human-translated benchmarks |
| Metric orthogonality ignored | "We score 0.8 overall" | Single aggregate hides which axis failed | Report all four Ragas metrics separately |

### Design alternatives for a RAG eval

| Approach | Cost | Consistency | Best at |
|----------|------|-------------|---------|
| Ragas (LLM judge) | per-call LLM cost | medium (judge-dependent) | fast iteration, production monitoring |
| ARES (trained classifiers) | training cost upfront | high after training | stable checkpoints |
| Human annotation | highest | varies by annotator | validating the other two |
| Reference metrics only (EM/F1) | none | high | extractive QA with gold spans |
| A/B in production | requires traffic | real-world | final acceptance |

The strongest setup layers them: reference metrics for cheap regression checks, Ragas for fast iteration, ARES at release gates, and a small human set to validate the judges.

---

## 🔧 How the LLM Twin Evaluates Today

The project implements a **task-specific** benchmark, not a general one:

- `llm_engineering/model/evaluation/evaluate.py` uses **vLLM** (Session 8.4) to generate answers for the held-out test split.
- A **pointwise LLM-as-a-judge** (`gpt-4o-mini`) scores each answer on **accuracy** and **style**, 1-3.
- It compares **SFT**, **DPO**, and the **Instruct** baseline.

This satisfies the book's "evaluating TwinLlama-3.1-8B" section. What is **not** implemented: Ragas, ARES, and the general/domain benchmark suites. For a RAG system, Ragas is the natural next step because it measures retrieval and grounding, not just answer quality.

**Mapping to the book's judge**: the book's own evaluation prompt (1-3 accuracy/style) is exactly what `session_7.3_model_evaluation.md` documents; the general-purpose and RAG frameworks in this session are the wider context around it.

**Where the current design would fail**: the accuracy/style judge scores the final answer only. It cannot tell whether a poor answer came from a bad retriever or a bad generator. Ragas context precision/recall isolates the retriever; faithfulness isolates the generator. Adding Ragas is what makes the failure attributable.

---

## 🛠️ Hands-On: Evaluate a RAG Answer with Ragas

### Step 1: Install Ragas

```bash
pip install ragas
```

> Installing a dependency changes your environment; confirm before running in a shared setup.

### Step 2: Build a tiny eval set

```python
from datasets import Dataset

data = {
    "question": ["What is our holiday return policy for laptops?"],
    "contexts": [[
        "Electronics have a standard 30-day return window. "
        "During the holiday sale, laptops get an extended 60-day return period."
    ]],
    "answer": ["Laptops bought during the holiday sale have a 60-day return window, longer than the standard 30 days."],
    "ground_truth": ["Laptops purchased during the holiday sale can be returned within 60 days."],
}
ds = Dataset.from_dict(data)
```

### Step 3: Score with Ragas metrics

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
```

Interpretation:
- **High faithfulness**: the answer does not hallucinate beyond the context.
- **High answer relevancy**: the answer addresses the question.
- **High context precision/recall**: the retriever surfaced the right material.

### Step 4: Diagnose a failure

Deliberately corrupt one field and observe which metric moves:

| Corrupted field | Expected metric to drop |
|-----------------|-------------------------|
| Add a claim not in any context | faithfulness |
| Answer a different question | answer relevancy |
| Reorder contexts so the useful one is last | context precision |
| Remove a context needed by the ground truth | context recall |

**Goal**: Learn that the four metrics are orthogonal - each catches a distinct failure mode.

---

## 📝 Exercise: Choose the Right Benchmark

### Task

For each scenario, pick the evaluation approach and justify it.

| Scenario | Your choice | Why |
|----------|-------------|-----|
| Choose a base model to fine-tune | MMLU + reasoning suite | knowledge and reasoning before task fit |
| A bilingual (Arabic/English) assistant | Open Arabic LLM Leaderboard + translated MMLU | native + translated coverage |
| A customer-support RAG bot | Ragas (faithfulness, context recall) | retrieval and grounding matter |
| A code-completion product | BigCodeBench Pass@1 | task-specific code correctness |
| A creative writing assistant | MT-Bench / AlpacaEval + LLM judge | open-ended quality and style |

Then implement one: run Ragas on the project's RAG output and record the four metrics.

**Goal**: Match the evaluation method to the failure mode you actually care about.

### Second Exercise: Design a Domain Suite

Build a 3-benchmark suite for a new niche (for example, Egyptian-Arabic legal Q&A).

1. Pick one **native** benchmark that only your domain has (analogous to AlGhafa).
2. Pick one **translated general** benchmark to anchor the model's baseline ability.
3. Pick one **task-specific** metric for the actual delivery format (e.g. exact clause extraction with F1).
4. For each, state the failure mode it catches and the one it misses.
5. Justify why the three together are stronger than any single one.

**Goal**: Internalize the book's "complex, diverse, practical" design principles by applying them.

---

## 🔁 Production Monitoring with Ragas (the MDD loop)

Metrics-driven development (MDD) means you do not evaluate once and walk away. You wrap the online path so every production request produces metrics, then watch the trend.

```
        ┌──────────────────────────────────────────────────────┐
        │  Production RAG request                                │
        │  query → retrieve → generate → answer                  │
        └───────────────────────┬──────────────────────────────┘
                                ▼
                  sample N requests per window
                                ▼
             compute faithfulness / relevancy / context metrics
                                ▼
                    dashboard + alert thresholds
                                ▼
            drift detected → inspect traces → ship a fix → re-evaluate
```

**What to monitor**:

| Signal | What a drop means | First place to look |
|--------|-------------------|---------------------|
| Faithfulness | generator hallucinating | prompt or model version |
| Answer relevancy | answers drifting off-topic | retrieval or query rewriting |
| Context precision | retriever ranking junk first | embedding model or reranker |
| Context recall | retriever missing documents | index freshness or chunking |
| Latency + token counts | cost/regression creep | prompt length, model swap |

The project already emits token metadata per request through Opik (`session_7.2_opik_monitoring.md`): `query_tokens`, `context_tokens`, `answer_tokens`. Adding Ragas on a sampled subset turns that trace stream into a quality signal, which is the step from "monitoring" to "evaluating".

**Why sample rather than score every request**: Ragas metrics each cost judge LLM calls. Scoring 100% of traffic can cost more than generation itself. Score a random sample per hour and alert on its moving average.

### Case study: evaluating the LLM Twin's RAG end to end

The project's `/rag` endpoint (`llm_engineering/infrastructure/inference_pipeline_api.py`) retrieves `k=3` documents and generates an answer. A complete evaluation layers the book's three dimensions onto that exact path:

```python
# Conceptual wiring: score the live RAG output with Ragas.
# The repo does NOT ship this - it is the intended next step.
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

rows = {
    "question": [], "contexts": [], "answer": [], "ground_truth": [],
}
for sample in eval_samples:
    documents = retriever.search(sample["query"], k=3)
    context = EmbeddedChunk.to_context(documents)
    rows["question"].append(sample["query"])
    rows["contexts"].append([context])
    rows["answer"].append(call_llm_service(sample["query"], context))
    rows["ground_truth"].append(sample["expected_answer"])

result = evaluate(Dataset.from_dict(rows),
                  metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
```

What each metric tells you about the two-microservice design (see `session_10.1_deployment_topologies.md`):

| Metric | Which microservice it blames |
|--------|------------------------------|
| faithfulness | the LLM microservice (generation) |
| answer relevancy | the LLM microservice (off-topic drift) |
| context precision | the business microservice + retriever (ranking) |
| context recall | the business microservice + index (coverage) |

This attribution is the whole reason to add Ragas: the project's accuracy/style judge scores the final answer and cannot say whether a weak answer came from retrieval or generation.

---

## 🐛 Common Pitfalls

- **Judging a RAG system with a model-only benchmark**: MMLU tells you nothing about retrieval or grounding.
- **Contaminated benchmarks**: a suspiciously large jump implies leakage, not improvement.
- **Trusting one signal**: benchmarks are noisy and gameable; triangulate with several.
- **Judge biases**: verbosity, position (pairwise), and intra-model favouritism. Use juries and randomization.
- **Ragas needs an LLM**: its metrics are LLM-assisted and cost money; budget accordingly.
- **Confusing faithfulness with correctness**: faithfulness checks grounding in the *provided* context; if the context itself is wrong, the answer can be faithful and wrong.
- **Context recall without ground truth**: context recall requires an annotated answer; without it you can only measure precision.
- **ARES setup cost**: training classifiers is not free; skip ARES until you have a stable checkpoint to evaluate, or use it sparingly.
- **Machine-translated benchmarks**: prefer human-translated ones; MT noise can misrank models.

---

## 🎓 Knowledge Check

1. **Name the three dimensions of RAG evaluation.**
   - Answer: retrieval accuracy, integration quality, and factuality/relevance.

2. **What does Ragas' faithfulness measure?**
   - Answer: the fraction of answer claims that can be inferred from the provided context.

3. **How does Ragas compute answer relevancy?**
   - Answer: it generates questions from the answer and compares them to the original via cosine similarity.

4. **What are the three ARES stages?**
   - Answer: synthetic data generation, classifier training, and RAG evaluation.

5. **Why prefer text-generation over log-likelihood for multiple-choice evaluation?**
   - Answer: it mimics human test-taking, is simpler, and is more discriminative (weak models overperform on likelihood).

6. **Which benchmark type does the LLM Twin's `evaluate.py` implement?**
   - Answer: a task-specific pointwise LLM-as-a-judge on accuracy and style.

7. **What is the phase-appropriate metric set during pre-training?**
   - Answer: training/validation loss, perplexity, and gradient norm.

8. **What does context precision reward that context recall does not?**
   - Answer: the *ranking* of relevant items - relevant context near the top scores higher.

9. **When is ARES preferable to Ragas?**
   - Answer: for consistent, faster evaluations at scale once classifiers are trained - typically at key checkpoints.

10. **What default models does ARES use?**
    - Answer: `google/flan-t5-xxl` for synthetic generation and `microsoft/deberta-v3-large` for classifiers.

11. **Why can a 10-point MMLU jump after fine-tuning be suspicious?**
    - Answer: It is implausibly large and usually signals test contamination.

12. **What three design principles should a benchmark suite satisfy?**
    - Answer: complex, diverse, and practical.

13. **Why combine Ragas and ARES?**
    - Answer: Ragas enables fast iteration and production monitoring; ARES gives deeper, consistent checkpoint evaluation.

14. **Why is benchmark contamination worse for public test sets?**
    - Answer: Public sets can be trained on directly; private sets avoid this but are less scrutinized and carry their own biases.

15. **Which Mintaka-style benchmark tests agentic, multi-step ability?**
    - Answer: GAIA (tool use and web browsing).

---

## 📖 Glossary

- **Benchmark**: a standardized dataset and scoring procedure for comparing models.
- **Contamination**: test data leaking into training, inflating scores.
- **MMLU**: Massive Multitask Language Understanding; 57-subject multiple choice.
- **IFEval**: instruction-following evaluation with explicit constraints.
- **MT-Bench**: multi-turn conversational benchmark scored by an LLM judge.
- **GAIA**: agentic benchmark for tool use and multi-step tasks.
- **Faithfulness**: fraction of answer claims grounded in the retrieved context.
- **Answer relevancy**: how well the answer addresses the original question.
- **Context precision**: whether relevant retrieved items are ranked highly.
- **Context recall**: whether all ground-truth-supported content is present in the context.
- **Ragas**: RAG Assessment; LLM-assisted RAG evaluation toolkit built on MDD.
- **ARES**: Automated RAG Evaluation System; synthetic data + fine-tuned classifiers.
- **MDD**: metrics-driven development; continuously monitoring metrics to guide decisions.
- **Evol-Instruct**: an evolutionary question-complexity generation approach Ragas borrows.
- **Pass@1**: fraction of tasks solved correctly on the first attempt (BigCodeBench).
- **Log-likelihood evaluation**: scoring MCQs by comparing option probabilities instead of generating text.
- **Training-serving skew**: a mismatch in preprocessing between training and serving.

---

## 🔗 Next Session

**Session 10.1 (book Chapter 10)**: see `session_5.3_sagemaker_deployment.md` and `session_10.1_deployment_topologies.md`.

---

## 📚 Additional Resources

- [Ragas documentation](https://docs.ragas.io/)
- [ARES: An Automated Evaluation Framework for RAG (Saad-Falcon et al.)](https://arxiv.org/abs/2311.09476)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)](https://arxiv.org/abs/2309.15217)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [lighteval](https://github.com/huggingface/lighteval)
- [Open Arabic LLM Leaderboard](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard)
- [Hallucinations Leaderboard (Hong et al., 2024)](https://arxiv.org/abs/2404.05904)
- [GAIA: a benchmark for General AI Assistants (Mialon et al., 2023)](https://arxiv.org/abs/2311.12983)
- [Measuring Massive Multitask Language Understanding (Hendrycks et al., 2020)](https://arxiv.org/abs/2009.03300)
- [Instruction-Following Evaluation for LLMs (Zhou et al., 2023)](https://arxiv.org/abs/2311.07911)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 3.1, 5.1, 5.2, 7.3

**Outcome**: You can choose and run the right evaluation for models and RAG systems, and you know how Ragas and ARES differ.
