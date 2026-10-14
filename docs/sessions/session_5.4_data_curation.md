# Session 5.4: Instruction Data Curation (Book Chapter 5)

## 🎯 Learning Objectives

By the end of this session, you will:
- Judge instruction data on accuracy, diversity, and complexity
- Size a dataset for general-purpose, task-specific, and domain-specific models
- Apply rule-based filtering, exact/fuzzy/semantic deduplication, and decontamination
- Evaluate data quality with LLM judges, reward models, and classifiers
- Explore datasets manually, statistically, and with topic clustering
- Generate and augment data (Evol-Instruct, UltraFeedback)
- Map every stage of the book's pipeline onto the repo's actual code

> This session covers **Book Chapter 5's data-engineering half** (pp. 207-225). `session_3.1_instruction_dataset.md` covers the project's hands-on generation; this session covers the curation pipeline that determines whether that data is any good. The project's own generation run (pp. 225-234) is covered there, and the preference-side differences (Chapter 6) are in `session_3.2_preference_dataset.md`.

---

## 🏗️ Architecture Overview

```
Post-training data pipeline
  Seed/raw text
      │
  (1) Curate      task-specific vs domain-specific; few-shot alternative
      │
  (2) Filter      rule-based (length, keywords, format)
      │
  (3) Deduplicate exact → fuzzy (MinHash) → semantic (embeddings)
      │
  (4) Decontaminate  remove anything near the test set
      │
  (5) Evaluate quality  LLM judge / reward model / classifier
      │
  (6) Explore     manual, statistical, topic clustering
      │
  (7) Generate    seed prompts → synthetic pairs (structured output)
      │
  (8) Augment     Evol-Instruct (instructions), UltraFeedback (answers)
      ▼
  High-quality instruction dataset
```

Each stage is independently skippable. The book's recommendation is to **select the stages relevant to your use case** rather than run all eight every time. The LLM Twin runs a reduced set: it generates (Chapter 5), lightly filters, and defers augmentation to the preference stage (Chapter 6).

### Where each stage lives in the repo

| Stage | Repo artifact | Notes |
|-------|---------------|-------|
| Filter (rule-based) | `application/dataset/utils.py`: `filter_short_answers`, `filter_answer_format` | applied to preference data |
| Dedup (exact) | chunk IDs are `md5` of content (Session 2.2) | idempotent upsert, not a pass |
| Generate | `application/dataset/generation.py`; `steps/generate_datasets/*` | Session 3.1 |
| Evaluate | `model/evaluation/evaluate.py` | LLM-as-a-judge, pointwise (Session 7.3) |
| Explore | not automated | manual + Hub viewer |
| Augment | not implemented | book-level technique |

---

## 📁 The General Framework

An instruction dataset is pairs of `instruction` and `answer`. Templates like **Alpaca** add `system` (a meta-prompt steering behavior) and `input` (data the task needs). During fine-tuning you can train on instruction+answer or answer-only.

The book's canonical example (Table 5.1) is from Open-Orca/SlimOrca:

```
System:      You are a helpful assistant, who always provide explanation.
             Think like you are answering to a five year old.
Instruction: Concepts: building, shop, town
             Write a sentence that includes all these words.
Output:      In our little town, there is a shop inside a big building where
             people go to buy their favorite toys and candies.
```

**Three quality dimensions**:
- **Accuracy**: factually correct and relevant to the instruction.
- **Diversity**: covers the range of topics, contexts, lengths, and styles the model will see.
- **Complexity**: non-trivial, multi-step, challenging samples that push capability.

These three axes are the scorecard for every later stage. Filtering targets accuracy; dedup targets diversity; augmentation targets complexity.

### Data Quantity

| Regime | Typical samples | Example |
|--------|-----------------|---------|
| Too few | < ~1,000 | augment with open data |
| Large model (~70B) | as few as ~1,000 | LIMA |
| General-purpose | ≥ 1,000,000 | OpenHermes, Dolphin (~1M) |
| Llama 3 (full pipeline) | ~10,000,000 | SFT + alignment |
| Yi | < 10,000 | 01-ai |
| Task-specific | 100 - 100,000 | translation, summarization, sentiment |
| Domain-specific | highly variable | medicine/law approach general scale |

**Key nuance**: sample count is a function of *model size* and *data quality*, not a fixed number. A 70B model can be steered with 1,000 excellent samples (LIMA); a 7B model needs more just to learn the chat template. The LLM Twin (8B) generates 3,335 pairs and later upsamples with 10,000 FineTome samples during SFT.

---

## 📁 Data Curation

- **Task-specific**: collect examples of the task from existing datasets or create new ones (original/summary pairs, parallel sentences).
- **Domain-specific**: harder; often requires subject-matter experts, research papers, and technical documents; sometimes organizational partnerships.
- **Few-shot prompting** is an alternative to fine-tuning for task-specific work when you do not need a new domain. The line between task- and domain-specific often blurs (medical diagnosis is both).

**Why this stage matters**: the data you curate fixes the ceiling for every later stage. No filter, judge, or augmentation technique can add knowledge that was never in the corpus.

---

## 📁 Rule-Based Filtering

Explicit, predefined rules for fast, transparent, scalable filtering:

- **Length filtering**: drop too-short (uninformative) and too-long (redundant) responses; thresholds are task-dependent. A summarization dataset wants a low max; a detailed-explanation dataset wants a high one.
- **Keyword exclusion**: drop samples containing profanity, spam, slang, or off-domain terms. In a professional-writing assistant, exclude slang and informal expressions.
- **Format checking**: enforce structure (code syntax, JSON validity, style). Especially important for code, JSON, and other formatted text.

**Pros**: fast, consistent, transparent, automatable, cheap to monitor continuously.

| Pros | Cons |
|------|------|
| Fast and scalable to millions | No nuance; removes valid but unusual samples |
| Consistent, uniform treatment | Binary pass/fail clashes with language nuance |
| Transparent and auditable | Rules need regular maintenance as standards evolve |
| Automatable, continuous | Poorly designed rules can amplify bias |

**In the repo**: Session 3.2's `filter_short_answers` (min 100 chars) and `filter_answer_format` (capitalized, ends with `. ! ?`) are exactly this pattern applied to preference data. Both are pure functions over `chosen`, they drop whole samples, and they run before the split. Acceptance rates are logged.

**Tradeoff to internalize**: the book's own preference run drops roughly half the samples (2,970 -> 1,467). Aggressive filtering is a feature, not a failure — but you must monitor retained counts so a small category does not empty out.

---

## 📁 Data Deduplication

Duplicates cause four concrete harms:

1. **Overfitting** — the model memorizes duplicates instead of learning patterns.
2. **Biased performance** — over-represented inputs skew behavior.
3. **Inefficient training** — wasted compute on redundant data.
4. **Inflated evaluation metrics** — duplicates in test sets produce optimistic scores.

Three levels, cheapest first:

### 1. Exact deduplication

Normalize (e.g. lowercase), hash (MD5 or SHA-256), compare hashes, drop matches. Fast but blind to near-duplicates.

```python
import hashlib

def exact_dedup(samples: list[str]) -> list[str]:
    seen, kept = set(), []
    for s in samples:
        h = hashlib.sha256(s.strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            kept.append(s)
    return kept
```

### 2. Fuzzy deduplication (MinHash)

The standard fuzzy approach: high accuracy at much lower complexity than pairwise comparison.

- Convert each text to a set of **shingles** (n-grams).
- Apply multiple hash functions; keep the **minimum** hash per function to form a **signature**.
- Compare signatures with **Jaccard similarity**: two sets are near-duplicates when their signatures agree above a threshold.

```python
from datasketch import MinHash, MinHashLSH

def minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in {text[i:i + 5] for i in range(max(1, len(text) - 4))}:
        m.update(shingle.encode())
    return m
```

**Why MinHash works**: the probability that two sets share a minimum hash under a random permutation equals their Jaccard similarity. Taking minima over many hash functions estimates Jaccard without ever computing the full overlap.

### 3. Semantic deduplication

Embed samples and compare meaning, not surface form.

- Word-level: Word2Vec, GloVe, FastText.
- Sentence/document-level: BERT, sentence-transformers, cross-encoders.
- Compare with cosine similarity or Euclidean distance; samples above a threshold are duplicates.
- At scale, cluster with K-means / DBSCAN / hierarchical clustering and keep one representative per cluster.

**In the repo**: chunk IDs are `md5` of content (Session 2.2), which makes exact-duplicate chunks collide to the same ID and upsert idempotently. This is exact dedup by construction — it does **not** catch near-duplicates.

| Level | Detects | Cost | Miss |
|-------|---------|------|------|
| Exact | identical after normalization | O(n) hashing | near-duplicates |
| Fuzzy (MinHash) | high Jaccard overlap | O(n) signatures | semantic paraphrases |
| Semantic | meaning-equivalent | O(n) embeddings + ANN | opposite-meaning with shared words |

---

## 📁 Data Decontamination

Ensure the training set does not overlap the evaluation/test set. This is dedup applied *across splits*, and it is what makes benchmark numbers trustworthy.

- Use **exact matching** (hash/string) to drop identical samples.
- Use **near-duplicate detection** (MinHash, n-grams, embeddings) to drop highly similar samples.
- Use **provenance tracking** to exclude data from sources known to be in the eval set (overlapping phrases, similar sentence structures, common metadata).

**Efficient trick** (book, p. 214): add your evaluation set into the deduplication stage, deduplicating only the instruction side. Record eval indexes and filter the first duplicate. When you iterate over several benchmark versions, automation pays off.

This prevents the implausible benchmark jumps the book warns about in Chapter 7 (see `session_7.4_rag_evaluation.md`). An 8-10 point MMLU jump after fine-tuning usually means leakage, not learning.

```python
test_hashes = {hashlib.sha256(t.strip().lower().encode()).hexdigest() for t in test_set}
train_clean = [
    s for s in train_set
    if hashlib.sha256(s.strip().lower().encode()).hexdigest() not in test_hashes
]
```

---

## 📁 Data Quality Evaluation

Four tiers, from most expensive to cheapest:

### Human annotation

Highest accuracy, most expensive, does not scale. Reserve it for calibrating automated judges and for subjective dimensions automation cannot capture.

### LLM-as-a-judge

Prompt an LLM to judge each sample. Flexible and scalable.

- **Comparative** ("Is answer A better than answer B?") generally beats **absolute** scoring ("Rate 1-4").
- Use a simple grading scale (few-shot examples), and aim for ~80% agreement with humans for chatbots.
- For domain-specific data, a domain-specific judge may beat a stronger general model.

The book's judge prompt (Table 5.2) asks for feedback plus a 1-4 score:

```
You are a data quality evaluator. Your goal is to assess an instruction and its
corresponding answer, determining how effectively the answer addresses the task.
In your evaluation, you will provide feedback detailing the strengths and weaknesses
of the answer, followed by a score on a scale of 1 to 4.
A score of 1 means that the answer is terrible and irrelevant to the instruction.
A score of 2 means that the answer is not helpful and misses important aspects of the instruction.
A score of 3 means that the answer is helpful but could be improved in terms of relevance, accuracy, and depth.
A score of 4 means that the answer is excellent and fully addresses the task.
Provide your evaluation as follows:
Feedback: (strengths and weaknesses you find relevant)
Score: (number between 1 and 4)
```

**Three biases and their fixes**:

| Bias | Behavior | Mitigation |
|------|----------|------------|
| Position | favors the first answer in pairwise mode | randomize A/B order |
| Length | favors longer answers | length normalization; few-shot balance |
| Intra-model (family) | favors its own model family | jury of multiple models |

Using a **jury** of smaller LLMs can reduce cost while improving accuracy and mitigating family bias.

### Reward models

A linear (regression + gating) head on a decoder-only base (Gemma, Llama). Takes an instruction/answer pair and returns a score.

- Example: **RLHFlow/ArmoRM-Llama3-8B-v0.1** adds regression and gating layers on Llama 3 8B and outputs multiple dimension scores: helpfulness, correctness, coherence, complexity, verbosity.
- Compare candidates on **allenai/reward-bench** (Hugging Face), which mixes generative, classifier, and DPO reward models on curated chosen/rejected pairs.

The term comes from RLHF (Chapter 6); a reward model is a proxy for human preference.

### Classifiers (encoder-only)

Add a classification head to an embedding encoder and train it to predict quality.

- Example: **HuggingFaceFW/fineweb-edu-classifier** is a head on `Snowflake/snowflake-arctic-embed-m`, trained 20 epochs on 450,000 samples annotated by Llama 3 70B Instruct.
- Pros: smaller, faster, scales to millions, ideal for filtering outliers in an automated pipeline.
- Cons: less accurate on complex reasoning where nuance matters.

| Method | Accuracy | Cost | Scale | Best for |
|--------|----------|------|-------|----------|
| Human | highest | highest | tiny | calibration, subjective nuance |
| LLM judge (jury) | high | medium | large | general quality, pairwise preference |
| Reward model | high | medium | large | multi-dimension scoring |
| Classifier | moderate | lowest | huge | fast bulk filtering, outlier removal |

**In the repo**: `model/evaluation/evaluate.py` is a pointwise LLM-as-a-judge (accuracy/style); `session_7.3_model_evaluation.md` documents its biases.

---

## 📁 Data Exploration

### Manual

Stratified sampling (select diverse samples), systematic review with a checklist, and collaborative review with multiple reviewers. Finds formatting errors, data-entry mistakes, incoherent reasoning, and factual errors that automation misses. **Argilla** is the book's example of a collaborative platform.

### Statistical

NLTK/spaCy for tokenization and analysis; Matplotlib/Seaborn for histograms and word clouds. Reveals vocabulary diversity, bias, and concept representation, plus possible cultural or contextual preferences that leak into model outputs.

### Topic clustering

Embed texts, reduce dimensions (UMAP), cluster (DBSCAN), and label clusters with an LLM. Reveals themes and imbalance.

- **Example from the book**: an instruction dataset about programming languages. Clustering first identifies languages (Python, JavaScript), then sub-topics within each (error handling, data structures, web frameworks). This exposes imbalance so you can balance coverage across language × sub-topic.
- Tools: Hugging Face **text-clustering** (sentence-transformers + UMAP + DBSCAN + LLM labels), **Nomic Atlas**, **BunkaTopics**, **Lilac**.

---

## 📁 Data Generation

Start from a **taxonomy of seed prompts**. The book lists five Alpaca seeds (Table 5.3), e.g.:

- "Is there anything I can eat for breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?"
- "What is the relation between the given pairs? Input: Night : Day :: Right : Left"
- "Generate a one-sentence description for each of the following people. Input: -Barack Obama\n- Elon Musk\n- Taylor Swift"

**Multi-step pipelines** improve quality: generate instructions, then answers, then **validate** with rules or another model.

**Control attributes**: complexity, length, tone, and topic. Structured generation with libraries like **Outlines** (or OpenAI JSON mode) enforces a schema.

The book's project section (p. 225) is explicit about the problem: raw articles are unstructured and limited in number.

- **Unstructured -> pairs**: use backtranslation + rephrasing with an LLM. Provide the raw chunk as the answer, generate the instruction, rephrase to imitate the author's style.
- **Limited count -> multiply**: split articles into chunks and generate **five** pairs per chunk. (This is exactly the repo's `extract_substrings` + "generate five" prompt.)
- **Structured output**: use OpenAI JSON mode (`response_format={"type": "json_object"}`) so parsing is reliable.

The book notes that even with JSON mode, a `response_format` of `json_object` still requires the word "json" in the prompt context for OpenAI to honor it.

**Risks**: inheriting the generator's biases and errors; producing too-simple or repetitive data. Mitigate with human oversight, diverse prompts, and filtering.

**In the repo**: Session 3.1 generates 5 pairs per extract with `gpt-4o-mini` and `ListPydanticOutputParser`; `constants.py` provides mocked responses for tests.

---

## 📁 Data Augmentation

Augmentation increases **quality** (diversity, complexity) using **pre-existing** samples as inputs. Unlike generation, it does not start from nothing.

### Evol-Instruct (instructions)

Evolves simple instructions into more qualitative ones; evolved instructions then generate answers with a strong LLM. Two strategies:

**In-depth** (complexity):
- **Constraints**: add requirements or limitations.
- **Deepening**: replace shallow questions with deeper ones.
- **Concretizing**: replace general concepts with specific ones.
- **Increasing reasoning steps**: explicitly request multi-step reasoning.
- **Complicating input**: add complex formats (XML, JSON, code).

**In-breadth** (diversity): generate entirely new instructions inspired by existing ones, targeting rare/long-tailed cases in the same domain.

The book's AutoEvol "Instruction Rewriter" prompt automates in-depth evolution in four steps and limits the rewrite to **+10-20 words**:

```
You are an Instruction Rewriter that rewrites the given #Instruction# into a more
complex version. Please follow the steps below...
Step 1: list all the possible methods to make this instruction more complex...
Step 2: create a comprehensive plan based on the #Methods List#...
Step 3: execute the plan and provide the #Rewritten Instruction#.
        #Rewritten Instruction# can only add 10 to 20 words...
Step 4: review and provide the #Finally Rewritten Instruction# without explanation.
```

### UltraFeedback (answers)

Evolves **answers** instead of instructions. A large pool of instructions and models generates varied responses; a strong model (GPT-4) critiques and scores them across instruction-following, truthfulness, honesty, and helpfulness. This produces preference-style data from answer variation.

| Method | Evolves | Targets | Output feeds |
|--------|---------|---------|--------------|
| Evol-Instruct | instructions | complexity + diversity | SFT pairs |
| UltraFeedback | answers | answer quality | preference pairs |

For a preference dataset, UltraFeedback maps directly onto the chosen/rejected pairs of `session_3.2_preference_dataset.md`.

---

## 🛠️ Hands-On: Deduplicate and Decontaminate

### Step 1: Exact dedup with hashing

```python
import hashlib

def exact_dedup(samples: list[str]) -> list[str]:
    seen, kept = set(), []
    for s in samples:
        h = hashlib.sha256(s.strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            kept.append(s)
    return kept
```

### Step 2: Fuzzy dedup with MinHash

```python
from datasketch import MinHash, MinHashLSH

def minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in {text[i:i+5] for i in range(max(1, len(text) - 4))}:
        m.update(shingle.encode())
    return m
```

Build an LSH index and query for near-duplicates above a Jaccard threshold:

```python
def find_near_duplicates(texts: list[str], threshold: float = 0.8) -> list[tuple[int, int]]:
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, t in enumerate(texts):
        lsh.insert(i, minhash(t))
    pairs = []
    for i, t in enumerate(texts):
        for j in lsh.query(minhash(t)):
            if i < j:
                pairs.append((i, j))
    return pairs
```

### Step 3: Decontaminate against the test set

```python
test_hashes = {hashlib.sha256(t.strip().lower().encode()).hexdigest() for t in test_set}
train_clean = [s for s in train_set if hashlib.sha256(s.strip().lower().encode()).hexdigest() not in test_hashes]
```

### Step 4: Judge a sample batch with an LLM

```python
JUDGE_PROMPT = """You are a data quality evaluator. Assess the instruction and answer.
Score 1 (terrible) to 4 (excellent). Respond as:
Feedback: ...
Score: ..."""

def judge(instruction: str, answer: str, client) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Instruction: {instruction}\nAnswer: {answer}"},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content
```

---

## 📝 Exercise 1: Audit a Dataset

### Task

Take the project's generated instruction dataset and audit it.

1. Compute length statistics (mean, p95, min, max) and plot a histogram.
2. Run exact dedup and report how many duplicates were removed.
3. Run topic clustering (or a keyword tally) to check balance across topics.
4. Sample 20 rows manually and list defects (formatting, factual issues).
5. Propose two rules to add to `filter_*` functions.

**Goal**: Experience the difference between "generated" data and "curated" data.

---

## 📝 Exercise 2: Build a Decontamination Gate

### Task

Make decontamination a reusable, testable function that mirrors the book's "add the eval set to the dedup stage" trick.

1. Write `decontaminate(train: list[str], test: list[str], mode: str)` supporting `"exact"` and `"minhash"`.
2. For exact mode, keep only train items whose hash is absent from the test hashes.
3. For minhash mode, drop any train item whose MinHash Jaccard against **any** test item exceeds a threshold.
4. Assert the result contains no test item exactly (sanity) and report how many train items were removed in each mode.
5. Explain why the minhash mode removes *more* items than exact mode, and whether that is desirable.

```python
def decontaminate(train, test, mode="exact", threshold=0.8):
    if mode == "exact":
        return exact_decontaminate(train, test)
    elif mode == "minhash":
        return minhash_decontaminate(train, test, threshold)
    raise ValueError(mode)
```

**Goal**: Turn a one-off snippet into pipeline-ready code, and see the precision/recall tradeoff between exact and fuzzy decontamination.

---

## 🐛 Common Pitfalls

- **Skipping decontamination**: an 8-10 point MMLU jump after fine-tuning usually means leakage, not learning.
- **Exact-only dedup**: near-duplicates still leak into test sets. Add fuzzy/semantic matching.
- **Single judge**: intra-model favoritism biases results. Use a jury.
- **Over-filtering**: aggressive length/keyword rules can empty small categories; monitor retained counts (the book's run keeps ~49%).
- **Synthetic echo chamber**: generated data inherits the generator's biases; always add human review and diversity controls.
- **Confusing task-specific with domain-specific**: the first needs task examples (few-shot can suffice); the second needs domain knowledge and often subject-matter experts.
- **Length bias in the judge**: a verbose answer can outscore a correct concise one. Normalize or use balanced few-shot examples.
- **Position bias in pairwise judging**: the first answer wins more often. Always randomize.
- **Treating a benchmark jump as progress**: verify the test set is disjoint before celebrating.
- **MinHash parameters drift**: shingle size and `num_perm` change the false-positive rate; fix them and document the choice.

---

## 🎓 Knowledge Check

1. **What three qualities define high-quality instruction data?**
   - Accuracy, diversity, complexity.

2. **How many samples does a general-purpose instruct model typically need?**
   - About one million or more; task-specific is 100-100,000.

3. **What algorithm is standard for fuzzy deduplication?**
   - MinHash (with Jaccard similarity), typically accelerated with LSH.

4. **What is decontamination and why does it matter?**
   - Removing training samples near the test set; it prevents inflated, leak-driven benchmark scores.

5. **Name two biases of LLM-as-a-judge and their mitigations.**
   - Position (randomize order) and intra-model/family favoritism (jury of models); length bias is mitigated by normalization.

6. **What does Evol-Instruct do?**
   - Evolves instructions to be more complex (in-depth) or more diverse (in-breadth).

7. **What does UltraFeedback evolve, and how does it differ from Evol-Instruct?**
   - It evolves answers using AI feedback; Evol-Instruct evolves instructions.

8. **Why can a 70B model be steered with ~1,000 samples?**
   - Larger models are more sample-efficient (LIMA); smaller models need more just to learn the chat template.

9. **What is the difference between exact, fuzzy, and semantic deduplication?**
   - Exact matches identical (normalized) strings; fuzzy catches high Jaccard overlap; semantic catches meaning-equivalent paraphrases.

10. **What is the book's efficient decontamination trick?**
    - Add the evaluation set into the dedup stage and filter only the instruction-side duplicates (record eval indexes).

11. **Why prefer comparative judging over absolute scoring?**
    - It correlates better with human judgment and mirrors how humans compare.

12. **What does `fineweb-edu-classifier` demonstrate?**
    - An encoder-only classifier head on an embedding model, trained on 450k LLM-annotated samples, that scales quality filtering to millions cheaply.

13. **What is a reward model, and how is it built?**
    - A linear/regression head on a decoder-only base (e.g. Llama) that scores instruction/answer pairs; compared on `allenai/reward-bench`.

14. **What are the two Evol-Instruct strategies?**
    - In-depth (constraints, deepening, concretizing, more reasoning, complex input) and in-breadth (new long-tail instructions).

15. **Which repo functions already implement rule-based filtering?**
    - `filter_short_answers` and `filter_answer_format` in `application/dataset/utils.py`.

---

## 📖 Glossary

- **Alpaca template** — an instruction/input/response format (plus optional system) widely used for SFT.
- **Contamination** — overlap between training and evaluation data that inflates metrics.
- **Decontamination** — removing training samples that match or resemble evaluation samples.
- **Evol-Instruct** — evolving instructions in-depth (complexity) and in-breadth (diversity).
- **Jaccard similarity** — |A ∩ B| / |A ∪ B|; the similarity MinHash estimates.
- **LLM-as-a-judge** — using an LLM to score or compare samples.
- **MinHash** — a signature scheme that estimates Jaccard similarity cheaply.
- **Reward model** — a model returning a scalar (or multi-dimension) quality score for a pair.
- **Shingle** — an n-gram of characters or tokens used as a set element in MinHash.
- **Taxonomy (seed prompts)** — a curated set of prompts used to seed synthetic generation.
- **UltraFeedback** — generating varied answers and scoring them with a strong model.
- **Stratified sampling** — sampling to preserve category proportions.

---

## 🔗 Next Session

**Session 3.1 and 3.2**: apply this curation pipeline to the project's own dataset generation.

- [`session_3.1_instruction_dataset.md`](./session_3.1_instruction_dataset.md) — the repo's instruction generation.
- [`session_3.2_preference_dataset.md`](./session_3.2_preference_dataset.md) — the repo's preference generation and the two filters.

Then [`session_7.3_model_evaluation.md`](./session_7.3_model_evaluation.md) and [`session_7.4_rag_evaluation.md`](./session_7.4_rag_evaluation.md) cover evaluation and the leakage trap decontamination prevents.

---

## 📚 Additional Resources

- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [Evol-Instruct / WizardLM](https://arxiv.org/abs/2304.12244)
- [Automatic Instruction Evolving for Large Language Models (AutoEvol)](https://arxiv.org/abs/2406.00770)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
- [datasketch (MinHash/LSH)](https://ekzhu.com/datasketch/)
- [Argilla](https://argilla.io/)
- [fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)
- [ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)
- [allenai/reward-bench](https://huggingface.co/datasets/allenai/reward-bench)
- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Sessions 3.1, 3.2

**Outcome**: You can curate, deduplicate, decontaminate, evaluate, explore, generate, and augment instruction data to a production standard, and point to the repo code for each stage.
