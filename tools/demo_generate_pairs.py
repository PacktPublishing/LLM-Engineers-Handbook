"""
Quick demo to generate and view instruction-answer pairs
"""

import json
from openai import OpenAI

# Load one article
with open("data/artifacts/cleaned_documents.json", "r") as f:
    data = json.load(f)
    article = data["artifact_data"][0]  # First article

# Take small chunk
chunk = article["content"][:1500]

print("=" * 60)
print("ARTICLE CHUNK (first 1500 chars):")
print("=" * 60)
print(chunk)
print("\n")

# Generate pairs with Ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

prompt = f"""Based on the following extract, generate FIVE instruction-answer pairs.

Rules:
- Each instruction must ask to write about a specific topic in the context
- Each answer must be a relevant paragraph from the context
- Instructions must NEVER mention "context", "system", "extract"
- Instructions must be self-contained and general
- Answers must imitate the writing style

Example:
Instruction: "Explain the concept of an LLM Twin"
Answer: "An LLM Twin is essentially an AI character that mimics your writing style..."

Extract:
{chunk}

Return a valid JSON object with this structure:
{{
  "instruction_answer_pairs": [
    {{"instruction": "...", "answer": "..."}},
    {{"instruction": "...", "answer": "..."}}
  ]
}}"""

print("=" * 60)
print("GENERATING PAIRS WITH OLLAMA...")
print("=" * 60)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant who generates instruction-answer pairs."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    max_tokens=1200,
)

result = response.choices[0].message.content
print(result)
print("\n")

# Try to parse JSON
try:
    pairs = json.loads(result)
    print("=" * 60)
    print("GENERATED PAIRS (formatted):")
    print("=" * 60)
    for i, pair in enumerate(pairs["instruction_answer_pairs"], 1):
        print(f"\n🔵 PAIR {i}:")
        print(f"📝 Instruction: {pair['instruction']}")
        print(f"💬 Answer: {pair['answer'][:200]}...")
        print()
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing failed: {e}")
