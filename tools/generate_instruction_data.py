"""
Script for generating instruction datasets from article data using LLMs.

This script loads article data from JSON files and generates instruction-format
datasets suitable for supervised fine-tuning of language models.

Pipeline Overview:
1. Load articles from JSON file into a Hugging Face Dataset
2. Clean text by removing special characters and redundant whitespace
3. Extract chunks (1000-2000 chars) from articles using sentence boundaries
4. Generate instruction-answer pairs using LLM for each chunk
5. Create final instruction dataset and push to Hugging Face Hub

Supported Providers (set via --provider flag):
- openai: OpenAI API (requires OPENAI_API_KEY) - gpt-4o-mini
- groq: Groq API (requires GROQ_API_KEY) - llama-3.1-70b-versatile (FREE!)
- together: Together AI (requires TOGETHER_API_KEY) - llama-3.1-70b ($5 free credits)
- ollama: Local Ollama (no API key needed) - llama3.1, mistral, etc. (FREE!)

Usage:
    # Using Groq (FREE - Recommended)
    export GROQ_API_KEY="gsk_..."
    python tools/generate_instruction_data.py --provider groq

    # Using Together AI (FREE $5 credits)
    export TOGETHER_API_KEY="..."
    python tools/generate_instruction_data.py --provider together

    # Using Ollama (FREE, local)
    ollama pull llama3.1
    python tools/generate_instruction_data.py --provider ollama --model llama3.1

    # Using OpenAI
    export OPENAI_API_KEY="your-openai-key"
    python tools/generate_instruction_data.py --provider openai
"""

import argparse
import concurrent.futures
import json
import os
import re
from typing import List, Tuple

from datasets import Dataset
from openai import OpenAI
from tqdm.auto import tqdm


# ============================================================================
# Provider Configuration
# ============================================================================

PROVIDER_CONFIG = {
    "openai": {
        "base_url": None,  # Uses default
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "supports_json_mode": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "supports_json_mode": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.1-70b-versatile",
        "supports_json_mode": True,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "supports_json_mode": True,
    },
    "hf": {
        "base_url": "https://api-inference.huggingface.co/v1/",
        "api_key_env": "HF_TOKEN",
        "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "supports_json_mode": False,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,  # No API key needed
        "default_model": "llama3.1:8b",
        "supports_json_mode": False,  # Most Ollama models don't support JSON mode
    },
}


def get_client(provider: str) -> OpenAI:
    """
    Get an OpenAI-compatible client for the specified provider.

    Args:
        provider: One of 'openai', 'groq', 'ollama'

    Returns:
        OpenAI: Client configured for the provider
    """
    config = PROVIDER_CONFIG[provider]

    api_key = "ollama"  # Default for Ollama (doesn't need real key)
    if config["api_key_env"]:
        api_key = os.environ.get(config["api_key_env"])
        if not api_key:
            raise ValueError(f"Please set {config['api_key_env']} environment variable for {provider}")

    return OpenAI(
        base_url=config["base_url"],
        api_key=api_key,
    )


# ============================================================================
# Step 1: Load Articles from JSON
# ============================================================================


def load_articles_from_json(file_path: str) -> Dataset:
    """
    Load articles from a JSON file and convert to a Hugging Face Dataset.

    The JSON file should have an "artifact_data" array containing article objects
    with fields: id, content, platform, author_id, author_full_name, link.

    Args:
        file_path: Path to the JSON file containing article data

    Returns:
        Dataset: A Hugging Face Dataset containing the article information
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    return Dataset.from_dict(
        {
            "id": [item["id"] for item in data["artifact_data"]],
            "content": [item["content"] for item in data["artifact_data"]],
            "platform": [item["platform"] for item in data["artifact_data"]],
            "author_id": [item["author_id"] for item in data["artifact_data"]],
            "author_full_name": [item["author_full_name"] for item in data["artifact_data"]],
            "link": [item["link"] for item in data["artifact_data"]],
        }
    )


# ============================================================================
# Step 2: Clean Text
# ============================================================================


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and normalizing whitespace.

    This function:
    1. Removes non-alphanumeric characters except apostrophes, periods,
       commas, exclamation marks, and question marks
    2. Replaces multiple consecutive whitespace characters with a single space
    3. Strips leading and trailing whitespace

    Args:
        text: The raw text to clean

    Returns:
        str: The cleaned text
    """
    # Remove non-alphanumeric characters except for apostrophes, periods,
    # commas, exclamation marks, and question marks
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    # Replace multiple consecutive whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================================
# Step 3: Extract Substrings (Chunking)
# ============================================================================


def extract_substrings(dataset: Dataset, min_length: int = 1000, max_length: int = 2000) -> List[str]:
    """
    Extract text chunks from articles in the dataset.

    This function processes each article by:
    1. Cleaning the text
    2. Splitting into sentences using regex
    3. Concatenating sentences into chunks of 1000-2000 characters

    The chunking ensures each extract has enough context for meaningful
    instruction-answer pair generation while staying within token limits.

    Args:
        dataset: Hugging Face Dataset containing articles with 'content' column
        min_length: Minimum character length for a chunk (default: 1000)
        max_length: Maximum character length for a chunk (default: 2000)

    Returns:
        List[str]: List of text chunks suitable for instruction generation
    """
    extracts = []

    # Regex pattern to split on sentence boundaries
    # Handles abbreviations and avoids splitting on "Mr.", "Dr.", etc.
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

    for article in dataset["content"]:
        # Clean the article text
        cleaned_article = clean_text(article)

        # Split into sentences
        sentences = re.split(sentence_pattern, cleaned_article)

        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add sentence to current chunk if within max_length
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                # Save current chunk if it meets minimum length
                if len(current_chunk) >= min_length:
                    extracts.append(current_chunk.strip())
                # Start new chunk with current sentence
                current_chunk = sentence + " "

        # Don't forget the last chunk
        if len(current_chunk) >= min_length:
            extracts.append(current_chunk.strip())

    return extracts


# ============================================================================
# Step 4: InstructionAnswerSet Class for JSON Parsing
# ============================================================================


class InstructionAnswerSet:
    """
    A class to manage instruction-answer pairs parsed from JSON.

    This class provides convenient methods to create instances from JSON strings
    (useful when parsing OpenAI API responses) and iterate over pairs.

    Attributes:
        pairs: List of tuples, each containing (instruction, answer)
    """

    def __init__(self, pairs: List[Tuple[str, str]]):
        """
        Initialize with a list of instruction-answer pairs.

        Args:
            pairs: List of (instruction, answer) tuples
        """
        self.pairs = pairs

    @classmethod
    def from_json(cls, json_str: str) -> "InstructionAnswerSet":
        """
        Create an InstructionAnswerSet from a JSON string.

        Expected JSON format:
        {
            "instruction_answer_pairs": [
                {"instruction": "...", "answer": "..."},
                ...
            ]
        }

        Args:
            json_str: JSON string containing instruction-answer pairs

        Returns:
            InstructionAnswerSet: Instance containing parsed pairs
        """
        data = json.loads(json_str)
        pairs = [(pair["instruction"], pair["answer"]) for pair in data["instruction_answer_pairs"]]
        return cls(pairs)

    def __iter__(self):
        """Allow iteration over the pairs."""
        return iter(self.pairs)


# ============================================================================
# Step 5: Generate Instruction-Answer Pairs using OpenAI
# ============================================================================


def generate_instruction_answer_pairs(
    extract: str,
    client: OpenAI,
    model: str,
    supports_json_mode: bool = True,
) -> List[Tuple[str, str]]:
    """
    Generate instruction-answer pairs from a text extract using an LLM.

    This function:
    1. Constructs a detailed prompt with the extract as context
    2. Calls LLM API (OpenAI-compatible)
    3. Parses the response into instruction-answer pairs

    The prompt instructs the model to:
    - Generate 5 instruction-answer pairs per extract
    - Create instructions that ask to write about specific topics
    - Provide answers that imitate the writing style of the context
    - Never mention "context", "system", "course", or "extract" in instructions

    Args:
        extract: Text chunk to generate instruction-answer pairs from
        client: OpenAI-compatible client instance
        model: Model name to use
        supports_json_mode: Whether the provider supports JSON response format

    Returns:
        List[Tuple[str, str]]: List of (instruction, answer) tuples
    """
    prompt = f"""Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \
Provide your response in JSON format with the following structure:
{{
    "instruction_answer_pairs": [
        {{"instruction": "...", "answer": "..."}},
        ...
    ]
}}

Extract:
{extract}
"""

    try:
        # Build request parameters
        request_params = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who generates instruction-answer pairs based on the given context. Provide your response in JSON format only, no other text.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1200,
            "temperature": 0.7,
        }

        # Add JSON mode if supported
        if supports_json_mode:
            request_params["response_format"] = {"type": "json_object"}

        completion = client.chat.completions.create(**request_params)

        response_text = completion.choices[0].message.content

        # For non-JSON mode, try to extract JSON from response
        if not supports_json_mode:
            # Try to find JSON in the response
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                response_text = json_match.group()

        # Parse the structured output
        result = InstructionAnswerSet.from_json(response_text)

        # Convert to list of tuples
        return result.pairs

    except Exception as e:
        print(f"Error generating pairs: {e}")
        return []


# ============================================================================
# Step 6: Create Instruction Dataset with Parallel Processing
# ============================================================================


def create_instruction_dataset(
    dataset: Dataset,
    client: OpenAI,
    model: str,
    supports_json_mode: bool = True,
    num_workers: int = 4,
) -> Dataset:
    """
    Create an instruction dataset from articles using parallel processing.

    This function:
    1. Extracts text chunks from the input dataset
    2. Uses ThreadPoolExecutor to process chunks in parallel
    3. Collects all instruction-answer pairs
    4. Returns a new Dataset with 'instruction' and 'output' columns

    Note: num_workers is set to 4 by default because higher values tend to
    exceed API rate limits, causing request failures or throttling.

    Args:
        dataset: Input Hugging Face Dataset with article content
        client: OpenAI-compatible client instance
        model: Model name to use
        supports_json_mode: Whether the provider supports JSON response format
        num_workers: Number of parallel workers (default: 4)

    Returns:
        Dataset: Hugging Face Dataset with 'instruction' and 'output' columns
    """
    # Extract chunks from articles
    extracts = extract_substrings(dataset)
    print(f"Extracted {len(extracts)} chunks from {len(dataset)} articles")

    instruction_answer_pairs = []

    # Process extracts in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(generate_instruction_answer_pairs, extract, client, model, supports_json_mode)
            for extract in extracts
        ]

        # Collect results with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Generating instruction-answer pairs"
        ):
            try:
                pairs = future.result()
                instruction_answer_pairs.extend(pairs)
            except Exception as e:
                print(f"Error processing extract: {e}")

    # Check if we got any pairs
    if not instruction_answer_pairs:
        raise ValueError("No instruction-answer pairs were generated!")

    # Unzip pairs into separate lists
    instructions, answers = zip(*instruction_answer_pairs)

    return Dataset.from_dict({"instruction": list(instructions), "output": list(answers)})


# ============================================================================
# Step 7: Main Function - Orchestrate the Pipeline
# ============================================================================


def main(
    dataset_id: str = "mlabonne/llmtwin",
    provider: str = "groq",
    model: str = None,
    num_workers: int = 4,
) -> Dataset:
    """
    Main entry point for the instruction data generation pipeline.

    This function orchestrates the entire pipeline:
    1. Initializes the LLM client for the specified provider
    2. Loads raw article data from JSON
    3. Creates instruction-answer pairs using the LLM
    4. Splits into train/test sets (90/10)
    5. Pushes the result to Hugging Face Hub

    Args:
        dataset_id: Hugging Face dataset ID for pushing results
        provider: LLM provider ('openai', 'groq', 'ollama')
        model: Model name (uses provider default if None)
        num_workers: Number of parallel workers

    Returns:
        Dataset: The final instruction dataset with train/test splits
    """
    # Get provider config
    config = PROVIDER_CONFIG[provider]
    model = model or config["default_model"]
    supports_json_mode = config["supports_json_mode"]

    print(f"Using provider: {provider}")
    print(f"Using model: {model}")
    print(f"JSON mode: {'enabled' if supports_json_mode else 'disabled'}")

    # Initialize client
    client = get_client(provider)

    # 1. Load the raw data
    print("\n" + "=" * 60)
    print("Step 1: Loading raw article data...")
    print("=" * 60)
    raw_dataset = load_articles_from_json("data/artifacts/cleaned_documents.json")
    print(f"Loaded {len(raw_dataset)} articles")
    print("\nRaw dataset preview:")
    print(raw_dataset.to_pandas().head())

    # 2. Create instruction dataset
    print("\n" + "=" * 60)
    print("Step 2: Creating ipfinetune_macnstruction dataset...")
    print("=" * 60)
    instruction_dataset = create_instruction_dataset(
        raw_dataset,
        client,
        model,
        supports_json_mode,
        num_workers,
    )
    print(f"\nGenerated {len(instruction_dataset)} instruction-answer pairs")
    print("\nInstruction dataset preview:")
    print(instruction_dataset.to_pandas().head())

    # Save raw dataset immediately (before any processing that might fail)
    print("\n" + "=" * 60)
    print("Saving raw dataset locally (backup)...")
    print("=" * 60)
    import json

    os.makedirs("data", exist_ok=True)

    # Save all pairs to CSV and JSON
    df_all = instruction_dataset.to_pandas()
    df_all = df_all.rename(columns={"output": "answer"})

    csv_path_raw = "data/generated_instruction_dataset_raw.csv"
    json_path_raw = "data/generated_instruction_dataset_raw.json"

    df_all.to_csv(csv_path_raw, index=False)

    all_pairs = [{"instruction": item["instruction"], "answer": item["answer"]} for _, item in df_all.iterrows()]
    with open(json_path_raw, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"✅ Saved {len(df_all)} pairs to '{csv_path_raw}'")
    print(f"✅ Saved {len(all_pairs)} pairs to '{json_path_raw}'")

    # 3. Train/test split and export
    print("\n" + "=" * 60)
    print("Step 3: Splitting dataset and pushing to Hub...")
    print("=" * 60)
    filtered_dataset = instruction_dataset.train_test_split(test_size=0.1)
    print(f"Train set: {len(filtered_dataset['train'])} samples")
    print(f"Test set: {len(filtered_dataset['test'])} samples")

    # Save split dataset to CSV and JSON files
    print(f"\nSaving train/test split to CSV and JSON files...")
    df_train = filtered_dataset["train"].to_pandas().rename(columns={"output": "answer"})
    df_test = filtered_dataset["test"].to_pandas().rename(columns={"output": "answer"})

    csv_path_train = "data/generated_instruction_dataset_train.csv"
    csv_path_test = "data/generated_instruction_dataset_test.csv"
    json_path_split = "data/generated_instruction_dataset.json"

    df_train.to_csv(csv_path_train, index=False)
    df_test.to_csv(csv_path_test, index=False)

    dataset_dict = {
        "train": [{"instruction": item["instruction"], "answer": item["answer"]} for _, item in df_train.iterrows()],
        "test": [{"instruction": item["instruction"], "answer": item["answer"]} for _, item in df_test.iterrows()],
    }

    with open(json_path_split, "w") as f:
        json.dump(dataset_dict, f, indent=2)

    print(f"✅ Train set saved to '{csv_path_train}'")
    print(f"✅ Test set saved to '{csv_path_test}'")
    print(f"✅ Train/test split saved to '{json_path_split}'")

    # Push to Hugging Face Hub
    print(f"\nPushing to Hugging Face Hub as '{dataset_id}'...")
    filtered_dataset.push_to_hub(dataset_id)
    print("Done!")

    return filtered_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate instruction datasets from article data using LLMs")
    parser.add_argument(
        "--dataset_id", type=str, default="mlabonne/llmtwin", help="Hugging Face dataset ID to push the results to"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["openai", "deepseek", "groq", "together", "hf", "ollama"],
        help="LLM provider to use (default: ollama - FREE!)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (uses provider default if not specified)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    main(
        dataset_id=args.dataset_id,
        provider=args.provider,
        model=args.model,
        num_workers=args.num_workers,
    )
