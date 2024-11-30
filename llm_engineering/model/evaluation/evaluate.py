import concurrent.futures
import gc
import json
import os

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from openai import OpenAI
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DATASET_HUGGINGFACE_WORKSPACE = os.environ["DATASET_HUGGINGFACE_WORKSPACE"]
MODEL_HUGGINGFACE_WORKSPACE = os.environ["MODEL_HUGGINGFACE_WORKSPACE"]
IS_DUMMY = os.environ.get("IS_DUMMY", False)

print("====== EVAL PARAMETERS ======")  # noqa
print(f"{DATASET_HUGGINGFACE_WORKSPACE=}")  # noqa
print(f"{MODEL_HUGGINGFACE_WORKSPACE=}")  # noqa
print(f"{IS_DUMMY=}")  # noqa
print("=============================")  # noqa


def generate_answers(model_id: str, dataset_name: str):
    def format(sample):
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n".format(
            sample["instruction"]
        )

    dataset = load_dataset(dataset_name, split="test")
    if IS_DUMMY:
        try:
            dataset = dataset.select(range(10))
        except Exception:
            print("Dummy mode active. Failed to trim the dataset to 10 samples.")  # noqa
    print(f"Dataset size: {len(dataset)}")  # noqa
    dataset = dataset.map(lambda sample: {"prompt": format(sample)})

    print(f"Generating answers for {model_id}")  # noqa
    llm = LLM(model=model_id, max_model_len=2048)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_p=0.05, max_tokens=2048)
    outputs = llm.generate(dataset["prompt"], sampling_params)

    answers = [output.outputs[0].text for output in outputs]
    dataset = dataset.add_column("answers", answers)

    print(f"Uploading results for {model_id}")  # noqa
    dataset.push_to_hub(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results")
    gc.collect()

    return dataset


def evaluate_answer(instruction: str, answer: str, client: OpenAI) -> dict:
    prompt = f"""You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:
1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

Accuracy scale:
1 (Poor): Contains factual errors or misleading information
2 (Good): Mostly accurate with minor errors or omissions
3 (Excellent): Highly accurate and comprehensive

Style scale:
1 (Poor): Too formal, uses some overly complex words
2 (Good): Good balance of technical content and accessibility, but still uses formal words and expressions
3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.

Instruction: {instruction}

Answer: {answer}

Provide your evaluation in JSON format with the following structure:
{{
    "accuracy": {{
        "analysis": "...",
        "score": 0
    }},
    "style": {{
        "analysis": "...",
        "score": 0
    }}
}}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who evaluates answers based on accuracy and style. Provide your response in JSON format with a short analysis and score for each criterion.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        temperature=0.9,
    )

    # Parse the structured output
    return json.loads(completion.choices[0].message.content)


def evaluate_batch(batch, start_index):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return [(i, evaluate_answer(instr, ans, client)) for i, (instr, ans) in enumerate(batch, start=start_index)]


def evaluate_answers(model_id: str, num_threads: int = 10, batch_size: int = 5) -> Dataset:
    # Load the dataset
    dataset = load_dataset(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results", split="all")

    # Create batches of instruction-answer pairs with their original indices
    batches = [
        (i, list(zip(dataset["instruction"][i : i + batch_size], dataset["answers"][i : i + batch_size], strict=False)))
        for i in range(0, len(dataset), batch_size)
    ]

    evaluations = [None] * len(dataset)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(evaluate_batch, batch, start_index) for start_index, batch in batches]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            for index, evaluation in future.result():
                evaluations[index] = evaluation

    # Replace the 'evaluation' column if it exists, otherwise add it
    if "evaluation" in dataset.column_names:
        dataset = dataset.remove_columns(["evaluation"])
    dataset = dataset.add_column("evaluation", evaluations)

    # Post-process evaluations
    accuracy_scores = []
    style_scores = []

    for evaluation in dataset["evaluation"]:
        try:
            eval_dict = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
            accuracy_score = eval_dict["accuracy"]["score"]
            style_score = eval_dict["style"]["score"]

            accuracy_scores.append(accuracy_score)
            style_scores.append(style_score)

        except (json.JSONDecodeError, KeyError, TypeError):
            # If there's an error, append None to maintain alignment
            accuracy_scores.append(None)
            style_scores.append(None)

    # Add new columns to the dataset
    if "accuracy" in dataset.column_names:
        dataset = dataset.remove_columns(["accuracy"])
    dataset = dataset.add_column("accuracy", accuracy_scores)
    if "style" in dataset.column_names:
        dataset = dataset.remove_columns(["style"])
    dataset = dataset.add_column("style", style_scores)

    dataset.push_to_hub(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results")

    return dataset


def check_if_huggingface_model_exists(model_id: str, default_value: str) -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
        print(f"Found model on HF: '{model_id}'.")  # noqa
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")  # noqa
        model_id = default_value
        print(f"Defaulting to '{model_id}'")  # noqa
        print("Train your own model to avoid this behavior.")  # noqa

    return model_id


def check_if_huggingface_dataset_exists(dataset_id: str, default_value: str) -> str:
    api = HfApi()

    try:
        api.dataset_info(dataset_id)
        print(f"Found dataset on HF: '{dataset_id}'.")  # noqa
    except RepositoryNotFoundError:
        print(f"Dataset '{dataset_id}' does not exist.")  # noqa
        dataset_id = default_value
        print(f"Defaulting to '{dataset_id}'")  # noqa
        print("Use a valid dataset or create your own to avoid this behavior.")  # noqa

    return dataset_id


model_ids = [
    check_if_huggingface_model_exists(
        f"{MODEL_HUGGINGFACE_WORKSPACE}/TwinLlama-3.1-8B", default_value="mlabonne/TwinLlama-3.1-8B"
    ),
    check_if_huggingface_model_exists(
        f"{MODEL_HUGGINGFACE_WORKSPACE}/TwinLlama-3.1-8B-DPO", default_value="mlabonne/TwinLlama-3.1-8B-DPO"
    ),
    "meta-llama/Llama-3.1-8B-Instruct",
]

if __name__ == "__main__":
    # Run generation
    for model_id in model_ids:
        dataset_name = check_if_huggingface_dataset_exists(
            f"{DATASET_HUGGINGFACE_WORKSPACE}/llmtwin", default_value="mlabonne/llmtwin"
        )
        generate_answers(model_id, dataset_name=dataset_name)

    # Run evaluation
    for model_id in model_ids:
        evaluate_answers(model_id)

    # Analyze results
    for model_id in model_ids:
        dataset = load_dataset(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results", split="all")

        score = sum(dataset["accuracy"]) / len(dataset["accuracy"])
        print(f"{model_id.split('/')[-1]} - Accuracy: {score:.2f}")  # noqa

        score = sum(dataset["style"]) / len(dataset["style"])
        print(f"{model_id.split('/')[-1]} - Style: {score:.2f}")  # noqa
