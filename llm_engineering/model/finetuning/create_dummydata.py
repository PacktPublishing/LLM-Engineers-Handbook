import pandas as pd
from datasets import Dataset, DatasetDict

# Create a small dummy dataset
data = {
    "instruction": [
        "Explain what a neural network is.",
        "What is the capital of France?",
        "Describe the process of photosynthesis.",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the formula for the area of a circle?",
    ],
    "output": [
        "A neural network is a type of machine learning model inspired by the human brain...",
        "The capital of France is Paris.",
        "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
        "'To Kill a Mockingbird' was written by Harper Lee.",
        "The formula for the area of a circle is A = πr², where r is the radius of the circle.",
    ],
}

# Create a DataFrame
df = pd.DataFrame(data)


# Create a function to format the data
def format_data(example):
    return {"text": f"User: {example['instruction']}\n\nAssistant: {example['output']}"}


# Create a Dataset
dataset = Dataset.from_pandas(df)

# Apply the formatting
dataset = dataset.map(format_data, remove_columns=dataset.column_names)

# Split the dataset into train and test
dataset = dataset.train_test_split(test_size=0.2)

# Create a DatasetDict
dataset_dict = DatasetDict({"train": dataset["train"], "test": dataset["test"]})

# Save the dataset
dataset_dict.save_to_disk("dummy_dataset")
