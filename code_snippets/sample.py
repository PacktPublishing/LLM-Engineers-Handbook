from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class DataEngineeringCopilot:
    def __init__(self, model_name="microsoft/phi-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.2,
        )

    def ask(self, prompt):
        response = self.generator(prompt, return_full_text=False)
        return response[0]['generated_text']

if __name__ == "__main__":
    copilot = DataEngineeringCopilot()
    user_prompt = (
        "Write a Python function to load a CSV file into a pandas DataFrame, "
        "clean missing values, and return the cleaned DataFrame."
    )
    answer = copilot.ask(user_prompt)
    print("Copilot suggestion:\n", answer)