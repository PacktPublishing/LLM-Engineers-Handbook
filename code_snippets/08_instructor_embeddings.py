from InstructorEmbedding import INSTRUCTOR

# Create virtual environment, install dependencies and run the code:
# 1. Create: python3 -m venv instructor_venv
# 2. Activate: source instructor_venv/bin/activate
# 3. Install: pip install sentence-transformers==2.2.2 InstructorEmbedding==1.0.1
# 4. Run the code: python code_snippets/08_instructor_embeddings.py

if __name__ == "__main__":
    model = INSTRUCTOR("hkunlp/instructor-base")

    sentence = "RAG Fundamentals First"

    instruction = "Represent the title of an article about AI:"

    embeddings = model.encode([[instruction, sentence]])
    print(embeddings.shape)  # noqa
    # Output: (1, 768)
