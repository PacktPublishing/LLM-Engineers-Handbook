from sentence_transformers import SentenceTransformer

# Leverage the Poetry virtual environment to run the code:
# poetry run python code_snippets/08_text_embeddings.py

if __name__ == "__main__":
    # 1. Load a pretrained Sentence Transformer model.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # The sentences to encode.
    sentences = ["The dog sits outside waiting for a treat.", "I am going swimming.", "The dog is swimming."]

    # 2. Calculate embeddings.
    embeddings = model.encode(sentences)
    print(embeddings.shape)  # noqa
    # Output: [3, 384]

    # 3. Calculate the embedding similarities using cosine similarity.
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)  # noqa
    # Output:
    # tensor([[ 1.0000, -0.0389,  0.2692],
    #     [-0.0389,  1.0000,  0.3837],
    #     [ 0.2692,  0.3837,  1.0000]])
    #
    # similarities[0, 0] = The similarity between the first sentence and itself.
    # similarities[0, 1] = The similarity between the first and second sentence.
    # similarities[2, 1] = The similarity between the third and second sentence.
