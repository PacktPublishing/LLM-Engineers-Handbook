from io import BytesIO

import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

# Leverage the Poetry virtual environment to run the code:
# poetry run python code_snippets/08_text_image_embeddings.py

if __name__ == "__main__":
    # Load an image with a crazy cat.
    response = requests.get(
        "https://github.com/PacktPublishing/LLM-Engineering/blob/main/images/crazy_cat.jpg?raw=true"
    )
    image = Image.open(BytesIO(response.content))

    # Load CLIP model.
    model = SentenceTransformer("clip-ViT-B-32")

    # Encode the loaded image.
    img_emb = model.encode(image)

    # Encode text descriptions.
    text_emb = model.encode(
        [
            "A crazy cat smiling.",
            "A white and brown cat with a yellow bandana.",
            "A man eating in the garden.",
        ]
    )
    print(text_emb.shape)  # noqa
    # Output: (3, 512)

    # Compute similarities.
    similarity_scores = model.similarity(img_emb, text_emb)
    print(similarity_scores)  # noqa
    # Output: tensor([[0.3068, 0.3300, 0.1719]])
