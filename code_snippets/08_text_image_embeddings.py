from io import BytesIO

import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    # Get the image
    response = requests.get(
        "https://github.com/PacktPublishing/LLM-Engineering/blob/main/images/crazy_cat.jpg?raw=true"
    )
    image = Image.open(BytesIO(response.content))

    # Load CLIP model

    model = SentenceTransformer("clip-ViT-B-32")

    # Encode an image:

    img_emb = model.encode(image)

    # Encode text descriptions

    text_emb = model.encode(["Two dogs in the snow", "A cat on a table", "A picture of London at night"])

    # Compute similarities

    similarity_scores = model.similarity(img_emb, text_emb)

    print(similarity_scores)  # noqa
