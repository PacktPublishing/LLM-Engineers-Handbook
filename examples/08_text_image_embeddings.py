from PIL import Image
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    # Load CLIP model

    model = SentenceTransformer("clip-ViT-B-32")

    # Encode an image:

    img_emb = model.encode(Image.open("two_dogs_in_snow.jpg"))

    # Encode text descriptions

    text_emb = model.encode(["Two dogs in the snow", "A cat on a table", "A picture of London at night"])

    # Compute similarities

    similarity_scores = model.similarity(img_emb, text_emb)

    print(similarity_scores)  # noqa
