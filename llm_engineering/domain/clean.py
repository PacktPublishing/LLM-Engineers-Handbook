from typing import Optional

from .base import VectorDBDataModel
from .types import DataCategory


class PostCleanedModel(VectorDBDataModel):
    platform: str
    cleaned_content: str
    author_id: str
    image: Optional[str] = None

    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


class ArticleCleanedModel(VectorDBDataModel):
    platform: str
    link: str
    cleaned_content: str
    author_id: str

    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class RepositoryCleanedModel(VectorDBDataModel):
    name: str
    link: str
    cleaned_content: str
    owner_id: str

    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False
