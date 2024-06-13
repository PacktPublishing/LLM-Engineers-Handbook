from typing import Optional

from .base import VectorDBDataModel
from .types import DataCategory


class PostCleanedModel(VectorDBDataModel):
    content: str
    platform: str
    author_id: str
    image: Optional[str] = None

    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


class ArticleCleanedModel(VectorDBDataModel):
    content: str
    platform: str
    link: str
    author_id: str

    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class RepositoryCleanedModel(VectorDBDataModel):
    content: str
    name: str
    link: str
    author_id: str

    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False
