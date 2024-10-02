from llm_engineering.domain.documents import ArticleDocument, UserDocument

if __name__ == "__main__":
    user = UserDocument.get_or_create(first_name="Paul", last_name="Iusztin")
    articles = ArticleDocument.bulk_find(author_id=str(user.id))

    print(f"User ID: {user.id}")  # noqa
    print(f"User name: {user.first_name} {user.last_name}")  # noqa
    print(f"Number of articles: {len(articles)}")  # noqa
    print("First article link:", articles[0].link)  # noqa
