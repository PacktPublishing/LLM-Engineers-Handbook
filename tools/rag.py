from langchain.globals import set_verbose
from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever

set_verbose(True)


if __name__ == "__main__":
    query = """
        My name is Paul Iusztin.
        
        Could you draft a LinkedIn post discussing RAG systems?
        I'm particularly interested in:
            - how RAG works
            - how it is integrated with vector DBs and large language models (LLMs).
        """

    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=9)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document}")
