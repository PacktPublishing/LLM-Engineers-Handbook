from dotenv import load_dotenv
from loguru import logger

from langchain.globals import set_verbose

from llm_engineering.application.rag.retriever import ContextRetriever

set_verbose(True)


if __name__ == "__main__":
    query = """
        Could you please draft a LinkedIn post discussing RAG systems?
        I'm particularly interested in how RAG works and how it is integrated with vector DBs and large language models (LLMs).
        """
        
    retriever = ContextRetriever(mock=True)
    documents = retriever.search(query, k=12)

    for rank, document in enumerate(documents):
        logger.info(f"{rank}: {document}")
