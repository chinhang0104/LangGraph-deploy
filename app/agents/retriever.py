from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStore
from langchain_core.tools.simple import Tool

def get_retriever_tool(vectorstore: VectorStore, description: str, document_prompt: str) -> Tool:
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        description, #"retrieve_blog_posts",
        document_prompt, #"Search and return information about Lilian Weng blog posts.",
    )
    return retriever_tool