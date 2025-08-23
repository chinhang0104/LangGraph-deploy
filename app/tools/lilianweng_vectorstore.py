import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_vectorstore() -> VectorStore:
    embedding = HuggingFaceEmbeddings()
    
    # Get the directory of the current file or use the base app directory
    base_dir = os.path.dirname(os.path.abspath(__file__))  # this gives /code/app/tools
    app_dir = os.path.abspath(os.path.join(base_dir, '..'))  # move one level up to /code/app
    persist_directory = os.path.join(app_dir, 'lilianwen_db')
    
    # Create vectorstore if not exist, else load vectorstore
    if not os.path.exists(persist_directory):
        # Download data
        urls = [
            "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        ]
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split text
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Store text in vector store
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
    else:
        # Load vectorstore from path
        vectorstore = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
    return vectorstore